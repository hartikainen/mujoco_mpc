// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <atomic>
#include <chrono>
#include <thread>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/match.h>
#include <mujoco/mujoco.h>
#include "agent.h"
#include "array_safety.h"
#include "planners/include.h"
#include "task.h"
#include "tasks/tasks.h"
#include "threadpool.h"
#include "utilities.h"

ABSL_FLAG(std::string, task, "", "Which model to load on startup.");

namespace {
namespace mju = mujoco::util_mjpc;

// load error string length
const int kErrorLength = 1024;
const int kMaxFilenameLength = 1000;

mjModel* model = nullptr;
mjData* data = nullptr;
mjpc::Agent agent;

// --------------------------------- callbacks ---------------------------------

// sensor callback
void sensor_callback(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) {
    agent.task().Residuals(model, data, data->sensordata);
  }
}

//--------------------------------- simulation ---------------------------------

mjModel* LoadModel(std::string file) {
  // this copy is needed so that the mju::strlen call below compiles

  char filename[1024];
  mujoco::util_mjpc::strcpy_arr(filename, file.c_str());

  // make sure filename is not empty
  if (!filename[0]) {
    return nullptr;
  }

  // load and compile
  char loadError[kErrorLength] = "";
  mjModel* mnew = 0;
  if (mju::strlen_arr(filename) > 4 &&
      !std::strncmp(
          filename + mju::strlen_arr(filename) - 4, ".mjb",
          mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4)) {
    mnew = mj_loadModel(filename, nullptr);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    mnew = mj_loadXML(filename, nullptr, loadError, kMaxFilenameLength);
    // remove trailing newline character from loadError
    if (loadError[0]) {
      int error_length = mju::strlen_arr(loadError);
      if (loadError[error_length - 1] == '\n') {
        loadError[error_length - 1] = '\0';
      }
    }
  }

  if (!mnew) {
    std::printf("%s\n", loadError);
    return nullptr;
  }

  // compiler warning: print and pause
  if (loadError[0]) {
    // mj_forward() below will print the warning message
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n",
                loadError);
    exit(1);
  }

  return mnew;
}

// returns the index of a task, searching by name, case-insensitive.
// -1 if not found.
int TaskIdByName(std::string_view name) {
  int i = 0;
  for (const auto& task : mjpc::kTasks) {
    if (absl::EqualsIgnoreCase(name, task.name)) {
      return i;
    }
    i++;
  }
  return -1;
}


}  // namespace


// run event loop
int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    std::printf("MuJoCo version %s\n", mj_versionString());
    if (mjVERSION_HEADER != mj_version()) {
      mju_error("Headers and library have Different versions");
    }

    // threads
    printf("Hardware threads: %i\n", mjpc::NumAvailableHardwareThreads());

    std::string task_flag = absl::GetFlag(FLAGS_task);

    if (!task_flag.empty()) {
        agent.task().id = TaskIdByName(task_flag);
        if (agent.task().id == -1) {
            std::cerr << "Invalid --task flag: '" << task_flag << "'. Valid values:\n";
            for (const auto& task : mjpc::kTasks) {
                std::cerr << '\t' << task.name << '\n';
            }
            mju_error("Invalid --task flag.");
            return -1;
        }
    }

    const auto& task = mjpc::kTasks[agent.task().id];

    // default model + task
    std::string filename =
        mjpc::GetModelPath(mjpc::kTasks[agent.task().id].xml_path);

    // load model + make data
    model = LoadModel(filename);

    // create data
    data = mj_makeData(model);

    // sensor callback
    mjcb_sensor = &sensor_callback;

    // ----- initialize agent ----- //
    const char task_str[] = "";
    const char planners_str[] = "";
    agent.Initialize(model, data, task_str, planners_str,
                      task.residual, task.transition);

    // pool
    mjpc::ThreadPool plan_pool(1);

    // ----- settings ----- //
    std::atomic<bool> exitrequest(false);
    std::atomic<int> uiloadrequest(0);

    // ----- switch to iLQG planner ----- //
    // TODO(hartikainen):
    // assert(agent.planner_ == 2);
    agent.Allocate();
    agent.Reset();

    // ----- plan w/ iLQG planner ----- //
    agent.plan_enabled = true;
    agent.action_enabled = true;
    agent.visualize_enabled = false;
    agent.plot_enabled = false;

    // int plan_time_per_step_s = 1.0;
    float fps = 30.0;
    float simulation_duration = (float)model->nkey / fps;
    int num_timesteps = simulation_duration / model->opt.timestep;
    // int num_timesteps = model->nkey / fps;
    num_timesteps = 5;
    std::cout << "num_timesteps: " << num_timesteps << "\n";

    mj_resetData(model, data);

    // set initial qpos via keyframe
    double* key_qpos = mjpc::KeyQPosByName(model, data, "home");
    if (key_qpos) {
      mju_copy(data->qpos, key_qpos, model->nq);
    }

    // set initial qvel via keyframe
    double* key_qvel = mjpc::KeyQVelByName(model, data, "home");
    if (key_qvel) {
      mju_copy(data->qvel, key_qvel, model->nv);
    }

    // mj_forward(model, data);

    int qpos_size = model->nq + model->nv;
    int action_size = model->nu;
    double qpos[qpos_size];
    double action[action_size];

    mju_copy(qpos, data->qpos, qpos_size);
    mju_copy(action, data->ctrl, action_size);

    double output_qpos[num_timesteps + 1][qpos_size];
    double output_action[num_timesteps + 1][action_size];

    mju_copy(output_qpos[0], data->qpos, qpos_size);

    std::stringstream data_output_path_stream;
    data_output_path_stream << "/tmp/mjpc_data-main_headless-" << 0 << ".txt";

    for (int i = 0; i < num_timesteps; ++i) {
        std::cout << "i: " << i << "\n";

        // std::cout << "agent.allocate_enabled: " << agent.allocate_enabled << "\n";
        // std::cout << "agent.plan_enabled: " << agent.plan_enabled << "\n";

        auto start = std::chrono::steady_clock::now();
        exitrequest.store(false);
        plan_pool.Schedule([&exitrequest, &uiloadrequest]() {
            agent.Plan(exitrequest, uiloadrequest);
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // std::this_thread::yield();
        exitrequest.store(true);
        plan_pool.WaitCount(1);
        plan_pool.ResetCount();

        auto end = std::chrono::steady_clock::now();

        std::cout << "end - start: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

        agent.ActivePlanner().ActionFromPolicy(
            data->ctrl, &agent.ActiveState().state()[0],
            agent.ActiveState().time());

        // Simulate one step.
        mj_step(model, data);

        mju_copy(qpos, data->qpos, qpos_size);
        mju_copy(action, data->ctrl, action_size);

        data_output_path_stream.str("");
        data_output_path_stream.clear();
        data_output_path_stream << "/tmp/mjpc_data-main_headless-" << i << ".txt";
        mj_printFormattedData(model, data, data_output_path_stream.str().c_str(), "%.5f");

        mju_copy(output_qpos[i+1], data->qpos, qpos_size);
        mju_copy(output_action[i], action, action_size);
    }

    mju_printMat(*output_qpos, num_timesteps + 1, qpos_size);
    mju_printMat(*output_action, num_timesteps + 1, action_size);

    std::ofstream myfile;
    myfile.open("/tmp/what.json");
    myfile << "{" << std::endl;
    myfile << "\"qpos\": [" << std::endl;

    for (int i = 0; i < num_timesteps; ++i) {
        myfile << "[";
        for (int j = 0; j < qpos_size; ++j) {
            myfile << output_qpos[i][j];
            if (j < qpos_size - 1) {
                myfile << ", ";
            }
        }
        if (i < num_timesteps - 1) {
            myfile << "]," << std::endl;
        } else {
            myfile << "]" << std::endl;
        }
    }
    myfile << "]," << std::endl;
    myfile << "\"actions\": [" << std::endl;
    for (int i = 0; i < num_timesteps; ++i) {
        myfile << "[";
        for (int j = 0; j < action_size; ++j) {
            myfile << output_action[i][j];
            if (j < action_size - 1) {
                myfile << ", ";
            }
        }
        if (i < num_timesteps - 1) {
            myfile << "]," << std::endl;
        } else {
            myfile << "]" << std::endl;
        }
    }
    myfile << "]" << std::endl;
    myfile << "}" << std::endl;
    myfile.close();

    // delete data
    mj_deleteData(data);

    // delete model
    mj_deleteModel(model);
}

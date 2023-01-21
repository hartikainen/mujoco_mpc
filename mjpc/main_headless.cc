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
#include <filesystem>
#include <thread>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/match.h>
#include <absl/strings/str_format.h>
#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "agent.h"
#include "array_safety.h"
#include "planners/include.h"
#include "task.h"
#include "tasks/tasks.h"
#include "threadpool.h"
#include "utilities.h"

ABSL_FLAG(std::string, task, "", "Which model to load on startup.");
ABSL_FLAG(
    std::optional<std::string>,
    mocap_id,
    std::nullopt,
    "Which mocap sequence to use for tracking.");

ABSL_FLAG(
    std::optional<std::string>,
    output_path,
    std::nullopt,
    "Destination path for output json.");
ABSL_FLAG(
    std::optional<double>,
    agent_horizon,
    std::nullopt,
    "agent_horizon");
ABSL_FLAG(
    std::optional<double>,
    agent_timestep,
    std::nullopt,
    "agent_timestep");
ABSL_FLAG(
    std::optional<int>,
    ilqg_num_rollouts,
    std::nullopt,
    "ilqg_num_rollouts");
ABSL_FLAG(
    std::optional<int>,
    ilqg_regularization_type,
    std::nullopt,
    "ilqg_regularization_type");
ABSL_FLAG(
    std::optional<int>,
    ilqg_representation,
    std::nullopt,
    "ilqg_representation");

namespace {
namespace mju = mujoco::util_mjpc;

// load error string length
const int kErrorLength = 1024;
const int kMaxFilenameLength = 1000;

mjModel* model = nullptr;
mjData* data = nullptr;
mjpc::Agent agent;


std::string UniformString(uint8_t len) {
    absl::BitGen bit_gen;
    static const std::string alphanum =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

    std::string result(len, '0');
    for (int i = 0; i < len; ++i) {
        result[i] = alphanum[
            absl::Uniform<uint8_t>(bit_gen, 0, alphanum.size())];
    }
    return result;
}


// --------------------------------- callbacks ---------------------------------

// sensor callback
void sensor_callback(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) {
    agent.task().Residuals(model, data, data->sensordata);
  }
}

// // sensor callback
// void control_callback(const mjModel* m, mjData* d) {
//     if (data != d) {
//         return;
//     }

//   if (agent.action_enabled) {
//     agent.ActivePlanner().ActionFromPolicy(
//         d->ctrl, &agent.ActiveState().state()[0],
//         agent.ActiveState().time());
//   }
//   // // if noise
//   // if (!agent.allocate_enabled && false) {
//   //   for (int j = 0; j < model->nu; j++) {
//   //     d->ctrl[j] += ctrlnoise[j];
//   //   }
//   // }
// }

//--------------------------------- simulation ---------------------------------

mjModel* LoadModel(std::string file, const mjVFS* vfs) {
  // this copy is needed so that the mju::strlen call below compiles

  char filename[1024];
  mju::strcpy_arr(filename, file.c_str());

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
    mnew = mj_loadModel(filename, vfs);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    mnew = mj_loadXML(filename, vfs, loadError, kMaxFilenameLength);
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

std::unique_ptr<mjVFS> CopyPathsToMjVFS(
    std::vector<std::tuple<std::string, std::string>> filenames) {

  // mjVFS structs need to be allocated on the heap, because it's ~2MB
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());

  std::cout << "Copying files to mujoco VFS." << std::endl;

  for (const auto& [source_filename, target_filename] : filenames) {
    std::cout << "source filename: " << source_filename << std::endl
              << "target filename: " << target_filename << std::endl;

    std::ifstream t(source_filename);
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string xml_string = buffer.str();
    // std::cout << "xml_string: " << xml_string << std::endl;
    mj_makeEmptyFileVFS(
        vfs.get(), target_filename.c_str(), xml_string.size() + 1);
    int file_idx = mj_findFileVFS(vfs.get(), target_filename.c_str());
    memcpy(vfs->filedata[file_idx], xml_string.data(), xml_string.size() + 1);
  }

  return vfs;
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
            std::cerr << "Invalid --task flag: '" << task_flag
                      << "'. Valid values:\n";
            for (const auto& task : mjpc::kTasks) {
                std::cerr << '\t' << task.name << '\n';
            }
            mju_error("Invalid --task flag.");
            return -1;
        }
    }

    auto mocap_id_flag = absl::GetFlag(FLAGS_mocap_id);
    if (!mocap_id_flag.has_value()) {
        mju_error("Invalid `--mocap_id` flag.");
        return -1;
    }

    auto mocap_id = mocap_id_flag.value();

    std::cout << "mocap_id: " << mocap_id << std::endl;

    auto output_path = absl::GetFlag(FLAGS_output_path);

    const auto& taskDef = mjpc::kTasks[agent.task().id];

    // default model + task
    std::string filename =
        mjpc::GetModelPath(mjpc::kTasks[agent.task().id].xml_path);

    std::ifstream t(filename);
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string xml_string = buffer.str();

    int keyframe_include_position =
        xml_string.find("</sensor>\n") + std::strlen("</sensor>\n");
    std::string xml_string_head = xml_string.substr(0, keyframe_include_position);
    std::string xml_string_tail =
        xml_string.substr(keyframe_include_position, xml_string.size());

    // std::cout << "xml_string_head: " << xml_string_head << std::endl;
    // std::cout << "xml_string_tail: " << xml_string_tail << std::endl;

    auto xml_string_with_keyframe_include =
        (xml_string_head +
         absl::StrFormat("  <include file=\"./keyframes/%s\" />", mocap_id) +
         xml_string_tail);

    // std::cout << "xml_string_with_keyframe_include: " <<
    // xml_string_with_keyframe_include << std::endl;

    std::filesystem::path task_xml_path =
        mjpc::GetModelPath(mjpc::kTasks[agent.task().id].xml_path);

    std::string temp_filename = UniformString(10);
    auto xml_with_keyframe_path = (
        std::filesystem::temp_directory_path()
        / temp_filename
    ).replace_extension(".xml");

    std::cout << "temp_filepath: " << temp_filename << std::endl;

    // mjpc/tasks/humanoid/tracking/task.xml
    // mjpc/tasks/common.xml
    // mjpc/tasks/humanoid/humanoid.xml
    // std::cout << "xml_with_keyframe_path: "
    //           << xml_with_keyframe_path
    //           << std::endl;

    std::ofstream xml_with_keyframe_filestream;
    xml_with_keyframe_filestream.open(xml_with_keyframe_path);
    xml_with_keyframe_filestream << xml_string_with_keyframe_include;
    xml_with_keyframe_filestream.close();

    std::filesystem::path vfs_base_path = "mjpc/tasks/";

    std::vector<
        std::tuple<std::string, std::string>> vfs_source_and_target_paths = {
      {xml_with_keyframe_path,
            vfs_base_path / mjpc::kTasks[agent.task().id].xml_path},
      {mjpc::GetModelPath("humanoid/humanoid.xml"),
            vfs_base_path / "humanoid/humanoid.xml"},
      {mjpc::GetModelPath("common.xml"), vfs_base_path / "common.xml"},
      {mjpc::GetModelPath("humanoid/tracking/keyframes/" + mocap_id),
            vfs_base_path / "humanoid/tracking/keyframes" / mocap_id},
    };

    auto vfs = CopyPathsToMjVFS(vfs_source_and_target_paths);

    // load model + make data
    model = LoadModel(xml_with_keyframe_path, vfs.get());

    // create data
    data = mj_makeData(model);

    // remove temporary data.
    // TODO(hartikainen): `mj_deleteFileVFS` has a bug resulting in infinite loop.
    // for (const auto& [_, vfs_filename] : vfs_source_and_target_paths) {
    //     std::cout << "what: " << vfs_filename << std::endl;
    //     assert(mj_deleteFileVFS(vfs.get(), vfs_filename.c_str()));
    // }
    mj_deleteVFS(vfs.get());
    std::filesystem::remove(xml_with_keyframe_path);

    // sensor callback
    mjcb_sensor = &sensor_callback;

    // // control callback
    // mjcb_control = &control_callback;

    auto agent_horizon_flag = absl::GetFlag(FLAGS_agent_horizon);
    if (agent_horizon_flag.has_value()) {
        std::cout << "agent_horizon_flag: " << agent_horizon_flag.value() << std::endl;
        mjpc::SetCustomNumericData(
            model, "agent_horizon", agent_horizon_flag.value());
    }
    auto agent_timestep_flag = absl::GetFlag(FLAGS_agent_timestep);
    if (agent_timestep_flag.has_value()) {
        std::cout << "agent_timestep_flag: " << agent_timestep_flag.value() << std::endl;
        mjpc::SetCustomNumericData(
            model, "agent_timestep", agent_timestep_flag.value());
    }
    auto ilqg_num_rollouts_flag = absl::GetFlag(FLAGS_ilqg_num_rollouts);
    if (ilqg_num_rollouts_flag.has_value()) {
        std::cout << "ilqg_num_rollouts_flag: " << ilqg_num_rollouts_flag.value() << std::endl;
        mjpc::SetCustomNumericData(
            model, "ilqg_num_rollouts", ilqg_num_rollouts_flag.value());
    }
    auto ilqg_regularization_type_flag = absl::GetFlag(
        FLAGS_ilqg_regularization_type);
    if (ilqg_regularization_type_flag.has_value()) {
        std::cout << "ilqg_regularization_type_flag: " << ilqg_regularization_type_flag.value() << std::endl;
        mjpc::SetCustomNumericData(
            model, "ilqg_regularization_type", ilqg_regularization_type_flag.value());
    }
    auto ilqg_representation_flag = absl::GetFlag(FLAGS_ilqg_representation);
    if (ilqg_representation_flag.has_value()) {
        std::cout << "ilqg_representation_flag: " << ilqg_representation_flag.value() << std::endl;
        mjpc::SetCustomNumericData(
            model, "ilqg_representation", ilqg_representation_flag.value());
    }

    // ----- initialize agent ----- //
    const char task_str[] = "";
    const char planners_str[] = "";
    agent.Initialize(model, task_str, planners_str,
                     taskDef.residual, taskDef.transition);

    // pool
    auto max_threads = mjpc::NumAvailableHardwareThreads();
    mjpc::ThreadPool plan_pool(max_threads);

    // ----- switch to iLQG planner ----- //
    agent.Allocate();
    agent.Reset();

    // ----- plan w/ iLQG planner ----- //
    agent.plan_enabled = true;
    agent.action_enabled = true;
    agent.visualize_enabled = false;
    agent.plot_enabled = false;

    float fps = 30.0;
    float simulation_duration_s = (float)model->nkey / fps;
    int num_timesteps = simulation_duration_s / model->opt.timestep;
    // num_timesteps = 5;
    std::cout << "num_timesteps: " << num_timesteps << "\n";

    float plan_times[num_timesteps];
    float total_times[num_timesteps];

    mj_resetData(model, data);
    mj_forward(model, data);

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

    int qpos_size = model->nq + model->nv;
    int action_size = model->nu;

    double output_qpos[num_timesteps + 1][qpos_size];
    double output_action[num_timesteps + 1][action_size];

    mju_copy(output_qpos[0] + 0, data->qpos, model->nq);
    mju_copy(output_qpos[0] + model->nq, data->qvel, model->nv);

    mjpc::Task task;
    task.Set(model, taskDef.residual, taskDef.transition);
    std::vector<double> residual;
    residual.resize(task.num_residual);

    double total_cost = 0.0;

    for (int i = 0; i < num_timesteps; ++i) {
        auto loop_start = std::chrono::steady_clock::now();
        auto plan_start = std::chrono::steady_clock::now();

        // Plan to get actions.
        agent.ActiveState().Set(model, data);
        agent.PlanIteration(&plan_pool);

        auto plan_end = std::chrono::steady_clock::now();

        int plan_time_ms = std::chrono::duration_cast
            <std::chrono::milliseconds>(plan_end - plan_start).count();
        plan_times[i] = plan_time_ms / 1000.0;

        // Set `data->ctrl` from agent.
        agent.ActivePlanner().ActionFromPolicy(
            data->ctrl, &agent.ActiveState().state()[0],
            agent.ActiveState().time());

        // Simulate one step.
        mj_step(model, data);

        // Store data.
        mju_copy(output_qpos[i+1] + 0, data->qpos, model->nq);
        mju_copy(output_qpos[i+1] + model->nq, data->qvel, model->nv);
        mju_copy(output_action[i], data->ctrl, action_size);

        auto loop_end = std::chrono::steady_clock::now();

        int total_time_ms = std::chrono::duration_cast
            <std::chrono::milliseconds>(loop_end - loop_start).count();
        total_times[i] = total_time_ms / 1000.0;

        // Print diagnostics.

        task.Transition(model, data);
        task.Residuals(model, data, residual.data());
        auto cost = task.CostValue(residual.data());

        // std::cout << "step: " << i << "/" << num_timesteps << "; "
        //           << "plan_time [s]: " << plan_times[i] << "; "
        //           << "total_time [s]: " << total_times[i] << "; "
        //           << "cost: " << cost << std::endl;
        total_cost += cost;
    }

    std::cout << "total_cost: " << total_cost << "; "
              << "total_cost per step: " << total_cost / (double)num_timesteps
              << std::endl;

    if (output_path.has_value()) {
        std::filesystem::create_directories(
            std::filesystem::path(output_path.value()).parent_path());
        std::ofstream output_stream;
        output_stream.open(output_path.value());
        output_stream << "{" << std::endl;
        output_stream << "\"qpos\": [" << std::endl;

        for (int i = 0; i < num_timesteps + 1; i++) {
            output_stream << "[";
            for (int j = 0; j < qpos_size; j++) {
                output_stream << output_qpos[i][j];
                if (j < qpos_size - 1) {
                    output_stream << ", ";
                }
            }
            if (i < num_timesteps) {
                output_stream << "]," << std::endl;
            } else {
                output_stream << "]" << std::endl;
            }
        }
        output_stream << "]," << std::endl;
        output_stream << "\"actions\": [" << std::endl;
        for (int i = 0; i < num_timesteps + 1; i++) {
            output_stream << "[";
            for (int j = 0; j < action_size; j++) {
                output_stream << output_action[i][j];
                if (j < action_size - 1) {
                    output_stream << ", ";
                }
            }
            if (i < num_timesteps) {
                output_stream << "]," << std::endl;
            } else {
                output_stream << "]" << std::endl;
            }
        }
        output_stream << "]" << std::endl;
        output_stream << "}" << std::endl;
        output_stream.close();
    }

    // delete data
    mj_deleteData(data);

    // delete model
    mj_deleteModel(model);

}

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

#include "mjpc/tasks/humanoid/tracking/tracking.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <span>
#include <string>
#include <tuple>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace {
// compute interpolation between mocap frames
std::tuple<int, int, double, double> ComputeInterpolationValues(double index,
                                                                int max_index) {
  int index_0 = std::floor(std::clamp(index, 0.0, (double)max_index));
  int index_1 = std::min(index_0 + 1, max_index);

  double weight_1 = std::clamp(index, 0.0, (double)max_index) - index_0;
  double weight_0 = 1.0 - weight_1;

  return {index_0, index_1, weight_0, weight_1};
}

// Hardcoded constant matching keyframes from CMU mocap dataset.
// names for humanoid bodies
const std::array<std::string, 16> body_names = {
    "pelvis",    "head",      "ltoe",  "rtoe",  "lheel",  "rheel",
    "lknee",     "rknee",     "lhand", "rhand", "lelbow", "relbow",
    "lshoulder", "rshoulder", "lhip",  "rhip",
};

}  // namespace

namespace mjpc::humanoid {

std::string Tracking::XmlPath() const {
  return GetModelPath("humanoid/tracking/task.xml");
}
std::string Tracking::Name() const { return "Humanoid Track"; }

// ------------- Residuals for humanoid tracking task -------------
//   Number of residuals:
//     Residual (0): Joint vel: minimise joint velocity
//     Residual (1): Control: minimise control
//     Residual (2-11): Tracking position: minimise tracking position error
//         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
//     Residual (11-20): Tracking velocity: minimise tracking velocity error
//         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
//   Number of parameters: 0
// ----------------------------------------------------------------
void Tracking::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                  double *residual) const {
  double fps = parameters_[ParameterIndex(model, "Mocap FPS")];

  // ----- get mocap frames ----- //
  // get motion start index
  int start = 0;
  // get motion trajectory length
  int length = model->nkey;
  double current_index = (data->time - reference_time_) * fps + start;
  int last_key_index = start + length - 1;

  int counter = 0;

  // Positions:
  // We interpolate linearly between two consecutive key frames in order to
  // provide smoother signal for tracking.
  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) =
      ComputeInterpolationValues(current_index, last_key_index);

  // ----- residual ----- //

  // ----- action ----- //
  if (data->time == 0) {
    mju_zero(residual + counter, model->nu);
  } else {
    mju_copy(residual + counter, data->ctrl, model->nu);
  }
  counter += model->nu;

  // ----- act_dot ----- //
  if (data->time == 0) {
    mju_zero(residual + counter, model->na);
  } else {
    mju_copy(residual + counter, data->act_dot, model->na);
  }
  counter += model->na;

  // ----- position ----- //
  // Compute interpolated frame.
  auto get_body_mpos = [&](const std::string &body_name, double result[3]) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
    assert(0 <= mocap_body_id);
    int body_mocapid = model->body_mocapid[mocap_body_id];
    assert(0 <= body_mocapid);

    // current frame
    mju_scl3(
        result,
        model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * body_mocapid,
        weight_0);

    // next frame
    mju_addToScl3(
        result,
        model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * body_mocapid,
        weight_1);
  };

  auto get_body_sensor_pos = [&](const std::string &body_name,
                                 double result[3]) {
    std::string pos_sensor_name = "tracking_pos[" + body_name + "]";
    double *sensor_pos = SensorByName(model, data, pos_sensor_name.c_str());
    mju_copy3(result, sensor_pos);
  };

  double pelvis_mpos[3];
  get_body_mpos("pelvis", pelvis_mpos);

  double pelvis_sensor_pos[3];
  get_body_sensor_pos("pelvis", pelvis_sensor_pos);

  auto &body_name = std::span(body_names).front();
  assert(body_name == "pelvis");

  mju_sub3(&residual[counter], pelvis_mpos, pelvis_sensor_pos);
  counter += 3;

  for (const auto &body_name : std::span(body_names).subspan(1)) {
    double body_mpos[3];
    get_body_mpos(body_name, body_mpos);

    // current position
    double body_sensor_pos[3];
    get_body_sensor_pos(body_name, body_sensor_pos);

    // // Pelvis and shoulders are global, rest are relative to pelvis.
    // bool is_pelvis_relative = (
    //   !(body_name == "pelvis" || body_name == "lshoulder" || body_name == "rshoulder")
    // );
    // if (is_pelvis_relative) {
    //   mju_subFrom3(body_mpos, pelvis_mpos);
    //   mju_subFrom3(body_sensor_pos, pelvis_sensor_pos);
    // }

    mju_sub3(&residual[counter], body_mpos, body_sensor_pos);
    counter += 3;
  }


  if (last_key_index < current_index) {
    mju_scl(
      residual + model->nu + model->na,
      residual + model->nu + model->na,
      mju_pow(0.8, current_index - last_key_index),
      counter - model->nu - model->na);
  }

  CheckSensorDim(model, counter);
}

// --------------------- Transition for humanoid task -------------------------
//   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
//   Linearly interpolate between two consecutive key frames in order to
//   smooth the transitions between keyframes.
// ----------------------------------------------------------------------------
void Tracking::TransitionLocked(mjModel *model, mjData *d) {}

}  // namespace mjpc::humanoid

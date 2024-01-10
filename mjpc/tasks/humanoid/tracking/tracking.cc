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

constexpr int kMotionLengths[] = {
    121,  // Jump - CMU-CMU-02-02_04
    154,  // Kick Spin - CMU-CMU-87-87_01
    115,  // Spin Kick - CMU-CMU-88-88_06
    78,   // Cartwheel (1) - CMU-CMU-88-88_07
    145,  // Crouch Flip - CMU-CMU-88-88_08
    188,  // Cartwheel (2) - CMU-CMU-88-88_09
    260,  // Monkey Flip - CMU-CMU-90-90_19
    279,  // Dance - CMU-CMU-103-103_08
    39,   // Run - CMU-CMU-108-108_13
    510,  // Walk - CMU-CMU-137-137_40
};

// return length of motion trajectory
int MotionLength(int id) { return kMotionLengths[id]; }

// return starting keyframe index for motion
int MotionStartIndex(int id) {
  int start = 0;
  for (int i = 0; i < id; i++) {
    start += MotionLength(i);
  }
  return start;
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
  int start = MotionStartIndex(current_mode_);
  // get motion trajectory length
  int length = MotionLength(current_mode_);
  double current_index = (data->time - reference_time_) * fps + start;
  int last_key_index = start + length - 1;

  int counter = 0;

  if (last_key_index < current_index) {
    counter =
      model->na
      + model->nv - 6
      + model->nu
      + 2 * 3
      + 7 * 6
      + 2 * 3
      + 7 * 6;
    mju_zero(residual, counter);
    CheckSensorDim(model, counter);
    return;
  }

  // Positions:
  // We interpolate linearly between two consecutive key frames in order to
  // provide smoother signal for tracking.
  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) =
      ComputeInterpolationValues(current_index, last_key_index);

  // ----- residual ----- //

  // ----- act_dot ----- //
  if (data->time == 0) {
    mju_zero(residual + counter, model->na);
  } else {
    mju_copy(residual + counter, data->act_dot, model->na);
  }
  counter += model->na;

  // ----- joint velocity ----- //
  if (data->time == 0) {
    mju_zero(residual + counter, model->nv - 6);
  } else {
    mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
  }
  counter += model->nv - 6;

  // ----- action ----- //
  if (data->time == 0) {
    mju_zero(residual + counter, model->nu);
  } else {
    mju_copy(residual + counter, data->ctrl, model->nu);
  }
  counter += model->nu;

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

  double pelvis_distance_xyz[3];
  mju_sub3(pelvis_distance_xyz, pelvis_mpos, pelvis_sensor_pos);

  double ltoe_mpos[3];
  get_body_mpos("ltoe", ltoe_mpos);
  double rtoe_mpos[3];
  get_body_mpos("rtoe", rtoe_mpos);

  double min_toe_z = std::min({ltoe_mpos[2], rtoe_mpos[2]});

  auto sigmoid = [] (double x) { return 1.0 / (1.0 + std::exp(-x)); };

  double pelvis_xy_multiplier = 20.0;
  // We ease the root's z-position tolerance when the toes are slightly off the
  // ground, because some of the motion sequences have small vertical errors
  // perhaps due to hidden platforms in the scene. The fix is roughly a soft
  // version of following:
  // `(0.04 < min_toe_z && min_toe_z < 0.09) ? 6.0 : 20.0;`
  // TODO(hartikainen): Fix/filter the mocap sequences and remove the custom
  // `pelvis_z_multiplier` from here.
  double pelvis_z_multiplier =
    pelvis_xy_multiplier - 18.0 * (sigmoid(200.0 * (min_toe_z - 0.00))
                                   - sigmoid(200.0 * (min_toe_z - 0.20)));
  residual[counter + 0] = pelvis_distance_xyz[0] * pelvis_xy_multiplier;
  residual[counter + 1] = pelvis_distance_xyz[1] * pelvis_xy_multiplier;
  residual[counter + 2] = pelvis_distance_xyz[2] * pelvis_z_multiplier;
  counter += 3;

  for (const auto &body_name : std::span(body_names).subspan(1)) {
    double body_mpos[3];
    get_body_mpos(body_name, body_mpos);

    // current position
    double body_sensor_pos[3];
    get_body_sensor_pos(body_name, body_sensor_pos);

    mju_subFrom3(body_mpos, pelvis_mpos);
    mju_subFrom3(body_sensor_pos, pelvis_sensor_pos);

    mju_sub3(&residual[counter], body_mpos, body_sensor_pos);
    counter += 3;
  }

  // ----- velocity ----- //
  for (const auto &body_name : body_names) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    std::string linvel_sensor_name = "tracking_linvel[" + body_name + "]";
    int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
    assert(0 <= mocap_body_id);
    int body_mocapid = model->body_mocapid[mocap_body_id];
    assert(0 <= body_mocapid);

    // compute finite-difference velocity
    mju_copy3(
        &residual[counter],
        model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * body_mocapid);
    mju_subFrom3(
        &residual[counter],
        model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * body_mocapid);
    mju_scl3(&residual[counter], &residual[counter], fps);

    // subtract current velocity
    double *sensor_linvel =
        SensorByName(model, data, linvel_sensor_name.c_str());
    mju_subFrom3(&residual[counter], sensor_linvel);

    counter += 3;
  }

  CheckSensorDim(model, counter);
}

// --------------------- Transition for humanoid task -------------------------
//   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
//   Linearly interpolate between two consecutive key frames in order to
//   smooth the transitions between keyframes.
// ----------------------------------------------------------------------------
void Tracking::TransitionLocked(mjModel *model, mjData *d) {
  double fps = residual_.parameters_[ParameterIndex(model, "Mocap FPS")];

  // get motion start index
  int start = MotionStartIndex(mode);
  // get motion trajectory length
  int length = MotionLength(mode);

  // check for motion switch
  if (residual_.current_mode_ != mode || d->time == 0.0) {
    residual_.current_mode_ = mode;       // set motion id
    residual_.reference_time_ = d->time;  // set reference time

    // set initial state
    mju_copy(d->qpos, model->key_qpos + model->nq * start, model->nq);
    mju_copy(d->qvel, model->key_qvel + model->nv * start, model->nv);
  }

  // indices
  double current_index = (d->time - residual_.reference_time_) * fps + start;
  int last_key_index = start + length - 1;

  // Positions:
  // We interpolate linearly between two consecutive key frames in order to
  // provide smoother signal for tracking.
  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) =
      ComputeInterpolationValues(current_index, last_key_index);

  mj_markStack(d);

  mjtNum *mocap_pos_0 = mj_stackAllocNum(d, 3 * model->nmocap);
  mjtNum *mocap_pos_1 = mj_stackAllocNum(d, 3 * model->nmocap);

  // Compute interpolated frame.
  mju_scl(mocap_pos_0, model->key_mpos + model->nmocap * 3 * key_index_0,
          weight_0, model->nmocap * 3);

  mju_scl(mocap_pos_1, model->key_mpos + model->nmocap * 3 * key_index_1,
          weight_1, model->nmocap * 3);

  mju_copy(d->mocap_pos, mocap_pos_0, model->nmocap * 3);
  mju_addTo(d->mocap_pos, mocap_pos_1, model->nmocap * 3);

  for (int i = 0; i < model->nmocap; ++i) {
    d->mocap_pos[i * 3 + 1] += 1.0;
  }

  mj_freeStack(d);
}

}  // namespace mjpc::humanoid

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

#include "mjpc/tasks/humanoid/skateboard/steering.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <string>
#include <tuple>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace {
  int jiiri = 0;
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
constexpr double kFps = 30.0;

constexpr int kMotionLengths[] = {
    1, // steering
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

// names for humanoid bodies
const std::array<std::string, 16> body_names = {
    "pelvis",    "head",      "ltoe",  "rtoe",  "lheel",  "rheel",
    "lknee",     "rknee",     "lhand", "rhand", "lelbow", "relbow",
    "lshoulder", "rshoulder", "lhip",  "rhip",
};
// compute mocap translations and rotations
void move_mocap_poses( mjtNum *result, const mjModel *model, const mjData *data,  const std::vector<double, std::allocator<double>> parameters, int mode) {
  // todo move residual here

    // mjtNum *modified_mocap_pos = new mjtNum[3 * (model->nmocap - 1)];
    std::vector<mjtNum> modified_mocap_pos(3 * model->nmocap);

    // Compute interpolated frame.
    mju_scl(modified_mocap_pos.data(), model->key_mpos + (model->nmocap)* 3 * mode,
            1, (model->nmocap)* 3);
    double skateboard_center[3] = {0.0, 0.0, 0.0};
    int skateboard_body_id_ = mj_name2id(model, mjOBJ_XBODY, "skateboard");

    // move mpos to x,y position of skateboard
    mju_copy(skateboard_center, data->xpos + 3 * skateboard_body_id_, 3);
    
    // print average center of mpos
    double average_mpos[2] = {0.0, 0.0};
    // if (mode == kModeSteer){
      // get average center of mpos
    for (int i = 0; i < model->nmocap -1; i++) {
      average_mpos[0] += modified_mocap_pos[3 * i + 0];
      average_mpos[1] += modified_mocap_pos[3 * i + 1];

      modified_mocap_pos[3 * i + 0] += skateboard_center[0];
      modified_mocap_pos[3 * i + 1] += skateboard_center[1];
      modified_mocap_pos[3 * i + 2] += skateboard_center[2]-0.1;
    
    }
    average_mpos[0] /= model->nmocap -1;
    average_mpos[1] /= model->nmocap -1;

    // subtract the difference between average_mpos and skateboard_center
    for (int i = 0; i < model->nmocap -1; i++) {
      modified_mocap_pos[3 * i + 0] -= average_mpos[0];
      modified_mocap_pos[3 * i + 1] -= average_mpos[1];
      // modified_mocap_pos[3 * i + 2] += skateboard_center[2];
    }
    // }    

    double skateboard_heading = 0.0;
    double skateboard_xmat[9] = {0.0, 0.0, 0.0};
    mju_copy(skateboard_xmat, data->xmat + 9 * skateboard_body_id_, 9);
    skateboard_heading = atan2(skateboard_xmat[3], skateboard_xmat[0]);
    skateboard_heading -= M_PI / 2.0; 
    
    int goal_id = mj_name2id(model, mjOBJ_XBODY, "goal");
    if (goal_id < 0) mju_error("body 'goal' not found");

    int goal_mocap_id_ = model->body_mocapid[goal_id];
    if (goal_mocap_id_ < 0) mju_error("body 'goal' is not mocap");

    // get goal position
    double* goal_pos = data->mocap_pos + 3*goal_mocap_id_;

    // Get goal heading from board position
    double goal_heading = atan2(goal_pos[1] - skateboard_center[1], goal_pos[0] - skateboard_center[0]) - M_PI / 2.0;;

    // Calculate heading error using sine function
    double heading_error = sin(goal_heading - skateboard_heading)/3;


    // Rotate the pixels in 3D space around the Z-axis (board_center)
    double mocap_tilt =  0.2; // parameters[ParameterIndex(model, "Tilt ratio")];
    // # TODO(eliasmikkola): fix ParameterIndex not working (from utilities.h)
    // tilt angle max is PI/3
    double tilt_angle = ( mju_min(0.5, mju_max(-0.5, heading_error)) * M_PI / 2.0) * mocap_tilt;
    // tilt_angle = 0.0;
    for (int i = 0; i < model->nmocap -1; i++) {
        // Get the pixel position relative to the board_center
        double rel_x = modified_mocap_pos[3 * i + 0] - skateboard_center[0];
        double rel_y = modified_mocap_pos[3 * i + 1] - skateboard_center[1];

        // perform rotation of tilt_angle around the Y-axis
        double rotated_z = rel_x * sin(tilt_angle) + modified_mocap_pos[3 * i + 2] * cos(tilt_angle);
        rel_x = rel_x * cos(tilt_angle) - modified_mocap_pos[3 * i + 2] * sin(tilt_angle);

        // Perform rotation around the Z-axis
        double rotated_x = cos(skateboard_heading) * rel_x - sin(skateboard_heading) * rel_y;
        double rotated_y = sin(skateboard_heading) * rel_x + cos(skateboard_heading) * rel_y;


        // Update the rotated pixel positions in modified_mocap_pos
        modified_mocap_pos[3 * i + 0] = skateboard_center[0] + rotated_x;
        modified_mocap_pos[3 * i + 1] = skateboard_center[1] + rotated_y;
        modified_mocap_pos[3 * i + 2] = rotated_z;


        rotated_x = cos(skateboard_heading) * rel_x - sin(skateboard_heading) * rel_y;
        rotated_y = sin(skateboard_heading) * rel_x + cos(skateboard_heading) * rel_y;

    }
    mju_copy(result, modified_mocap_pos.data(), (model->nmocap - 1) * 3);
}
// std::vector<double> ComputeFeetErrorResidual(const mjModel *model, const mjData *data, const int current_mode_, const double reference_time_, const std::function<double*(const mjModel*, const mjData*, const char*)>& SensorByName){
//     // ----- Skateboard: Feet should be on the skateboard ----- //

//   double *back_plate_pos = SensorByName(model, data, "track-back-plate");
//   double *tail_pos = SensorByName(model, data, "track-tail");
//   double *front_plate_pos = SensorByName(model, data, "track-front-plate");

//   double *left_foot_pos = SensorByName(model, data, "tracking_foot_left");
//   double *right_foot_pos = SensorByName(model, data, "tracking_foot_right");

//   double right_feet_slider = parameters[ParameterIndex(model, "rFeet pos")];
//   double left_feet_slider = parameters[ParameterIndex(model, "LFeet pos")];
  
//   // calculate x-wise difference between the plates, based on right_feet_slider
//   double plate_distance_x = mju_abs(back_plate_pos[0] - front_plate_pos[0]);
//   double plate_distance_y = mju_abs(back_plate_pos[1] - front_plate_pos[1]);
//   // calculate the x position of the line set by the plates
//   double right_feet_x = front_plate_pos[0] - right_feet_slider * plate_distance_x;
//   double right_feet_y = front_plate_pos[1] - right_feet_slider * plate_distance_y;
  

  
//   // print target and current z position
//   // left feet error, distance to back plate position 
//   double distance_x = mju_abs(left_foot_pos[0] - back_plate_pos[0]);
//   double distance_y = mju_abs(left_foot_pos[1] - back_plate_pos[1]);
//   double distance_z = mju_abs(left_foot_pos[2] - (back_plate_pos[2]));
//   if (left_feet_slider > 0) {
//     distance_x = mju_abs(left_foot_pos[0] - tail_pos[0]);
//     distance_y = mju_abs(left_foot_pos[1] - tail_pos[1]);
//     distance_z = mju_abs(left_foot_pos[2] - (tail_pos[2]));
    
//   }
//   if (left_foot_pos[2] < back_plate_pos[2] && distance_x < 0.2 && distance_y < 0.2 && back_plate_pos[2] <0.3) distance_z *= 10;
//   double left_feet_error = mju_sqrt(distance_x*distance_x + distance_y*distance_y + (distance_z*distance_z));

//   // right feet error, distance to front plate position
//   distance_x = mju_abs(right_foot_pos[0] - right_feet_x);
//   distance_y = mju_abs(right_foot_pos[1] - right_feet_y);
//   distance_z = mju_abs(right_foot_pos[2] - front_plate_pos[2]);
//   if (right_foot_pos[2] < front_plate_pos[2] && distance_x < 0.2 && distance_y < 0.2 && front_plate_pos[2] <0.3) distance_z *= 10;
//   double right_feet_error = mju_sqrt(distance_x*distance_x + distance_y*distance_y + (distance_z*distance_z));

//   residual[counter++] = right_feet_error * standing;
//   residual[counter++] = left_feet_error * standing;
// }  

// current_mode_ and reference_time_ as Int, pass function SensorByName
std::vector<double> ComputeTrackingResidual(const mjModel *model, const mjData *data, const int current_mode_, const double reference_time_) {
  // TODO(eliasmikkola): doesn't match the original tracking behavior
  // could be either SensorByName or the vector addition 
  //   * Figure out `SensorByName`

  
  std::vector<mjtNum> mocap_translated(3 * model->nmocap);

  // if jiiri % 50, else copy data mocap_pos
  move_mocap_poses(mocap_translated.data(), model, data, {}, current_mode_);

  // ----- get mocap frames ----- //
  // get motion start index
  int start = MotionStartIndex(current_mode_);
  // get motion trajectory length
  int length = MotionLength(current_mode_);
  double current_index = (data->time - reference_time_) * kFps + start;
  int last_key_index = start + length - 1;

  // create a vector to store the residuals
  std::vector<double> residual_to_return;
  // Positions:
  // We interpolate linearly between two consecutive key frames in order to
  // provide smoother signal for steering.
  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) =
      ComputeInterpolationValues(current_index, last_key_index);

  // ----- position ----- //
  // Compute interpolated frame.
  auto get_body_mpos = [&](const std::string &body_name, double result[3]) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
    assert(0 <= mocap_body_id);
    int body_mocapid = model->body_mocapid[mocap_body_id];
    assert(0 <= body_mocapid);

    // current frame
    // mju_copy( result, mocap_translated + 3 * body_mocapid, 3);
    mju_scl3(
        result,
        mocap_translated.data() + model->nmocap * 3 * key_index_0 + 3 * body_mocapid,
        weight_0);

    // next frame
    mju_addToScl3(
        result,
        mocap_translated.data() + model->nmocap * 3 * key_index_1 + 3 * body_mocapid,
        weight_1);
  };

  auto get_body_sensor_pos = [&](const std::string &body_name,
                                 double result[3]) {
    std::string pos_sensor_name = "tracking_pos[" + body_name + "]";
    double *sensor_pos = mjpc::SensorByName(model, data, pos_sensor_name.c_str());
    // printf("pos_sensor_name: %s\n", pos_sensor_name.c_str());
    // printf("sensor_pos: %f, %f, %f\n", sensor_pos[0], sensor_pos[1], sensor_pos[2]);
    mju_copy3(result, sensor_pos);
  };

  // compute marker and sensor averages
  double avg_mpos[3] = {0};
  double avg_sensor_pos[3] = {0};
  int num_body = 0;
  for (const auto &body_name : body_names) {
    double body_mpos[3];
    double body_sensor_pos[3];
    get_body_mpos(body_name, body_mpos);
    mju_addTo3(avg_mpos, body_mpos);
    get_body_sensor_pos(body_name, body_sensor_pos);
    mju_addTo3(avg_sensor_pos, body_sensor_pos);
    num_body++;
  }
  mju_scl3(avg_mpos, avg_mpos, 1.0/num_body);
  mju_scl3(avg_sensor_pos, avg_sensor_pos, 1.0/num_body);

  // residual_to_return for averages
  residual_to_return.push_back(avg_mpos[0] - avg_sensor_pos[0]);
  residual_to_return.push_back(avg_mpos[1] - avg_sensor_pos[1]);
  residual_to_return.push_back(avg_mpos[2] - avg_sensor_pos[2]);

  for (const auto &body_name : body_names) {
    double body_mpos[3];
    get_body_mpos(body_name, body_mpos);

    // current position
    double body_sensor_pos[3];
    get_body_sensor_pos(body_name, body_sensor_pos);

    mju_subFrom3(body_mpos, avg_mpos);
    mju_subFrom3(body_sensor_pos, avg_sensor_pos);
    
    residual_to_return.push_back(body_mpos[0] - body_sensor_pos[0]);
    residual_to_return.push_back(body_mpos[1] - body_sensor_pos[1]);
    residual_to_return.push_back(body_mpos[2] - body_sensor_pos[2]);
  }

  // ----- velocity ----- //
  for (const auto &body_name : body_names) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    std::string linvel_sensor_name = "tracking_linvel[" + body_name + "]";
    int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
    assert(0 <= mocap_body_id);
    int body_mocapid = model->body_mocapid[mocap_body_id];
    assert(0 <= body_mocapid);

    // Compute finite-difference velocity for x, y, z components
    double fd_velocity[3]; // Finite-difference velocity
    for (int i = 0; i < 3; ++i) {
      fd_velocity[i] = (model->key_mpos[model->nmocap * 3 * key_index_1 + 3 * body_mocapid + i] -
                        model->key_mpos[model->nmocap * 3 * key_index_0 + 3 * body_mocapid + i]) * kFps;
    }

    // Get current velocity from sensor
    double *sensor_linvel = mjpc::SensorByName(model, data, linvel_sensor_name.c_str());

    for (int i = 0; i < 3; ++i) {
      double velocity_residual = fd_velocity[i] - sensor_linvel[i];
      residual_to_return.push_back(velocity_residual);
    }
  }
  return residual_to_return;
}

}  // Namespace

namespace mjpc::humanoid {

std::string Steering::XmlPath() const {
  return GetModelPath("humanoid/skateboard/steering-task.xml");
}
std::string Steering::Name() const { return "Humanoid Skateboard Steer"; }

// ------------- Residuals for humanoid skateboard steering task -------------
//   Number of residuals:
//     Residual (0): Joint vel: minimise joint velocity
//     Residual (1): Control: minimise control
//     Residual (2-11): Steering position: minimise steering position error
//         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
//     Residual (11-20): Steering velocity: minimise steering velocity error
//         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
//   Number of parameters: 0
// ----------------------------------------------------------------

void Steering::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                    double *residual) const {
  std::vector<mjtNum> mocap_translated(3 * model->nmocap);

  // if jiiri % 50, else copy data mocap_pos
  move_mocap_poses(mocap_translated.data(), model, data, {}, current_mode_);

  // ----- residual ----- //
  int counter = 0;
    // ----- joint velocity ----- //
  int n_humanoid_joints = model->nv - 6 - 6 - 7;
  mju_copy(residual + counter, data->qvel + 6, n_humanoid_joints);
  
  counter += n_humanoid_joints;

  // ----- action ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;


  // // TODO(hartikainen): Compute each residual in their own functions, then fill
  // // in the `residual`, update `counter` and `CheckSensorDim` at the end.
  auto tracking_residual = ComputeTrackingResidual(model, data, current_mode_, reference_time_);
  mju_copy(residual + counter, tracking_residual.data(), tracking_residual.size());
  counter += tracking_residual.size();

  // print both tracking residual and tracking_residual
  CheckSensorDim(model, counter);
}

// --------------------- Transition for humanoid task -------------------------
//   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
//   Linearly interpolate between two consecutive key frames in order to
//   smooth the transitions between keyframes.
// ----------------------------------------------------------------------------
void Steering::TransitionLocked(mjModel *model, mjData *d) {
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
  double current_index = (d->time - residual_.reference_time_) * kFps + start;
  int last_key_index = start + length - 1;

  // Positions:
  // We interpolate linearly between two consecutive key frames in order to
  // provide smoother signal for steering.
  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) =
      ComputeInterpolationValues(current_index, last_key_index);

  mj_markStack(d);

  mjtNum *modified_mocap_pos = mj_stackAllocNum(d, 3 * model->nmocap);
  mjtNum *mocap_pos_1 = mj_stackAllocNum(d, 3 * model->nmocap);

  // Compute interpolated frame.
  mju_scl(modified_mocap_pos, model->key_mpos + model->nmocap * 3 * key_index_0,
          weight_0, model->nmocap * 3);

  mju_scl(mocap_pos_1, model->key_mpos + model->nmocap * 3 * key_index_1,
          weight_1, model->nmocap * 3);

  mju_copy(d->mocap_pos, modified_mocap_pos, model->nmocap * 3);
  mju_addTo(d->mocap_pos, mocap_pos_1, model->nmocap * 3);
  
  mjtNum *mocap_pos_result = mj_stackAllocNum(d, 3 * model->nmocap);
  move_mocap_poses(mocap_pos_result, model, d, parameters, mode);
  mju_copy(d->mocap_pos, mocap_pos_result, (model->nmocap - 1) * 3);
  mj_freeStack(d);
}

}  // namespace mjpc::humanoid

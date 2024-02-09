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

#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <random>
#include <string>
#include <tuple>

#include "mjpc/utilities.h"

namespace {
int jiiri = 0;

void move_goal(const mjModel *model, mjData *d,
               const std::vector<double, std::allocator<double>> parameters,
               int mode) {
  // Set new goal position in `data->mocap_pos` if we've reached the goal.
  const int goal_body_id = mj_name2id(model, mjOBJ_XBODY, "goal");
  if (goal_body_id < 0) mju_error("body 'goal' not found");
  const int goal_mocapid = model->body_mocapid[goal_body_id];
  if (goal_mocapid < 0) mju_error("mocap 'goal' not found");
  double goal_position[3];
  mju_copy3(goal_position, d->mocap_pos + 3 * goal_mocapid);

  double skateboard_position[3];
  int skateboard_body_id_ = mj_name2id(model, mjOBJ_XBODY, "skateboard");

  // move mpos to x,y position of skateboard
  mju_copy(skateboard_position, d->xpos + 3 * skateboard_body_id_, 3);

  double skateboard_goal_error[3];
  mju_sub3(skateboard_goal_error, goal_position, skateboard_position);
  double skateboard_goal_distance = mju_norm(skateboard_goal_error, 2);

  double goal_switch_threshold_m = 0.5;
  if (skateboard_goal_distance < goal_switch_threshold_m) {
    std::random_device rd;              // Obtain a random number from hardware
    std::mt19937 eng(rd());             // Seed the generator
    std::bernoulli_distribution distr;  // Define the distribution

    // Move goal to a new position. We choose a random position that is
    // `goal_offset_x` ahead and `goal_offset_y` to either left or right,
    // direction chosen uniformly. The "zero"-direction is the skateboard's
    // heading direction at the time when we reach the goal.

    // get skateboard heading.
    double skateboard_xmat[9];
    mju_copy(skateboard_xmat, d->xmat + 9 * skateboard_body_id_, 9);
    // double skateboard_heading = atan2(skateboard_xmat[3],
    // skateboard_xmat[0]);
    double skateboard_heading[2] = {skateboard_xmat[0], skateboard_xmat[3]};

    double goal_offset_xy[2];
    mju_copy(goal_offset_xy, skateboard_heading, 2);
    mju_normalize(goal_offset_xy, 2);

    double goal_move_distance_forward = 8.0;
    double goal_move_distance_side = 4.0;

    bool left_or_right = distr(eng);  // Generate a random boolean

    // // compute offset vector in front of the board.
    double goal_offset_forward[2] = {
        goal_offset_xy[0] * goal_move_distance_forward,
        goal_offset_xy[1] * goal_move_distance_forward,
    };

    // compute offset vector to the side of the board.
    double goal_offset_perpendicular[2];

    if (left_or_right) {
      goal_offset_perpendicular[0] =
          -goal_offset_xy[1] * goal_move_distance_side;
      goal_offset_perpendicular[1] =
          +goal_offset_xy[0] * goal_move_distance_side;
    } else {
      goal_offset_perpendicular[0] =
          +goal_offset_xy[1] * goal_move_distance_side;
      goal_offset_perpendicular[1] =
          -goal_offset_xy[0] * goal_move_distance_side;
    }

    double goal_offset[3] = {
        goal_offset_forward[0] + goal_offset_perpendicular[0],
        goal_offset_forward[1] + goal_offset_perpendicular[1],
        0.0,
    };

    double new_goal_position[3] = {
        skateboard_position[0] + goal_offset[0],
        skateboard_position[1] + goal_offset[1],
        goal_position[2],
    };
    mju_copy3(d->mocap_pos + 3 * goal_mocapid, new_goal_position);
  }
}

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
    1,  // steering
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
const std::array<std::string, 7> track_body_names = {
    "pelvis", "lhand", "rhand", "lshoulder", "rshoulder", "lhip", "rhip",
};
// compute mocap translations and rotations
void move_mocap_poses(mjtNum *result, const mjModel *model, const mjData *data,
                      std::vector<double> parameters, int mode) {
  // todo move residual here

  std::vector<mjtNum> modified_mocap_pos(3 * (model->nmocap - 1));

  // Compute interpolated frame.
  mju_scl(modified_mocap_pos.data(), model->key_mpos + 3 * model->nmocap * mode,
          1, 3 * (model->nmocap - 1));
  double skateboard_center[3];
  int skateboard_body_id = mj_name2id(model, mjOBJ_XBODY, "skateboard");

  // move mpos to x,y position of skateboard
  mju_copy(skateboard_center, data->xpos + 3 * skateboard_body_id, 3);

  // print average center of mpos
  double average_mpos[2] = {0};

  // get average center of mpos
  for (int i = 0; i < model->nmocap - 1; i++) {
    average_mpos[0] += modified_mocap_pos[3 * i + 0];
    average_mpos[1] += modified_mocap_pos[3 * i + 1];

    modified_mocap_pos[3 * i + 0] += skateboard_center[0];
    modified_mocap_pos[3 * i + 1] += skateboard_center[1];
    modified_mocap_pos[3 * i + 2] += skateboard_center[2] - 0.1;
  }
  average_mpos[0] /= model->nmocap - 1;
  average_mpos[1] /= model->nmocap - 1;

  // subtract the difference between average_mpos and skateboard_center
  for (int i = 0; i < model->nmocap - 1; i++) {
    modified_mocap_pos[3 * i + 0] -= average_mpos[0];
    modified_mocap_pos[3 * i + 1] -= average_mpos[1];
  }

  double skateboard_heading = 0.0;
  double skateboard_xmat[9];
  mju_copy(skateboard_xmat, data->xmat + 9 * skateboard_body_id, 9);
  skateboard_heading = atan2(skateboard_xmat[3], skateboard_xmat[0]);
  skateboard_heading -= M_PI / 2.0;

  int goal_id = mj_name2id(model, mjOBJ_XBODY, "goal");
  if (goal_id < 0) mju_error("body 'goal' not found");

  int goal_mocap_id_ = model->body_mocapid[goal_id];
  if (goal_mocap_id_ < 0) mju_error("body 'goal' is not mocap");

  // get goal position
  double *goal_pos = data->mocap_pos + 3 * goal_mocap_id_;

  // Get goal heading from board position
  double goal_heading = atan2(goal_pos[1] - skateboard_center[1],
                              goal_pos[0] - skateboard_center[0]) -
                        M_PI / 2.0;
  ;

  // Calculate heading error using sine function
  double heading_error = sin(goal_heading - skateboard_heading) / 3;

  // Rotate the pixels in 3D space around the Z-axis (board_center)
  double mocap_tilt = parameters[mjpc::ParameterIndex(model, "Tilt ratio")];
  // # TODO(eliasmikkola): fix ParameterIndex not working (from utilities.h)
  // tilt angle max is PI/3
  double tilt_angle =
      (mju_min(0.5, mju_max(-0.5, heading_error)) * M_PI / 2.0) * mocap_tilt;
  // tilt_angle = 0.0;
  for (int i = 0; i < model->nmocap - 1; i++) {
    // Get the pixel position relative to the board_center
    double rel_x = modified_mocap_pos[3 * i + 0] - skateboard_center[0];
    double rel_y = modified_mocap_pos[3 * i + 1] - skateboard_center[1];

    // perform rotation of tilt_angle around the Y-axis
    double rotated_z = rel_x * sin(tilt_angle) +
                       modified_mocap_pos[3 * i + 2] * cos(tilt_angle);
    rel_x = rel_x * cos(tilt_angle) -
            modified_mocap_pos[3 * i + 2] * sin(tilt_angle);

    // Perform rotation around the Z-axis
    double rotated_x =
        cos(skateboard_heading) * rel_x - sin(skateboard_heading) * rel_y;
    double rotated_y =
        sin(skateboard_heading) * rel_x + cos(skateboard_heading) * rel_y;

    // Update the rotated pixel positions in modified_mocap_pos
    modified_mocap_pos[3 * i + 0] = skateboard_center[0] + rotated_x;
    modified_mocap_pos[3 * i + 1] = skateboard_center[1] + rotated_y;
    modified_mocap_pos[3 * i + 2] = rotated_z;

    rotated_x =
        cos(skateboard_heading) * rel_x - sin(skateboard_heading) * rel_y;
    rotated_y =
        sin(skateboard_heading) * rel_x + cos(skateboard_heading) * rel_y;
  }
  mju_copy(result, modified_mocap_pos.data(), 3 * (model->nmocap - 1));
}

}  // namespace

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

std::vector<double> Steering::ResidualFn::ComputeTrackingResidual(
    const mjModel *model, const mjData *data) const {
  std::vector<mjtNum> mocap_translated(3 * (model->nmocap - 1));

  move_mocap_poses(mocap_translated.data(), model, data, parameters_,
                   current_mode_);

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
    mju_scl3(result,
             mocap_translated.data() + 3 * (model->nmocap - 1) * key_index_0 +
                 3 * body_mocapid,
             weight_0);

    // next frame
    mju_addToScl3(result,
                  mocap_translated.data() +
                      3 * (model->nmocap - 1) * key_index_1 + 3 * body_mocapid,
                  weight_1);
  };

  auto get_body_sensor_pos = [&](const std::string &body_name,
                                 double result[3]) {
    std::string pos_sensor_name = "tracking_pos[" + body_name + "]";
    double *sensor_pos =
        mjpc::SensorByName(model, data, pos_sensor_name.c_str());
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
  mju_scl3(avg_mpos, avg_mpos, 1.0 / num_body);
  mju_scl3(avg_sensor_pos, avg_sensor_pos, 1.0 / num_body);

  // residual_to_return for averages
  residual_to_return.push_back(avg_mpos[0] - avg_sensor_pos[0]);
  residual_to_return.push_back(avg_mpos[1] - avg_sensor_pos[1]);
  residual_to_return.push_back(avg_mpos[2] - avg_sensor_pos[2]);

  for (const auto &body_name : track_body_names) {
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
  for (const auto &body_name : track_body_names) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    std::string linvel_sensor_name = "tracking_linvel[" + body_name + "]";
    int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
    assert(0 <= mocap_body_id);
    int body_mocapid = model->body_mocapid[mocap_body_id];
    assert(0 <= body_mocapid);

    // Compute finite-difference velocity for x, y, z components
    double fd_velocity[3];  // Finite-difference velocity
    for (int i = 0; i < 3; ++i) {
      fd_velocity[i] = (model->key_mpos[3 * model->nmocap * key_index_1 +
                                        3 * body_mocapid + i] -
                        model->key_mpos[3 * model->nmocap * key_index_0 +
                                        3 * body_mocapid + i]) *
                       kFps;
    }

    // Get current velocity from sensor
    double *sensor_linvel =
        mjpc::SensorByName(model, data, linvel_sensor_name.c_str());

    for (int i = 0; i < 3; ++i) {
      double velocity_residual = fd_velocity[i] - sensor_linvel[i];
      residual_to_return.push_back(velocity_residual);
    }
  }
  return residual_to_return;
}

std::array<double, 2> Steering::ResidualFn::ComputeFootPositionsResidual(
    const mjModel *model, const mjData *data) const {
  // ----- Skateboard: Feet should be on the skateboard ----- //
  double *back_plate_pos = mjpc::SensorByName(model, data, "track-back-plate");
  double *tail_pos = mjpc::SensorByName(model, data, "track-tail");
  double *front_plate_pos =
      mjpc::SensorByName(model, data, "track-front-plate");

  double *left_foot_pos = mjpc::SensorByName(model, data, "tracking_foot_left");
  double *right_foot_pos =
      mjpc::SensorByName(model, data, "tracking_foot_right");

  double right_feet_slider =
      parameters_[mjpc::ParameterIndex(model, "Right Foot Pos")];
  double left_feet_slider =
      parameters_[mjpc::ParameterIndex(model, "Left Foot Pos")];

  // calculate x-wise difference between the plates, based on right_feet_slider
  double plate_distance_x = mju_abs(back_plate_pos[0] - front_plate_pos[0]);
  double plate_distance_y = mju_abs(back_plate_pos[1] - front_plate_pos[1]);
  // calculate the x position of the line set by the plates
  double right_feet_x =
      front_plate_pos[0] - right_feet_slider * plate_distance_x;
  double right_feet_y =
      front_plate_pos[1] - right_feet_slider * plate_distance_y;

  // left feet error, distance to back plate position
  double distance_x = mju_abs(left_foot_pos[0] - back_plate_pos[0]);
  double distance_y = mju_abs(left_foot_pos[1] - back_plate_pos[1]);
  double distance_z = mju_abs(left_foot_pos[2] - back_plate_pos[2]);
  if (left_feet_slider > 0) {
    distance_x = mju_abs(left_foot_pos[0] - tail_pos[0]);
    distance_y = mju_abs(left_foot_pos[1] - tail_pos[1]);
    distance_z = mju_abs(left_foot_pos[2] - tail_pos[2]);
  }
  double left_feet_error =
      mju_sqrt(distance_x * distance_x + distance_y * distance_y +
               (distance_z * distance_z));

  // right feet error, distance to front plate position
  distance_x = mju_abs(right_foot_pos[0] - right_feet_x);
  distance_y = mju_abs(right_foot_pos[1] - right_feet_y);
  distance_z = mju_abs(right_foot_pos[2] - front_plate_pos[2]);
  double right_feet_error =
      mju_sqrt(distance_x * distance_x + distance_y * distance_y +
               (distance_z * distance_z));

  return {left_feet_error, right_feet_error};
}

std::array<double, 2> Steering::ResidualFn::ComputeBoardHeadingResidual(
    const mjModel *model, const mjData *data) const {
  double skateboard_xmat[9];
  mju_copy(skateboard_xmat, data->xmat + 9 * skateboard_body_id_, 9);

  std::array<double, 2> skateboard_heading = {
      skateboard_xmat[0],
      skateboard_xmat[3],
  };
  double skateboard_yaw = atan2(skateboard_heading[1], skateboard_heading[0]);
  int heading_parameter_index = ParameterIndex(model, "Heading");
  assert(0 <= heading_parameter_index);
  double skateboard_heading_target_offset =
      parameters_[heading_parameter_index];
  double cos_offset = std::cos(skateboard_heading_target_offset);
  double sin_offset = std::sin(skateboard_heading_target_offset);

  std::array<double, 2> skateboard_heading_target = {
      (skateboard_heading[0] * cos_offset - skateboard_heading[1] * sin_offset),
      (skateboard_heading[0] * sin_offset +
       skateboard_heading[1] * cos_offset)};
  std::array<double, 2> result = {
      skateboard_heading[0] - skateboard_heading_target[0],
      skateboard_heading[1] - skateboard_heading_target[1],
  };

  // double skateboard_yaw = atan2(skateboard_heading[1],
  // skateboard_heading[0]);
  // // double skateboard_roll = atan2(skateboard_xmat[7], skateboard_xmat[8]);
  // // double skateboard_pitch = atan2(skateboard_xmat[1], skateboard_xmat[0]);

  // std::array<mjtNum, 2> skateboard_center = {
  //   data->xpos[3 * skateboard_body_id_ + 0],
  //   data->xpos[3 * skateboard_body_id_ + 1],
  // };

  // std::array<mjtNum, 2> goal_position = {
  //     data->mocap_pos[3 * goal_body_mocap_id_ + 0],
  //     data->mocap_pos[3 * goal_body_mocap_id_ + 1],
  // };

  // std::array<mjtNum, 2> skateboard_position = {
  //     data->xpos[3 * skateboard_body_id_ + 0],
  //     data->xpos[3 * skateboard_body_id_ + 1],
  // };

  // std::array<mjtNum, 2> board_to_goal = {
  //     goal_position[0] - skateboard_position[0],
  //     goal_position[1] - skateboard_position[1],
  // };

  // mju_normalize3(board_to_goal.data());

  // mjtNum target_yaw = atan2(board_to_goal[1], board_to_goal[0]);

  // // Normalize yaw error to [0, pi]. Should be at minimum, `0`, when heading
  // // faces the goal and maximum, `pi`, when tail is facing the goal.
  // auto normalize_angle = [](double angle) {
  //   while (angle < -M_PI) angle += 2 * M_PI;
  //   while (+M_PI < angle) angle -= 2 * M_PI;
  //   return angle;
  // };

  // double yaw_error = normalize_angle(target_yaw - skateboard_yaw);
  // double yaw_error_raw = yaw_error;

  // // // parameter "Heading clamp"
  // // double clamp_l = parameters_[mjpc::ParameterIndex(model, "Heading clamp
  // l")];
  // // double clamp_k = parameters_[mjpc::ParameterIndex(model, "Heading clamp
  // k")];

  // // auto soft_clamp = [](double x, double limit, double k) {
  // //   return limit * std::tanh(x / limit * k);
  // // };
  // // yaw_error = soft_clamp(yaw_error, clamp_l, clamp_k);

  // std::array<double, 1> result = {yaw_error};

  // if (jiiri++ % 5000 == 0) {
  //   printf("\n======= HEADING RESIDUAL =======\n");
  //   printf("skateboard_body_id: %d\n", skateboard_body_id_);
  //   printf("xmat: [%f, %f, %f, %f, %f, %f, %f, %f, %f]\n",
  //   skateboard_xmat[0],
  //          skateboard_xmat[1], skateboard_xmat[2], skateboard_xmat[3],
  //          skateboard_xmat[4], skateboard_xmat[5], skateboard_xmat[6],
  //          skateboard_xmat[7], skateboard_xmat[8]);
  //   printf("board_heading_residual: %f\n", result[0]);
  //   printf("skateboard_yaw: %f\n", skateboard_yaw);
  //   printf("target_yaw: %f\n", target_yaw);
  //   printf("yaw_error: %f\n", yaw_error);
  //   printf("yaw_error_raw: %f\n", yaw_error_raw);
  //   printf("goal_position: [%f, %f]\n", goal_position[0], goal_position[1]);
  //   printf("skateboard_position: [%f, %f]\n\n", skateboard_position[0],
  //          skateboard_position[1]);
  // }
  return result;
}

std::array<mjtNum, 3> Steering::ResidualFn::ComputeBoardVelocityResidual(
    const mjModel *model, const mjData *data) const {
  std::array<mjtNum, 3> skateboard_linear_velocity_target = {
      parameters_[ParameterIndex(model, "Velocity")],
      0.0,
      0.0,
  };
  double *skateboard_framelinvel =
      SensorByName(model, data, "skateboard_framelinvel");
  std::array<mjtNum, 3> skateboard_linear_velocity_global = {
      skateboard_framelinvel[0],
      skateboard_framelinvel[1],
      skateboard_framelinvel[2],
  };

  std::array<mjtNum, 9> skateboard_xmat;
  mju_copy(skateboard_xmat.data(), data->xmat + 9 * skateboard_body_id_, 9);

  // Transform the global velocity to local velocity
  std::array<mjtNum, 3> skateboard_linear_velocity_local;
  mju_rotVecMatT(skateboard_linear_velocity_local.data(),
                 skateboard_linear_velocity_global.data(),
                 skateboard_xmat.data());

  // NOTE: we add small tolerance to x here.
  std::array<mjtNum, 3> result = {
      skateboard_linear_velocity_target[0] -
          skateboard_linear_velocity_local[0] - 0.03,
      skateboard_linear_velocity_target[1] -
          skateboard_linear_velocity_local[1],
      skateboard_linear_velocity_target[2] -
          skateboard_linear_velocity_global[2],
  };

  return result;
}

void Steering::ModifyScene(const mjModel *model, const mjData *data,
                           mjvScene *scene) const {}

void Steering::ResetLocked(const mjModel *model) {
  residual_.skateboard_xbody_id_ = mj_name2id(model, mjOBJ_XBODY, "skateboard");
  if (residual_.skateboard_xbody_id_ < 0)
    mju_error("xbody 'skateboard' not found");

  residual_.skateboard_body_id_ = mj_name2id(model, mjOBJ_BODY, "skateboard");
  if (residual_.skateboard_body_id_ < 0)
    mju_error("body 'skateboard' not found");

  // TODO(hartikainen): `mjOBJ_XBODY` or `mjOBJ_BODY`? Does it matter?
  residual_.goal_body_id_ = mj_name2id(model, mjOBJ_XBODY, "goal");
  if (residual_.goal_body_id_ < 0) mju_error("body 'goal' not found");
  residual_.goal_body_mocap_id_ = model->body_mocapid[residual_.goal_body_id_];
  if (residual_.goal_body_mocap_id_ < 0) mju_error("body 'goal' is not mocap");
}
void Steering::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                    double *residual) const {
  // ----- residual ----- //
  int counter = 0;
  // ----- joint velocity ----- //
  int n_humanoid_joints = model->nv - 6 - 6 - 7;
  mju_copy(residual + counter, data->qvel + 6, n_humanoid_joints);

  counter += n_humanoid_joints;

  // ----- action ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // Tracking Residual
  auto tracking_residual = ComputeTrackingResidual(model, data);
  mju_copy(residual + counter, tracking_residual.data(),
           tracking_residual.size());
  counter += tracking_residual.size();

  // Foot Positions Residual
  auto foot_positions_residual = ComputeFootPositionsResidual(model, data);
  mju_copy(residual + counter, foot_positions_residual.data(),
           foot_positions_residual.size());
  counter += foot_positions_residual.size();

  // Board Heading Residual
  auto board_heading_residual = ComputeBoardHeadingResidual(model, data);
  mju_copy(residual + counter, board_heading_residual.data(),
           board_heading_residual.size());
  counter += board_heading_residual.size();

  // Goal Position Residual
  auto board_velocity_residual = ComputeBoardVelocityResidual(model, data);

  mju_copy(residual + counter, board_velocity_residual.data(),
           board_velocity_residual.size());
  counter += board_velocity_residual.size();

  // TODO(eliasmikkola): fill missing skateboard residuals

  CheckSensorDim(model, counter);
}

// --------------------- Transition for humanoid task -------------------------
//   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
//   Linearly interpolate between two consecutive key frames in order to
//   smooth the transitions between keyframes.
// ----------------------------------------------------------------------------
void Steering::TransitionLocked(mjModel *model, mjData *d) {
  assert(residual_.skateboard_body_id_ >= 0);
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

  mjtNum *mocap_pos_0 = mj_stackAllocNum(d, 3 * (model->nmocap - 1));
  mjtNum *mocap_pos_1 = mj_stackAllocNum(d, 3 * (model->nmocap - 1));

  // Compute interpolated frame.
  mju_scl(mocap_pos_0, model->key_mpos + 3 * model->nmocap * key_index_0,
          weight_0, 3 * (model->nmocap - 1));

  mju_scl(mocap_pos_1, model->key_mpos + 3 * model->nmocap * key_index_1,
          weight_1, 3 * (model->nmocap - 1));

  mju_copy(d->mocap_pos, mocap_pos_0, 3 * (model->nmocap - 1));
  mju_addTo(d->mocap_pos, mocap_pos_1, 3 * (model->nmocap - 1));

  mjtNum *mocap_pos_result = mj_stackAllocNum(d, 3 * (model->nmocap - 1));
  move_mocap_poses(mocap_pos_result, model, d, parameters, mode);
  mju_copy(d->mocap_pos, mocap_pos_result, 3 * (model->nmocap - 1));
  move_goal(model, d, parameters, mode);
  mj_freeStack(d);
}

}  // namespace mjpc::humanoid

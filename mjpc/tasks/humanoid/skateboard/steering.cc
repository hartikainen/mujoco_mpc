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
}  // namespace

namespace mjpc::humanoid {

std::string Steering::XmlPath() const {
  return GetModelPath("humanoid/skateboard/steering-task.xml");
}
std::string Steering::Name() const { return "Humanoid Skateboard Steer"; }


std::array<double, 2>
Steering::ResidualFn::ComputeBoardHeadingResidual(const mjModel *model,
                                                  const mjData *data) const {
  int heading_target_offset_parameter_index = ParameterIndex(model, "Heading");
  assert(0 <= heading_target_offset_parameter_index);
  double heading_target_offset =
      parameters_[heading_target_offset_parameter_index];

  double xmat[9];
  mju_copy(xmat, data->xmat + 9 * skateboard_body_id_, 9);

  std::array<double, 2> heading = {xmat[0], xmat[3]};
  mju_normalize(heading.data(), 2);

  std::array<mjtNum, 2> xpos = {
      data->xpos[3 * skateboard_body_id_ + 0],
      data->xpos[3 * skateboard_body_id_ + 1],
  };

  std::array<mjtNum, 2> xpos_target = {
      data->mocap_pos[3 * goal_body_mocap_id_ + 0],
      data->mocap_pos[3 * goal_body_mocap_id_ + 1],
  };

  std::array<mjtNum, 2> board_to_goal = {
      xpos_target[0] - xpos[0],
      xpos_target[1] - xpos[1],
  };

  mju_normalize(board_to_goal.data(), 2);

  std::array<double, 2> result = {
      heading[0] - board_to_goal[0],
      heading[1] - board_to_goal[1],
  };

  if (jiiri++ % 10000 == 0) {
    printf("\n======= HEADING RESIDUAL =======\n");
    printf("skateboard_body_id: %d\n", skateboard_body_id_);
    printf("xmat: [%f, %f, %f, %f, %f, %f, %f, %f, %f]\n", xmat[0],
           xmat[1], xmat[2], xmat[3],
           xmat[4], xmat[5], xmat[6],
           xmat[7], xmat[8]);

    printf("board_heading_residual: [%.3f, %.3f]\n", result[0], result[1]);
    printf("skateboard_yaw: %f\n", atan2(heading[1], heading[0]));
    printf("target_yaw: %f\n", atan2(board_to_goal[1], board_to_goal[0]));
    printf("yaw_error: %f\n", atan2(heading[1], heading[0]) - atan2(board_to_goal[1], board_to_goal[0]));
    printf("heading: [%f, %f]\n", heading[0], heading[1]);
    printf("skateboard_position: [%f, %f]\n", xpos[0],
           xpos[1]);
    printf("goal_position: [%f, %f]\n", xpos_target[0], xpos_target[1]);
    printf("board_to_goal: [%f, %f]\n\n", board_to_goal[0], board_to_goal[1]);
  }
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

  // if (jiiri++ % 5001 == 1) {
  //   printf("\n======= VELOCITY RESIDUAL =======\n");
  //   printf("skateboard_linear_velocity_global: [%f, %f, %f]\n",
  //          skateboard_linear_velocity_global[0],
  //          skateboard_linear_velocity_global[1],
  //          skateboard_linear_velocity_global[2]);
  //   printf("skateboard_linear_velocity_local: [%f, %f, %f]\n",
  //          skateboard_linear_velocity_local[0],
  //          skateboard_linear_velocity_local[1],
  //          skateboard_linear_velocity_local[2]);
  //   printf("board_velocity_residual: [%f, %f, %f]\n",
  //          result[0], result[1], result[2]);
  // }

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

  // ----- action ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

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

void Steering::TransitionLocked(mjModel *model, mjData *d) {
  assert(residual_.skateboard_body_id_ >= 0);

  move_goal(model, d, parameters, mode);
  mj_freeStack(d);
}

}  // namespace mjpc::humanoid

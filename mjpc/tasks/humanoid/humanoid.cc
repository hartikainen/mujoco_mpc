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

#include "tasks/humanoid/humanoid.h"
#include "mujoco/mjmodel.h"
#include "tasks/humanoid_cmu/motions.h"

#include <array>
#include <cmath>
#include <iostream>
#include <map>

#include <mujoco/mujoco.h>
#include "utilities.h"

namespace mjpc {

// def deep_mimic_multiclip(walker_features, reference_features, **unused_kwargs):
//   """A reward similar to deepmimic (larger weight on joints_velocity)."""
//   differences = _compute_squared_differences(walker_features,
//                                              reference_features)
//   com = .1 * np.exp(-10 * differences['center_of_mass'])
//   joints_velocity = 1.0 * np.exp(-0.1 * differences['joints_velocity'])
//   appendages = 0.15 * np.exp(-40. * differences['appendages'])
//   body_quaternions = 0.65 * np.exp(-2 * differences['body_quaternions'])
//   terms = {
//       'center_of_mass': com,
//       'joints_velocity': joints_velocity,
//       'appendages': appendages,
//       'body_quaternions': body_quaternions
//   }
//   reward = sum(terms.values())
//   return reward, terms, _sort_dict(terms)


// const mjpc::MocapMotions mocap_motions;
// ------------------ Residuals for humanoid stand task ------------
//   Number of residuals: 16
//     Residual (0): `pelvis` mocap error
//     Residual (1): `toe_left` mocap error
//     Residual (2): `toe_right` mocap error
//     Residual (3): `heel_left` mocap error
//     Residual (4): `heel_right` mocap error
//     Residual (5): `knee_left` mocap error
//     Residual (6): `knee_right` mocap error
//     Residual (7): `left_hand_left` mocap error
//     Residual (8): `right_hand_right` mocap error
//     Residual (9): `elbow_left` mocap error
//     Residual (10): `elbow_right` mocap error
//     Residual (11): `shoulder1_left` mocap error
//     Residual (12): `shoulder1_right` mocap error
//     Residual (13): `head` mocap error
//     Residual (14): `hip_z_left` mocap error
//     Residual (15): `hip_z_right` mocap error
//   Number of parameters: 1
//     Parameter (0): TODO(hartikainen): parameterize the loss.
// ----------------------------------------------------------------
void Humanoid::ResidualStand(const double* parameters, const mjModel* model,
                             const mjData* data, double* residual) {
  int counter = 0;

  // ----- Height: head feet vertical error ----- //

  // feet sensor positions
  double* f1_position = mjpc::SensorByName(model, data, "sp0");
  double* f2_position = mjpc::SensorByName(model, data, "sp1");
  double* f3_position = mjpc::SensorByName(model, data, "sp2");
  double* f4_position = mjpc::SensorByName(model, data, "sp3");
  double* head_position = mjpc::SensorByName(model, data, "head_position");
  double head_feet_error =
      head_position[2] - 0.25 * (f1_position[2] + f2_position[2] +
                                 f3_position[2] + f4_position[2]);
  residual[counter++] = head_feet_error - parameters[0];

  // ----- Balance: CoM-feet xy error ----- //

  // capture point
  double* com_position = mjpc::SensorByName(model, data, "torso_subtreecom");
  double* com_velocity = mjpc::SensorByName(model, data, "torso_subtreelinvel");
  double kFallTime = 0.2;
  double capture_point[3] = {com_position[0], com_position[1], com_position[2]};
  mju_addToScl3(capture_point, com_velocity, kFallTime);

  // average feet xy position
  double fxy_avg[2] = {0.0};
  mju_addTo(fxy_avg, f1_position, 2);
  mju_addTo(fxy_avg, f2_position, 2);
  mju_addTo(fxy_avg, f3_position, 2);
  mju_addTo(fxy_avg, f4_position, 2);
  mju_scl(fxy_avg, fxy_avg, 0.25, 2);

  mju_subFrom(fxy_avg, capture_point, 2);
  double com_feet_distance = mju_norm(fxy_avg, 2);
  residual[counter++] = com_feet_distance;

  // ----- COM xy velocity should be 0 ----- //
  mju_copy(&residual[counter], com_velocity, 2);
  counter += 2;

  // ----- joint velocity ----- //
  mju_copy(residual + counter, data->qvel, model->nv);
  counter += model->nv;

  // ----- action ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        counter);
  }
}

// ------------------ Residuals for humanoid walk task ------------
//   Number of residuals:
//     Residual (0): torso height
//     Residual (1): pelvis-feet aligment
//     Residual (2): balance
//     Residual (3): upright
//     Residual (4): posture
//     Residual (5): walk
//     Residual (6): move feet
//     Residual (7): control
//   Number of parameters:
//     Parameter (0): torso height goal
//     Parameter (1): speed goal
// ----------------------------------------------------------------
void Humanoid::ResidualWalk(const double* parameters, const mjModel* model,
                            const mjData* data, double* residual) {
  int counter = 0;

  // ----- torso height ----- //
  double torso_height = mjpc::SensorByName(model, data, "torso_position")[2];
  residual[counter++] = torso_height - parameters[0];

  // ----- pelvis / feet ----- //
  double* foot_right = mjpc::SensorByName(model, data, "foot_right");
  double* foot_left = mjpc::SensorByName(model, data, "foot_left");
  double pelvis_height = mjpc::SensorByName(model, data, "pelvis_position")[2];
  residual[counter++] =
      0.5 * (foot_left[2] + foot_right[2]) - pelvis_height - 0.2;

  // ----- balance ----- //
  // capture point
  double* subcom = mjpc::SensorByName(model, data, "torso_subcom");
  double* subcomvel = mjpc::SensorByName(model, data, "torso_subcomvel");

  double capture_point[3];
  mju_addScl(capture_point, subcom, subcomvel, 0.3, 3);
  capture_point[2] = 1.0e-3;

  // project onto line segment

  double axis[3];
  double center[3];
  double vec[3];
  double pcp[3];
  mju_sub3(axis, foot_right, foot_left);
  axis[2] = 1.0e-3;
  double length = 0.5 * mju_normalize3(axis) - 0.05;
  mju_add3(center, foot_right, foot_left);
  mju_scl3(center, center, 0.5);
  mju_sub3(vec, capture_point, center);

  // project onto axis
  double t = mju_dot3(vec, axis);

  // clamp
  t = mju_max(-length, mju_min(length, t));
  mju_scl3(vec, axis, t);
  mju_add3(pcp, vec, center);
  pcp[2] = 1.0e-3;

  // is standing
  double standing =
      torso_height / mju_sqrt(torso_height * torso_height + 0.45 * 0.45) - 0.4;

  mju_sub(&residual[counter], capture_point, pcp, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);

  counter += 2;

  // ----- upright ----- //
  double* torso_up = mjpc::SensorByName(model, data, "torso_up");
  double* pelvis_up = mjpc::SensorByName(model, data, "pelvis_up");
  double* foot_right_up = mjpc::SensorByName(model, data, "foot_right_up");
  double* foot_left_up = mjpc::SensorByName(model, data, "foot_left_up");
  double z_ref[3] = {0.0, 0.0, 1.0};

  // torso
  residual[counter++] = torso_up[2] - 1.0;

  // pelvis
  residual[counter++] = 0.3 * (pelvis_up[2] - 1.0);

  // right foot
  mju_sub3(&residual[counter], foot_right_up, z_ref);
  mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
  counter += 3;

  mju_sub3(&residual[counter], foot_left_up, z_ref);
  mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
  counter += 3;

  // ----- posture ----- //
  mju_copy(&residual[counter], data->qpos + 7, model->nq - 7);
  counter += model->nq - 7;

  // ----- walk ----- //
  double* torso_forward = mjpc::SensorByName(model, data, "torso_forward");
  double* pelvis_forward = mjpc::SensorByName(model, data, "pelvis_forward");
  double* foot_right_forward =
      mjpc::SensorByName(model, data, "foot_right_forward");
  double* foot_left_forward =
      mjpc::SensorByName(model, data, "foot_left_forward");

  double forward[2];
  mju_copy(forward, torso_forward, 2);
  mju_addTo(forward, pelvis_forward, 2);
  mju_addTo(forward, foot_right_forward, 2);
  mju_addTo(forward, foot_left_forward, 2);
  mju_normalize(forward, 2);

  // com vel
  double* waist_lower_subcomvel =
      mjpc::SensorByName(model, data, "waist_lower_subcomvel");
  double* torso_velocity = mjpc::SensorByName(model, data, "torso_velocity");
  double com_vel[2];
  mju_add(com_vel, waist_lower_subcomvel, torso_velocity, 2);
  mju_scl(com_vel, com_vel, 0.5, 2);

  // walk forward
  residual[counter++] =
      standing * (mju_dot(com_vel, forward, 2) - parameters[1]);

  // ----- move feet ----- //
  double* foot_right_velocity =
      mjpc::SensorByName(model, data, "foot_right_velocity");
  double* foot_left_velocity =
      mjpc::SensorByName(model, data, "foot_left_velocity");
  double move_feet[2];
  mju_copy(move_feet, com_vel, 2);
  mju_addToScl(move_feet, foot_right_velocity, -0.5, 2);
  mju_addToScl(move_feet, foot_left_velocity, -0.5, 2);

  mju_copy(&residual[counter], move_feet, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);
  counter += 2;

  // ----- control ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        counter);
  }
}


// std::array<double, 4> compute_target_root_quat(const double left_pelvis[3],
//                                                 const double mid_pelvis[3],
//                                                 const double right_pelvis[3]) {
//   // Given three points, return position and orientation of coordinate frame.
//   double root_pos[3] = {
//     0.5 * left_pelvis[0] + right_pelvis[0],
//     0.5 * left_pelvis[1] + right_pelvis[1],
//     0.5 * left_pelvis[2] + right_pelvis[2],
//   };

//   double left_dir[3] = {
//     left_pelvis[0] - root_pos[0],
//     left_pelvis[1] - root_pos[1],
//     left_pelvis[2] - root_pos[2],
//   };
//   mju_normalize3(left_dir);

//   double up_dir[3] = {
//     mid_pelvis[0] - root_pos[0],
//     mid_pelvis[1] - root_pos[1],
//     mid_pelvis[2] - root_pos[2],
//   };
//   mju_normalize3(up_dir);

//   double forward_dir[3] = {0.0};
//   mju_cross(forward_dir, left_dir, up_dir);
//   mju_normalize3(forward_dir);

//   double rot_mat[9] = {
//     forward_dir[0], left_dir[0], up_dir[0],
//     forward_dir[1], left_dir[1], up_dir[1],
//     forward_dir[2], left_dir[2], up_dir[2],
//   };
//   double target_root_quat[4] = {0.0};
//   mju_mat2Quat(target_root_quat, rot_mat);

//   return target_root_quat;
// }

// ------------- Residuals for humanoid tracking task -------------
//   Number of residuals: 6
//     Residual (0): Desired height
//     Residual (1): Balance: COM_xy - average(feet position)_xy
//     Residual (2): Com Vel: should be 0 and equal feet average vel
//     Residual (3): Control: minimise control
//     Residual (4): Joint vel: minimise joint velocity
//   Number of parameters: 1
//     Parameter (0): height_goal
// ----------------------------------------------------------------
void Humanoid::ResidualTrackSequence(const double* parameters, const mjModel* model,
                                     const mjData* data, double* residual) {
  double current_pose[16][3] = {{0.0}};
  double target_pose[16][3] = {{0.0}};
  int counter = 0;

  //  std::map<std::string, double[3]> current_pos = {
  //   {"pelvis", {0.0}},
  //   {"head", {0.0}},
  //   {"toe_left", {0.0}}, {"toe_right", {0.0}},
  //   {"heel_left", {0.0}}, {"heel_right", {0.0}},
  //   {"knee_left", {0.0}}, {"knee_right", {0.0}},
  //   {"left_hand_left", {0.0}}, {"right_hand_right", {0.0}},
  //   {"elbow_left", {0.0}}, {"elbow_right", {0.0}},
  //   {"shoulder1_left", {0.0}}, {"shoulder1_right", {0.0}},
  //   {"hip_z_left", 0.0}, {"hip_z_right", {0.0}},
  // };

  // std::map<std::string, double[3]> target_pos = {
  //   {"pelvis", {0.0}},
  //   {"head", {0.0}},
  //   {"toe_left", {0.0}}, {"toe_right", {0.0}},
  //   {"heel_left", {0.0}}, {"heel_right", {0.0}},
  //   {"knee_left", {0.0}}, {"knee_right", {0.0}},
  //   {"left_hand_left", {0.0}}, {"right_hand_right", {0.0}},
  //   {"elbow_left", {0.0}}, {"elbow_right", {0.0}},
  //   {"shoulder1_left", {0.0}}, {"shoulder1_right", {0.0}},
  //   {"hip_z_left", 0.0}, {"hip_z_right", {0.0}},
  // };

  std::array<std::string, 16> body_names = {
    "pelvis", "head", "ltoe", "rtoe", "lheel", "rheel", "lknee", "rknee",
    "lhand", "rhand", "lelbow", "relbow", "lshoulder", "rshoulder", "lhip",
    "rhip",
  };
  int i = 0;
  // int to_log_joint_index = 4;
  for (const auto& body_name : body_names) {
    std::string mocap_body_name = "mocap-" + body_name;
    std::string sensor_name = "tracking_pose[" + body_name + "]";
    int body_mocapid = model->body_mocapid[mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str())];
    assert (0 <= body_mocapid);
    mju_copy3(current_pose[i], mjpc::SensorByName(model, data, sensor_name.c_str()));
    mju_copy3(target_pose[i], data->mocap_pos + 3 * body_mocapid);
    // if (i == to_log_joint_index or i == 2) {
    //   std::cout << body_name << ", current_pose=[" << current_pose[i][0] << "; " << current_pose[i][1] << "; " << current_pose[i][2] << "]" << std::endl;
    //   std::cout << body_name << ", target_pose=[" << target_pose[i][0] << "; " << target_pose[i][1] << "; " << target_pose[i][2] << "]" << std::endl;
    // }
    i++;
  }

  double mid_pelvis[3] = {0.0};
  mju_copy3(
    mid_pelvis,
    data->mocap_pos + 3 * model->body_mocapid[mj_name2id(model, mjOBJ_BODY, "mocap-pelvis")]);
  double left_pelvis[3] = {0.0};
  mju_copy3(
    left_pelvis,
    data->mocap_pos + 3 * model->body_mocapid[mj_name2id(model, mjOBJ_BODY, "mocap-lhip")]);
  double right_pelvis[3] = {0.0};
  mju_copy3(
    right_pelvis,
    data->mocap_pos + 3 * model->body_mocapid[mj_name2id(model, mjOBJ_BODY, "mocap-rhip")]);

  // Given three points, return position and orientation of coordinate frame.
  double root_pos[3] = {
    0.5 * (left_pelvis[0] + right_pelvis[0]),
    0.5 * (left_pelvis[1] + right_pelvis[1]),
    0.5 * (left_pelvis[2] + right_pelvis[2]),
  };

  double left_dir[3] = {
    left_pelvis[0] - root_pos[0],
    left_pelvis[1] - root_pos[1],
    left_pelvis[2] - root_pos[2],
  };
  mju_normalize3(left_dir);

  double up_dir[3] = {
    mid_pelvis[0] - root_pos[0],
    mid_pelvis[1] - root_pos[1],
    mid_pelvis[2] - root_pos[2],
  };
  mju_normalize3(up_dir);

  double forward_dir[3] = {0.0};
  mju_cross(forward_dir, left_dir, up_dir);
  mju_normalize3(forward_dir);

  double rot_mat[9] = {
    forward_dir[0], left_dir[0], up_dir[0],
    forward_dir[1], left_dir[1], up_dir[1],
    forward_dir[2], left_dir[2], up_dir[2],
  };
  double target_root_quat[4] = {0.0};
  mju_mat2Quat(target_root_quat, rot_mat);

  double current_root_quat[4] = {0.0};
  mju_copy4(current_root_quat, mjpc::SensorByName(model, data, "tracking_quat[pelvis]"));
  double root_quat_error[3] = {0.0};
  mju_subQuat(root_quat_error, target_root_quat, current_root_quat);

  mju_copy3(&residual[counter], root_quat_error);
  counter += 3;

  for (int joint_index = 0; joint_index < 16; joint_index++) {
    double joint_pos_error[3] = {0.0};
    mju_sub3(joint_pos_error, target_pose[joint_index], current_pose[joint_index]);
    // if (joint_index == to_log_joint_index) {
    //   std::printf("error: [%f, %f, %f]\n", joint_pos_error[0], joint_pos_error[1], joint_pos_error[1]);
    // }
    mju_copy3(&residual[counter], joint_pos_error);
    counter += 3;
  }

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i=0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    std::printf("user_sensor_dim=%d, counter=%d", user_sensor_dim, counter);
    mju_error_i("mismatch between total user-sensor dimension"
                "and actual length of residual %d", user_sensor_dim);
  }

}

int Humanoid::TransitionTrackSequence(int state, const mjModel* model, mjData* data) {
  // TODO(hartikainen): Add distance-based target transition logic.

  // int sequence_length = 46;
  // TODO(hartikainen): is `data->time` the right thing to index here?
  float fps = 30.0;
  int step_index = data->time * fps;
  // std::printf("data->time=%f; step_index=%d\n", data->time, step_index);

  //  0: root        , root
  // 15: head_offset , head
  // 10: ltoes_offset, left_foot
  // 11: rtoes_offset, right_foot
  //  7: lheel_offset, left_ankle
  //  8: rheel_offset, right_ankle
  //  4: ltibia      , left_knee
  //  5: rtibia      , right_knee
  // 20: lwrist      , left_wrist
  // 21: rwrist      , right_wrist
  // 18: lradius     , left_elbow
  // 19: rradius     , right_elbow
  // 16: lhumerus    , left_shoulder
  // 17: rhumerus    , right_shoulder
  //  1: lfemur      , left_hip
  //  2: rfemur      , right_hip

  // std::array<std::string, 16> body_names = {
  //   "pelvis", "head", "ltoe", "rtoe", "lheel", "rheel", "lknee", "rknee",
  //   "lhand", "rhand", "lelbow", "relbow", "lshoulder", "rshoulder", "lhip",
  //   "rhip",
  // };
  // std::array<int, 16> kinematic_ids = {
  //   0, 15,  10,  11,  7,  8,  4,  5,  20,  21,  18,  19,  16,  17,  1,  2,
  // };

  // // model->key_mpos
  // std::printf("model->nmocap=%d", model->nmocap * 3 * step_index);

  // int i = 0;
  // for (const auto& body_name : body_names) {
  //   std::string mocap_body_name = "mocap-" + body_name;
  //   int body_mocapid = model->body_mocapid[mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str())];
  //   int kinematic_id = kinematic_ids[i];
  //   // std::cout << mocap_body_name << ": " << body_mocapid << ", " << kinematic_id << std::endl;
  //   assert (0 <= body_mocapid);
  //   mju_copy3(data->mocap_pos + 3 * body_mocapid, mocap_motions.motion_sequence[step_index][kinematic_id]);
  //   i++;
  // }

  // std::printf("data->time = %f; std::fmod(data->time, 1.0) = %f\n", data->time, std::fmod(data->time, 1.0));
  // if (0.998 <= std::fmod(data->time, 1.0)) {
  //   std::printf("data->time=%f\n", data->time);
  // }
  mju_copy(data->mocap_pos, model->key_mpos + model->nmocap * 3 * step_index, model->nmocap * 3);

  // TODO(hartikainen)
  // int new_state = (state + 1) % sequence_length;
  int new_state = 0;

  return new_state;
}

}  // namespace mjpc

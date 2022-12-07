#include "tasks/humanoid_cmu/humanoid_cmu.h"
#include "mujoco/mjmodel.h"
#include "tasks/humanoid_cmu/motions.h"

#include <iostream>

#include <mujoco/mujoco.h>
#include "utilities.h"

namespace mjpc {

const mjpc::MocapMotions mocap_motions;
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
void HumanoidCMU::ResidualTrackSequence(const double* parameters, const mjModel* model,
                                        const mjData* data, double* residual) {
  int counter = 0;

  int sequence_length = 45;

  int step_index = parameters[0] * double(sequence_length);

  double current_pose[16][3] = {{0.0}};
  double target_pose[16][3] = {{0.0}};

  mju_copy3(current_pose[0], mjpc::SensorByName(model, data, "tracking_pose[root]"));
  mju_copy3(target_pose[0], mocap_motions.motion_sequence[step_index][0]);
  mju_copy3(current_pose[1], mjpc::SensorByName(model, data, "tracking_pose[ltoes_offset]"));
  mju_copy3(target_pose[1], mocap_motions.motion_sequence[step_index][10]);
  mju_copy3(current_pose[2], mjpc::SensorByName(model, data, "tracking_pose[rtoes_offset]"));
  mju_copy3(target_pose[2], mocap_motions.motion_sequence[step_index][11]);
  mju_copy3(current_pose[3], mjpc::SensorByName(model, data, "tracking_pose[lheel_offset]"));
  mju_copy3(target_pose[3], mocap_motions.motion_sequence[step_index][7]);
  mju_copy3(current_pose[4], mjpc::SensorByName(model, data, "tracking_pose[rheel_offset]"));
  mju_copy3(target_pose[4], mocap_motions.motion_sequence[step_index][8]);
  mju_copy3(current_pose[5], mjpc::SensorByName(model, data, "tracking_pose[ltibia]"));
  mju_copy3(target_pose[5], mocap_motions.motion_sequence[step_index][4]);
  mju_copy3(current_pose[6], mjpc::SensorByName(model, data, "tracking_pose[rtibia]"));
  mju_copy3(target_pose[6], mocap_motions.motion_sequence[step_index][5]);
  mju_copy3(current_pose[7], mjpc::SensorByName(model, data, "tracking_pose[lwrist]"));
  mju_copy3(target_pose[7], mocap_motions.motion_sequence[step_index][20]);
  mju_copy3(current_pose[8], mjpc::SensorByName(model, data, "tracking_pose[rwrist]"));
  mju_copy3(target_pose[8], mocap_motions.motion_sequence[step_index][21]);
  mju_copy3(current_pose[9], mjpc::SensorByName(model, data, "tracking_pose[lradius]"));
  mju_copy3(target_pose[9], mocap_motions.motion_sequence[step_index][18]);
  mju_copy3(current_pose[10], mjpc::SensorByName(model, data, "tracking_pose[rradius]"));
  mju_copy3(target_pose[10], mocap_motions.motion_sequence[step_index][19]);
  mju_copy3(current_pose[11], mjpc::SensorByName(model, data, "tracking_pose[lhumerus]"));
  mju_copy3(target_pose[11], mocap_motions.motion_sequence[step_index][16]);
  mju_copy3(current_pose[12], mjpc::SensorByName(model, data, "tracking_pose[rhumerus]"));
  mju_copy3(target_pose[12], mocap_motions.motion_sequence[step_index][17]);
  mju_copy3(current_pose[13], mjpc::SensorByName(model, data, "tracking_pose[head_offset]"));
  mju_copy3(target_pose[13], mocap_motions.motion_sequence[step_index][15]);
  mju_copy3(current_pose[14], mjpc::SensorByName(model, data, "tracking_pose[lfemur]"));
  mju_copy3(target_pose[14], mocap_motions.motion_sequence[step_index][1]);
  mju_copy3(current_pose[15], mjpc::SensorByName(model, data, "tracking_pose[rfemur]"));
  mju_copy3(target_pose[15], mocap_motions.motion_sequence[step_index][2]);

  for (int joint_index = 0; joint_index < 16; joint_index++) {
    double norm[3] = {0.0};
    mju_sub3(norm, target_pose[joint_index], current_pose[joint_index]);
    residual[counter] = mju_norm(norm, 3);
    counter += 3;
  }

  // model->name_siteadr[mj_name2id(model, mjOBJ_SITE, "mocap_site[0]")]
  mju_copy3(data->site_xpos + 3 * 0, target_pose[0]);
  mju_copy3(data->site_xpos + 3 * 1, target_pose[1]);
  mju_copy3(data->site_xpos + 3 * 2, target_pose[2]);
  mju_copy3(data->site_xpos + 3 * 3, target_pose[3]);
  mju_copy3(data->site_xpos + 3 * 4, target_pose[4]);
  mju_copy3(data->site_xpos + 3 * 5, target_pose[5]);
  mju_copy3(data->site_xpos + 3 * 6, target_pose[6]);
  mju_copy3(data->site_xpos + 3 * 7, target_pose[7]);
  mju_copy3(data->site_xpos + 3 * 8, target_pose[8]);
  mju_copy3(data->site_xpos + 3 * 9, target_pose[9]);
  mju_copy3(data->site_xpos + 3 * 10, target_pose[10]);
  mju_copy3(data->site_xpos + 3 * 11, target_pose[11]);
  mju_copy3(data->site_xpos + 3 * 12, target_pose[12]);
  mju_copy3(data->site_xpos + 3 * 13, target_pose[13]);
  mju_copy3(data->site_xpos + 3 * 14, target_pose[14]);
  mju_copy3(data->site_xpos + 3 * 15, target_pose[15]);

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i=0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i("mismatch between total user-sensor dimension "
                "and actual length of residual %d", counter);
  }

}

// ------------------ Residuals for humanoid stand task ------------
//   Number of residuals: 6
//     Residual (0): Desired height
//     Residual (1): Balance: COM_xy - average(feet position)_xy
//     Residual (2): Com Vel: should be 0 and equal feet average vel
//     Residual (3): Control: minimise control
//     Residual (4): Joint vel: minimise joint velocity
//   Number of parameters: 1
//     Parameter (0): height_goal
// ----------------------------------------------------------------
void HumanoidCMU::ResidualStand(const double* parameters, const mjModel* model,
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
  for (int i=0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i("mismatch between total user-sensor dimension "
                "and actual length of residual %d", counter);
  }
}

}  // namespace mjpc

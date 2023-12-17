#include "mjpc/tasks/skateboard/pendulum_3d/pendulum_3d.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"


namespace mjpc::pendulum3d {

std::string Skateboard::XmlPath() const {
  return GetModelPath("skateboard/pendulum_3d/task.xml");
}
std::string Skateboard::Name() const { return "Pendulum 3d"; }

int jiiri = 0;
Skateboard::ResidualFn::ResidualFn(const Skateboard* task) : mjpc::BaseResidualFn(task) {}

// ------------------ Residuals for pendulum skateboard task ------------
//   Number of residuals: 5
//     Residual (0): Desired height
//     Residual (1): Balance: COM_xy - average(feet position)_xy
//     Residual (2): Com Vel: should be 0 and equal feet average vel
//     Residual (3): Control: minimise control
//     Residual (4): Joint vel: minimise joint velocity
//   Number of parameters: 1
//     Parameter (0): height_goal
//     Parameter (1): speed_goal
// ----------------------------------------------------------------
void Skateboard::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                 double* residual) const {
  int counter = 0;
  // --------- user residual dimensions ---------
  // name="Height" dim="1"
  // name="Balance" dim="1"
  // name="CoM Vel." dim="2
  // name="Joint Vel." dim="2
  // name="Control" dim="2"
  // --------------------------------------------

  int skateboard_body_id_ = mj_name2id(model, mjOBJ_XBODY, "board_center");
  if (skateboard_body_id_ < 0) mju_error("body 'skateboard' not found");
  // `skateboard_xmat` is the flattened 9-dimensional global rotation matrix
  // describing the orientation of the board.
  double* skateboard_xmat = data->xmat + 9*skateboard_body_id_;


  //#####################
  //#     HEIGHT (dim=1)     # 
  //#####################

  double* cube_position = SensorByName(model, data, "cube_position");
  double cube_height_current = cube_position[2];
  double cube_height_goal = parameters_[ParameterIndex(model, "Height Goal")];
  residual[counter++] = cube_height_current - cube_height_goal;

  //#####################
  //#     VELOCITY (dim=1)     # 
  //#####################

  double linear_velocity_goal_x = parameters_[ParameterIndex(model, "Velocity")];

  mjtNum* linear_velocity_global = SensorByName(model, data, "pendulum_framelinvel");
  mjtNum linear_velocity_local[3];
  
  // Transform the global velocity to local velocity
  mju_rotVecMatT(linear_velocity_local, linear_velocity_global, skateboard_xmat);

  double linear_velocity_local_x = linear_velocity_local[0];
  double linear_velocity_local_y = linear_velocity_local[1];
  double linear_velocity_local_z = linear_velocity_local[2];

  residual[counter++] = (linear_velocity_goal_x - linear_velocity_local_x) + linear_velocity_local_y + linear_velocity_local_z;

  //#####################
  //# GOAL POSITION (dim=3, x,y,x)   # 
  //#####################

  // get mocap goal position
  int goal_id = mj_name2id(model, mjOBJ_XBODY, "goal");
  if (goal_id < 0) mju_error("body 'goal' not found");

  int goal_mocap_id_ = model->body_mocapid[goal_id];
  if (goal_mocap_id_ < 0) mju_error("body 'goal' is not mocap");

  // get goal position
  double* goal_pos = data->mocap_pos + 3*goal_mocap_id_;
  
  // get skateboard position
  double* skateboard_pos = data->xpos + 3*skateboard_body_id_;

  double skateboard_position_x = skateboard_pos[0];
  double skateboard_position_y = skateboard_pos[1];
  double skateboard_position_z = skateboard_pos[2];
  // absolute position error
  double position_error_x = goal_pos[0] - skateboard_position_x;
  position_error_x = abs(position_error_x);
  double position_error_y = goal_pos[1] - skateboard_position_y;
  position_error_y = abs(position_error_y);
  double position_error_z = goal_pos[2] - skateboard_position_z;
  position_error_z = abs(position_error_z);
  // get xyz wise distance to goal
  double error_total = sqrt(position_error_x*position_error_x + position_error_y*position_error_y + position_error_z*position_error_z);
  
  residual[counter++] = position_error_x;
  residual[counter++] = position_error_y;
  residual[counter++] = position_error_z;
  
  
  //#####################
  //# GOAL ORIENTATION (dim=1) # 
  //#####################

  // get board angle in global frame
  double skateboard_heading = atan2(skateboard_xmat[3], skateboard_xmat[0]);

  // Get goal heading from board position
  double goal_heading = atan2(goal_pos[1] - skateboard_position_y, goal_pos[0] - skateboard_position_x);

  // Calculate heading error using sine function
  double heading_error = sin(goal_heading - skateboard_heading);

  // Normalize heading error to the range of 0 to pi/2
  heading_error = fabs(heading_error);

  residual[counter++] = heading_error;

  if (jiiri++ % 30000 == 0) {
    // print positions
    printf("error_total=%f\n", error_total);
    printf("position_error_x=%f\n", position_error_x);
    printf("position_error_y=%f\n", position_error_y);
    printf("position_error_z=%f\n\n", position_error_z);
    // print heading values
    printf("skateboard_heading=%f\n", skateboard_heading);
    printf("goal_heading=%f\n", goal_heading);
    printf("heading_error=%f\n\n", heading_error);
  }





  // NOTE(hartikainen): simple sanity check to make sure the model has as many
  // joints as I would expect. This obviously doesn't guarantee anything about
  // the ordering.
  int skateboard_joint_nq = mjpc::CountJointNqUnderBody(model, "skateboard");
  int skateboard_joint_nv = mjpc::CountJointNvUnderBody(model, "skateboard");
  // 14 = 3 freejoint translation 
  //    + 4 freejoint quaternion
  //    + 2 trucks 
  //    + 4 wheels  
  //    + 1 x-throttle
  assert(skateboard_joint_nq == 14);
  // 13 = 3 + 3 + 4 * 1 = freejoint-translational + freejoint-angular + 4 * dim(wheel)
  assert(skateboard_joint_nv == 13);

  int pendulum_joint_nq = mjpc::CountJointNqUnderBody(model, "pendulum_root");
  int pendulum_joint_nv = mjpc::CountJointNvUnderBody(model, "pendulum_root");
  // 9 = 3 freejoint translation
  //   + 4 freejoint quaternion
  //   + 2 pendulum hinges
  assert(pendulum_joint_nq == 9);
  // 6 = 3 + 3 = freejoint-translational + freejoint-angular
  assert(pendulum_joint_nv == 8);

  if (jiiri++ < 1) mjpc::PrintJointDebugInfo(model);

  assert(model->nq == skateboard_joint_nq + pendulum_joint_nq);
  assert(model->nv == skateboard_joint_nv + pendulum_joint_nv);
  
  
  // //----- JOINT VELOCITY ----- //
  // NOTE(hartikainen): use `model->nu` because we almost always apply the
  // velocity regularization to the actuated joints.
  mju_copy(residual + counter, data->qvel + 6, model->nu);
  counter += model->nu;


  // // ----- ACTION ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;
  // counter +3= 10
  CheckSensorDim(model, counter);
}

}  // namespace mjpc::pendulum3d

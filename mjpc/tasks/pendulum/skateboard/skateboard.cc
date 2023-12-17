#include "mjpc/tasks/pendulum/skateboard/skateboard.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"


namespace mjpc::pendulum {

std::string Skateboard::XmlPath() const {
  return GetModelPath("pendulum/skateboard/task.xml");
}
std::string Skateboard::Name() const { return "Pendulum 2d"; }

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

  double* cube_position = SensorByName(model, data, "cube_position");
  double cube_height_current = cube_position[2];
  double cube_height_goal = parameters_[ParameterIndex(model, "Height Goal")];
  residual[counter++] = cube_height_current - cube_height_goal;
  // counter 1 
  if (jiiri++ % 1000 == 0) {
    printf("cube_height_current=%f\n", cube_height_current);
  }

  // ----- skateboard x-velocity ----- //
  // TODO(hartikainen): The only-x-based cost might be too simplistic.
  double linear_velocity_goal = parameters_[ParameterIndex(model, "Velocity")];
  double* linear_velocity_board = SensorByName(model, data, "skateboard_framelinvel");
  double* linear_velocity_cube = SensorByName(model, data, "cube_framelinvel");
  residual[counter++] = linear_velocity_goal - ((linear_velocity_board[0] + linear_velocity_cube[0])/2);
  // counter 2 

  // ----- skateboard heading ----- //
  // TODO(hartikainen): Move the model identifier access into `Reset`-function
  // as is done in `quadruped.cc`?
  int skateboard_body_id_ = mj_name2id(model, mjOBJ_XBODY, "skateboard");
  if (skateboard_body_id_ < 0) mju_error("body 'skateboard' not found");
  double* skateboard_xmat = data->xmat + 9*skateboard_body_id_;
  double skateboard_heading[2] = {skateboard_xmat[0], skateboard_xmat[3]};
  mju_normalize(skateboard_heading, 2);
  double heading_goal = parameters_[ParameterIndex(model, "Heading")];
  residual[counter++] = skateboard_heading[0] - mju_cos(heading_goal);
  // counter 3
  residual[counter++] = skateboard_heading[1] - mju_sin(heading_goal);
  // counter 4

  // ----- joint velocity ----- //
  // NOTE(hartikainen): use `model->nu` because we almost always apply the
  // velocity regularization to the actuated joints.
  mju_copy(residual + counter, data->qvel + 6, model->nu);
  counter += model->nu;


  // // ----- action ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;
  // counter +3= 10
  CheckSensorDim(model, counter);
}

}  // namespace mjpc::pendulum

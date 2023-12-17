#include "mjpc/tasks/skateboard/cart_pole/cart_pole.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"


namespace mjpc::skateboard {

std::string CartPole::XmlPath() const {
  return GetModelPath("skateboard/cart_pole/task.xml");
}
std::string CartPole::Name() const { return "Skateboard CartPole"; }

int iteration = 0;

// ------------------ Residuals for skateboard cartpole task ------------
//   Number of residuals: 5
//     Residual (0): Velocity
//     Residual (1-4): Goal Position
//     Residual (5): Goal Orientation
//     Residual (6): Upright
//     Residual (7-9): Joint Vel
//     Residual (10-12): Control

//   Number of parameters: 1
//     Parameter (0): Velocity
// ----------------------------------------------------------------
void CartPole::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                 double* residual) const {
  int counter = 0;

  // TODO(hartikainen): Move the model identifier access into `Reset`-function
  // as is done in `quadruped.cc`?
  int skateboard_body_id_ = mj_name2id(model, mjOBJ_XBODY, "skateboard");
  if (skateboard_body_id_ < 0) mju_error("body 'skateboard' not found");
  // `skateboard_xmat` is the flattened 9-dimensional global rotation matrix
  // describing the orientation of the board.
  double* skateboard_xmat = data->xmat + 9*skateboard_body_id_;


  // ----- skateboard x-velocity ----- //
  double linear_velocity_goal_x = parameters_[ParameterIndex(model, "Velocity")];

  mjtNum* linear_velocity_global = SensorByName(model, data, "board_linvel");

  mjtNum linear_velocity_local[3];
  // Transform the global velocity to local velocity
  mju_rotVecMatT(linear_velocity_local, linear_velocity_global, skateboard_xmat);

  double linear_velocity_local_x = linear_velocity_local[0];
  double linear_velocity_local_y = linear_velocity_local[1];
  double linear_velocity_local_z = linear_velocity_local[2];
  // double total_velocity = sqrt(linear_velocity_local_x*linear_velocity_local_x + linear_velocity_local_y*linear_velocity_local_y + linear_velocity_local_z*linear_velocity_local_z);
  // calculate total velocity error, now that the board can't move freely in y and z directions
  residual[counter++] = (linear_velocity_goal_x - linear_velocity_local_x) + linear_velocity_local_y + linear_velocity_local_z;

  // ----- skateboard Position ----- //
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
  // normalize to [0,1]
  residual[counter++] = position_error_x;
  residual[counter++] = position_error_y;
  residual[counter++] = position_error_z;
  
  // get board angle in global frame
  double skateboard_heading = atan2(skateboard_xmat[3], skateboard_xmat[0]);

  // Get goal heading from board position
  double goal_heading = atan2(goal_pos[1] - skateboard_position_y, goal_pos[0] - skateboard_position_x);

  // Calculate heading error using sine function
  double heading_error = sin(goal_heading - skateboard_heading);

  // Normalize heading error to the range of 0 to pi/2
  heading_error = fabs(heading_error);

  residual[counter++] = heading_error / 10;

  if (iteration++ % 30000 == 0) {
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
  
  // ----- pole upright ----- //
  // NOTE(hartikainen): This encourages the board to stay upright, that is, s.t.
  // the grip points upwards. Obviously, if we ever want to do flips or roll on
  // a ramp, we will have to remove or tweak this reward so as to not discourage
  // from those task behaviors.
  // `skateboard_xmat[8]` gives us the global z vector between `[-1, +1]` in the
  // world coordinates. It's `-1` when the board points towards gravity (i.e.
  // towards ground) and `+1` when the board is upright. We scale the cost to a
  // range between `[0, 1]` to make is nicer to deal with.
  // TODO(hartikainen): make sure that this is correct.
  mjtNum* pole_upright = SensorByName(model, data, "pole_z");
  printf("pole_upright=%f\n", pole_upright[2]);
  residual[counter++] = pole_upright[2] - 1.0;

  // ----- angular momentum ----- //
  // TODO(hartikainen): Add angular momentum velocity? Copied over from `quadruped.cc:214`.
  // mju_copy3(residual + counter, SensorByName(model, data, "skateboard_angmom"));
  // counter +=3;

  // ----- joint velocity ----- //
  // NOTE(hartikainen): We only regularize the `no-rider-{xy}` joints. We could
  // technically regularize the `no-rider-ball` too but given that this is just
  // a debug environment, no need to spend too much time on the regularization
  // costs. After all, they're mostly meant to make the behaviors look nicer (
  // although in some case my also contribute to stability as well.)

  // NOTE(hartikainen): simple sanity check to make sure the model has as many
  // joints as I would expect. This obviously doesn't guarantee anything about
  // the ordering.
  int skateboard_joint_nq = mjpc::CountJointNqUnderBody(model, "skateboard");
  int skateboard_joint_nv = mjpc::CountJointNvUnderBody(model, "skateboard");
  // 14 = 3 + 4 + 4 * 1 = freejoint-translation + freejoint-quaternion + 4*dim(wheel) + trucks + board tilt 
  assert(skateboard_joint_nq == 14);
  // 13 = 3 + 3 + 4 * 1 = freejoint-translational + freejoint-angular + 4 * dim(wheel)
  assert(skateboard_joint_nv == 13);

  int cart_pole_joint_nq = mjpc::CountJointNqUnderBody(model, "cartpole");
  int cart_pole_joint_nv = mjpc::CountJointNvUnderBody(model, "cartpole");
  // 7 = 3 + 4 = freejoint-translation + freejoint-quaternion
  assert(cart_pole_joint_nq == 9);
  // 6 = 3 + 3 = freejoint-translational + freejoint-angular
  assert(cart_pole_joint_nv == 8);

  if (iteration++ < 1) mjpc::PrintJointDebugInfo(model);

  assert(model->nq == skateboard_joint_nq + cart_pole_joint_nq);
  assert(model->nv == skateboard_joint_nv + cart_pole_joint_nv);

  mju_copy(residual + counter, data->qvel + skateboard_joint_nv, 3);
  //                                        │                    │
  //          11 for: skateboard nv (skip) ─┘                    │
  //           1 for `cartpole-{x}` joint (regularize throttle) ─┘
  counter += 3;

  // ----- action ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  CheckSensorDim(model, counter);
}

}  // namespace mjpc::skateboard

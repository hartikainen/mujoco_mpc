#include "mjpc/tasks/skateboard/humanoid/skateboard.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc::humanoid
{

  std::string Skateboard::XmlPath() const
  {
    return GetModelPath("skateboard/humanoid/task.xml");
  }
  std::string Skateboard::Name() const { return "Skateboard Humanoid"; }
  int jiiri = 0;

  // ------------------ Residuals for humanoid skateboard task ------------
  //   Number of residuals: 6
  //     Residual (0): Desired height
  //     Residual (1): Balance: COM_xy - average(feet position)_xy
  //     Residual (2): Com Vel: should be 0 and equal feet average vel
  //     Residual (3): Feet on Board: Feet should touch the skateboard.
  //     Residual (4): Control: minimise control
  //     Residual (5): Joint vel: minimise joint velocity
  //   Number of parameters: 1
  //     Parameter (0): height_goal
  // ----------------------------------------------------------------
  void Skateboard::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                        double *residual) const
  {
    int counter = 0;


    // ----- skateboard velocity ----- //

    double *linear_velocity_global = SensorByName(model, data, "skateboard_framelinvel");
    double* com_position = SensorByName(model, data, "torso_subtreecom");
    double* com_velocity = SensorByName(model, data, "torso_subtreelinvel");

    double linear_velocity_goal_x = parameters_[ParameterIndex(model, "Velocity")];

    int skateboard_body_id_ = mj_name2id(model, mjOBJ_XBODY, "board_center");
    if (skateboard_body_id_ < 0) mju_error("body 'skateboard' not found");
    // `skateboard_xmat` is the flattened 9-dimensional global rotation matrix
    // describing the orientation of the board.
    double* skateboard_xmat = data->xmat + 9*skateboard_body_id_;

    mjtNum linear_velocity_local[3];
    
    // Transform the global velocity to local velocity
    mju_rotVecMatT(linear_velocity_local, linear_velocity_global, skateboard_xmat);

    double linear_velocity_local_x = linear_velocity_local[0];
    double linear_velocity_local_y = linear_velocity_local[1];
    double linear_velocity_local_z = linear_velocity_local[2];

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
    // normalize to [0,1]
    residual[counter++] = position_error_x;
    residual[counter++] = position_error_y;
    residual[counter++] = position_error_z;
    
    // get board angle in global frame
    // TODO: (elias) should not matter if the nose or tail is pointing towards the goal
    double skateboard_heading = atan2(skateboard_xmat[3], skateboard_xmat[0]);

    // Get goal heading from board position
    double goal_heading = atan2(goal_pos[1] - skateboard_position_y, goal_pos[0] - skateboard_position_x);

    // Calculate heading error using sine function
    double heading_error = sin(goal_heading - skateboard_heading);

    // Normalize heading error to the range of 0 to pi/2
    heading_error = fabs(heading_error);

    residual[counter++] = heading_error;



    // ----- Height: head feet vertical error ----- //

    // feet sensor positions
    double *f1_position = SensorByName(model, data, "sp0");
    double *f2_position = SensorByName(model, data, "sp1");
    double *f3_position = SensorByName(model, data, "sp2");
    double *f4_position = SensorByName(model, data, "sp3");
    double *head_position = SensorByName(model, data, "head_position");
    double head_feet_error =
        head_position[2] - 0.25 * (f1_position[2] + f2_position[2] +
                                   f3_position[2] + f4_position[2]);
    residual[counter++] = head_feet_error - parameters_[0];


    // ----- Balance: CoM-feet xy error and board xy error ----- //

    // capture point
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

    // board xy error
    double board_xy[2] = {skateboard_pos[0], skateboard_pos[1]};
    
    double board_distance = com_position[0] - board_xy[0] + com_position[1] - board_xy[1];
    board_distance = abs(board_distance);

    residual[counter++] = board_distance;
    
     // ----- upright ----- //
    double torso_height = SensorByName(model, data, "torso_position")[2];
     // is standing
    double standing =
      torso_height / mju_sqrt(torso_height * torso_height + 0.45 * 0.45) - 0.4;
    double* torso_up = SensorByName(model, data, "torso_up");
    double* pelvis_up = SensorByName(model, data, "pelvis_up");
    double* foot_right_up = SensorByName(model, data, "foot_right_up");
    double* foot_left_up = SensorByName(model, data, "foot_left_up");
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

    // ----- COM xy velocity should be compared to linear_velocity_global ----- //
    double *com_difference = new double[2];
    com_difference[0] = (linear_velocity_global[0] - com_velocity[0] * standing);
    com_difference[1] = (linear_velocity_global[1] - com_velocity[1] * standing);
    
    mju_copy(&residual[counter], com_difference, 2);
    counter += 2;


    // ----- Skateboard: Feet should be on the skateboard ----- //

    double *back_plate_pos = SensorByName(model, data, "track-back-plate");
    double *front_plate_pos = SensorByName(model, data, "track-front-plate");

    double *left_foot_pos = SensorByName(model, data, "tracking_foot_left");
    double *right_foot_pos = SensorByName(model, data, "tracking_foot_right");
 
    double right_feet_slider = parameters_[ParameterIndex(model, "rFeet pos")];
    double left_feet_slider = parameters_[ParameterIndex(model, "LFeet pos")];
    
    // calculate x-wise difference between the plates, based on right_feet_slider
    double plate_distance_x = mju_abs(back_plate_pos[0] - front_plate_pos[0]);
    double plate_distance_y = mju_abs(back_plate_pos[1] - front_plate_pos[1]);
    // calculate the x position of the line set by the plates
    double right_feet_x = front_plate_pos[0] - right_feet_slider * plate_distance_x;
    double right_feet_y = front_plate_pos[1] - right_feet_slider * plate_distance_y;


    // calculate y-wise difference between the plates, based on left_feet_slider
    plate_distance_y = mju_abs(back_plate_pos[0] + 0.2);
    double left_feet_y = back_plate_pos[0] - left_feet_slider * plate_distance_y;
    double y_weight = (left_feet_slider == 0.0);
    // left foot should be on a parallel line from the line set by the plates, off-set by 2.0
    
    double left_foot_x = left_foot_pos[0];
    double left_foot_y = left_foot_pos[1];
    double back_plate_x = back_plate_pos[0];
    double back_plate_y = back_plate_pos[1];
    double front_plate_x = front_plate_pos[0];
    double front_plate_y = front_plate_pos[1];
     // Calculate the line equation
    double A = front_plate_y - back_plate_y;
    double B = back_plate_x - front_plate_x;
    double C = (front_plate_x * back_plate_y) - (back_plate_x * front_plate_y);

    // Calculate the distance from the point to the line
    double d = std::abs((A * left_foot_x + B * left_foot_y + C) / std::sqrt(A*A + B*B));

    double left_foot_distance_to_back_plate = d;
    // left_foot_distance_to_back_plate should be 0.2
    double left_foot_pushing_error = mju_abs(left_foot_distance_to_back_plate - 0.3);
    
    // left feet error, distance to back plate position 
    double distance_x = mju_abs(left_foot_pos[0] - left_feet_y);
    double distance_y = mju_abs(left_foot_pos[1] - back_plate_pos[1]);
    double distance_z = mju_abs(left_foot_pos[2] - back_plate_pos[2]);
    double left_feet_error = standing*mju_sqrt(distance_x*distance_x + distance_y*distance_y + distance_z*distance_z);
    if (y_weight == 0.0) {
      left_feet_error = left_foot_pushing_error;
    }

    // right feet error, distance to front plate position
    distance_x = mju_abs(right_foot_pos[0] - right_feet_x);
    distance_y = mju_abs(right_foot_pos[1] - right_feet_y);
    distance_z = mju_abs(right_foot_pos[2] - front_plate_pos[2]);
    double right_feet_error = standing*mju_sqrt(distance_x*distance_x + distance_y*distance_y + distance_z*distance_z);
    
    residual[counter++] = right_feet_error;
    residual[counter++] = left_feet_error;

    // board tilt angle
    double tilt_angle_goal = parameters_[ParameterIndex(model, "Tilt angle")];
    // skateboard_xmat[8] is the tilt angle, get absolute error , [0 , 1]
    double uprightness = (1.0 - skateboard_xmat[8]) * 10;
    double tilt_angle_error = mju_abs(tilt_angle_goal - uprightness);
   
    residual[counter++] = tilt_angle_error;
    
    
     // foot orientation
    int foot_body_id_ = mj_name2id(model, mjOBJ_XBODY, "foot_right");
    if (foot_body_id_ < 0) mju_error("body 'pelvis' not found");
    // `foot_xmat` is the flattened 9-dimensional global rotation matrix
    // describing the orientation of the board.
    double foot_parameter = parameters_[ParameterIndex(model, "rFeet orientation")];
    double* foot_xmat = data->xmat + 9*foot_body_id_;
    // compare pelvis orientation to skateboard orientation, on 360 degrees scale
    double foot_orientation = atan2(foot_xmat[3], foot_xmat[0]) + M_PI;
    double foot_orientation_goal = atan2(skateboard_xmat[3], skateboard_xmat[0]) + M_PI;
    double desired_orientation = fmod(foot_orientation_goal + foot_parameter, 2 * M_PI);
    double angle_difference = atan2(sin(foot_orientation - desired_orientation), cos(foot_orientation - desired_orientation));
    double foot_error = fabs(angle_difference) / M_PI;

    // pelvis orientation
    int torso_body_id_ = mj_name2id(model, mjOBJ_XBODY, "torso");
    if (torso_body_id_ < 0) mju_error("body 'pelvis' not found");
    // `torso_xmat` is the flattened 9-dimensional global rotation matrix
    // describing the orientation of the board.
    double torso_parameter = parameters_[ParameterIndex(model, "rFeet orientation")];
    double* torso_xmat = data->xmat + 9*torso_body_id_;
    // compare pelvis orientation to skateboard orientation, on 360 degrees scale
    double torso_orientation = atan2(torso_xmat[3], torso_xmat[0]) + M_PI;
    double torso_orientation_goal = atan2(skateboard_xmat[3], skateboard_xmat[0]) + M_PI;
    double desired_orientation_p = fmod(torso_orientation_goal + torso_parameter, 2 * M_PI);
    double angle_difference_p = atan2(sin(torso_orientation - desired_orientation_p), cos(torso_orientation - desired_orientation_p));
    double torso_error = fabs(angle_difference_p) / M_PI;


    residual[counter++] = (foot_error + torso_error) * standing;

    if (jiiri++ % 30000 == 0) {
      // print positions
      printf("error_left=%f\n", left_feet_error);
      printf("error_right=%f\n\n", right_feet_error);
      printf("board_distance=%f\n", board_distance);
      // print qvel as 'qvel="0 0 0 1 0"' etc
      // printf("qvel=\"");
      // for (int i = 0; i < model->nv; i++) {
      //   printf("%f ", data->qvel[i]);
      // }
      // printf("\"\n");
      // printf("board_distance=%f\n", board_distance);
      // printf("skateboard_heading=%f\n", skateboard_heading);
      // printf("pelvis_orientation_error=%f\n", torso_error);
      // printf("torso_orientation_goal=%f\n", torso_orientation_goal);
      // printf("torso_orientation=%f\n\n", torso_orientation);
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

    int humanoid_joint_nq = mjpc::CountJointNqUnderBody(model, "torso");
    int humanoid_joint_nv = mjpc::CountJointNvUnderBody(model, "torso");
    // 28 = 3 freejoint translation
    //   + 4 freejoint quaternion
    //   + 21 pendulum hinges
    assert(humanoid_joint_nq == 28);
    // 27 = 3 + 3 = freejoint-translational + freejoint-angular + joints
    assert(humanoid_joint_nv == 27);

    assert(model->nq == skateboard_joint_nq + humanoid_joint_nq);
    assert(model->nv == skateboard_joint_nv + humanoid_joint_nv);

    // ----- joint velocity ----- //
    mju_copy(residual + counter, data->qvel + 6, humanoid_joint_nv - 6);
    counter += humanoid_joint_nv - 6;
    // printf counter after joint velocity

    mju_copy(residual + counter, data->qvel + humanoid_joint_nv + 6 + 3, 1);
    counter += 1;
    // ----- action ----- //
    mju_copy(&residual[counter], data->ctrl, model->nu);
    counter += model->nu;

    // sensor dim sanity check
    CheckSensorDim(model, counter);

  }

} // namespace mjpc::humanoid

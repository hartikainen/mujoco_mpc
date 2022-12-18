#ifndef MJPC_TASKS_HUMANOID_CMU_STAND_TASK_H_
#define MJPC_TASKS_HUMANOID_CMU_STAND_TASK_H_

#include <mujoco/mujoco.h>

namespace mjpc {
namespace HumanoidCMU {

struct Stand {

  // -------------- Residuals for HumanoidCMU stand task ------------
  //   Number of residuals: 6
  //     Residual (0): control
  //     Residual (1): COM_xy - average(feet position)_xy
  //     Residual (2): torso_xy - COM_xy
  //     Residual (3): head_z - feet^{(i)}_position_z - height_goal
  //     Residual (4): velocity COM_xy
  //     Residual (5): joint velocity
  //   Number of parameters: 1
  //     Parameter (0): height_goal
  // ----------------------------------------------------------------
  static void Residual(const double* parameters, const mjModel* model,
                       const mjData* data, double* residual);

};

} // namespace HumanoidCMU
}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_CMU_STAND_TASK_H_

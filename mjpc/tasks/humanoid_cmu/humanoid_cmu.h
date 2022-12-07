#ifndef MJPC_TASKS_HUMANOID_HUMANOID_CMU_H_
#define MJPC_TASKS_HUMANOID_HUMANOID_CMU_H_

#include <mujoco/mujoco.h>

namespace mjpc {
struct HumanoidCMU {
  // ------------------ Residuals for humanoid stand task ------------
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
  static void ResidualStand(const double* parameters, const mjModel* model,
                            const mjData* data, double* residual);
};
}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_HUMANOID_CMU_H_

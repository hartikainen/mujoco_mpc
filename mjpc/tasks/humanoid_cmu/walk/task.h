#ifndef MJPC_TASKS_HUMANOID_CMU_WALK_TASK_H_
#define MJPC_TASKS_HUMANOID_CMU_WALK_TASK_H_

#include <mujoco/mujoco.h>

namespace mjpc {
namespace HumanoidCMU {

struct Walk {

  // --------------- Residuals for HumanoidCMU walk task ------------
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
  static void Residual(const double* parameters, const mjModel* model,
                       const mjData* data, double* residual);

};

} // namespace HumanoidCMU
}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_CMU_WALK_TASK_H_

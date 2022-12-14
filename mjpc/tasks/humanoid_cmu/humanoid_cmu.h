#ifndef MJPC_TASKS_HUMANOID_HUMANOID_CMU_H_
#define MJPC_TASKS_HUMANOID_HUMANOID_CMU_H_

#include <mujoco/mujoco.h>

namespace mjpc {
struct HumanoidCMU {
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
  static void ResidualStand(const double* parameters, const mjModel* model,
                            const mjData* data, double* residual);

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
  static void ResidualWalk(const double* parameters, const mjModel* model,
                            const mjData* data, double* residual);

  // ----------- Residuals for HumanoidCMU tracking task ------------
  //   Number of residuals: TODO(hartikainen)
  //     Residual (0): TODO(hartikainen)
  //   Number of parameters: TODO(hartikainen)
  //     Parameter (0): TODO(hartikainen)
  // ----------------------------------------------------------------
  static void ResidualTrackSequence(const double* parameters, const mjModel* model,
                                    const mjData* data, double* residual);

  // -------- Transition for HumanoidCMU tracking task ---------
  //   TODO(hartikainen)
  // -----------------------------------------------------------
  static int TransitionTrackSequence(int state, const mjModel* model, mjData* data);

};
}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_HUMANOID_CMU_H_

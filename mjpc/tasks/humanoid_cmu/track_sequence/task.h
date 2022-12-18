#ifndef MJPC_TASKS_HUMANOID_CMU_TRACK_SEQUENCE_TASK_H_
#define MJPC_TASKS_HUMANOID_CMU_TRACK_SEQUENCE_TASK_H_

#include <mujoco/mujoco.h>

namespace mjpc {
namespace HumanoidCMU {

struct TrackSequence {

  // ----------- Residuals for HumanoidCMU tracking task ------------
  //   Number of residuals: TODO(hartikainen)
  //     Residual (0): TODO(hartikainen)
  //   Number of parameters: TODO(hartikainen)
  //     Parameter (0): TODO(hartikainen)
  // ----------------------------------------------------------------
  static void Residual(const double* parameters, const mjModel* model,
                       const mjData* data, double* residual);

  // -------- Transition for HumanoidCMU tracking task ---------
  //   TODO(hartikainen)
  // -----------------------------------------------------------
  static int Transition(int state, const mjModel* model, mjData* data);

};

} // namespace HumanoidCMU
}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_CMU_TRACK_SEQUENCE_TASK_H_

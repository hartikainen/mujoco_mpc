#ifndef MJPC_TASKS_WELDEDHUMANOID_SKATEBOARD_TASK_H_
#define MJPC_TASKS_WELDEDHUMANOID_SKATEBOARD_TASK_H_

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace weldedhumanoid {

class Skateboard : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Skateboard* task, int current_mode = 0,
                        double reference_time = 0)
        : mjpc::BaseResidualFn(task),
          current_mode_(current_mode),
          reference_time_(reference_time) {}

    // ------------------ Residuals for weldedhumanoid skateboard task ------------
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
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
   private:
    friend class Skateboard;
    int current_mode_;
    double reference_time_;
  };

  Skateboard() : residual_(this) {}

  std::string Name() const override;
  std::string XmlPath() const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.current_mode_,
                                        residual_.reference_time_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  // int current_mode_;
  // double reference_time_;
  ResidualFn residual_;
};

}  // namespace weldedhumanoid
}  // namespace mjpc

#endif  // MJPC_TASKS_WELDEDHUMANOID_SKATEBOARD_TASK_H_

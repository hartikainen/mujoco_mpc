#ifndef MJPC_TASKS_SKATEBOARD_CART_POLE_TASK_H_
#define MJPC_TASKS_SKATEBOARD_CART_POLE_TASK_H_

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace skateboard {

class CartPole : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const CartPole* task, int current_mode = 0,
                        double reference_time = 0)
        : mjpc::BaseResidualFn(task),
          current_mode_(current_mode),
          reference_time_(reference_time) {}

    // ------------------ Residuals for skateboard cart_pole task ------------
    //   Number of residuals: 6
    //     Residual (0): TODO(hartikainen)
    //   Number of parameters: 1
    //     Parameter (0): TODO(hartikainen)
    // ----------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
   private:
    friend class CartPole;
    int current_mode_;
    double reference_time_;
  };

  CartPole() : residual_(this) {}

  std::string Name() const override;
  std::string XmlPath() const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.current_mode_,
                                        residual_.reference_time_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace skateboard
}  // namespace mjpc

#endif  // MJPC_TASKS_SKATEBOARD_CART_POLE_TASK_H_

#ifndef MJPC_TASKS_SKATEBOARD_CART_POLE_TASK_H_
#define MJPC_TASKS_SKATEBOARD_CART_POLE_TASK_H_

#include <memory>
#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace skateboard {

class CartPole : public ThreadSafeTask {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const CartPole* task);

    // ------------------ Residuals for skateboard cart_pole task ------------
    //   Number of residuals: 6
    //     Residual (0): TODO(hartikainen)
    //   Number of parameters: 1
    //     Parameter (0): TODO(hartikainen)
    // ----------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  CartPole() : residual_(this) {}

  std::string Name() const override;
  std::string XmlPath() const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace skateboard
}  // namespace mjpc

#endif  // MJPC_TASKS_SKATEBOARD_CART_POLE_TASK_H_

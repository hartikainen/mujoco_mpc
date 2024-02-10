// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MJPC_TASKS_HUMANOID_SKATEBOARD_STEERING_TASK_H_
#define MJPC_TASKS_HUMANOID_SKATEBOARD_STEERING_TASK_H_

#include <mujoco/mujoco.h>

#include "mjpc/task.h"

namespace mjpc {
namespace humanoid {

class Steering : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Steering* task) : mjpc::BaseResidualFn(task) {}
    ResidualFn(const ResidualFn&) = default;

    // ------- Residuals for humanoid skateboard steering task --------
    //   Number of residuals:
    //     Residual (0): Joint vel: minimise joint velocity
    //     Residual (1): Control: minimise control
    //     Residual (2-11): Steering position: minimise steering position
    //     error
    //         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
    //     Residual (11-20): Steering velocity: minimise steering velocity
    //     error
    //         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
    //   Number of parameters: 0
    // ----------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

   private:
    friend class Steering;
    int current_mode_ = 0;
    double reference_time_ = 0;

    //  ============  states updated in Transition()  ============
    // std::array<mjtNum, 3> skateboard_position_ = {0, 0, 0};
    // std::array<mjtNum, 2> skateboard_heading_ = {0, 0};
    // std::array<mjtNum, 2> goal_position_ = {0, 0};

    //  ============  constants, computed in Reset()  ============
    int goal_body_id_ = -1;
    int goal_body_mocap_id_ = -1;
    int goal_geom_id_ = -1;
    int skateboard_body_id_ = -1;
    int skateboard_xbody_id_ = -1;

    //  ===================  helper functions  ===================
    std::vector<double> ComputeTrackingResidual(const mjModel* model,
                                                const mjData* data) const;
    std::array<double, 6> ComputeFootPositionsResidual(
        const mjModel* model, const mjData* data) const;
    std::array<double, 2> ComputeBoardHeadingResidual(const mjModel* model,
                                                      const mjData* data) const;
    std::array<double, 3> ComputeBoardVelocityResidual(
        const mjModel* model, const mjData* data) const;
    std::array<double, 3> ComputeHumanoidVelocityResidual(
        const mjModel* model, const mjData* data) const;
  };

  Steering() : residual_(this) {}

  // --------------------- Transition for humanoid task
  // ------------------------
  //   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
  //   Linearly interpolate between two consecutive key frames in order to
  //   smooth the transitions between keyframes.
  // ---------------------------------------------------------------------------
  void TransitionLocked(mjModel* model, mjData* data) override;

  // call base-class Reset, save task-related ids
  void ResetLocked(const mjModel* model) override;

  // draw task-related geometry in the scene
  void ModifyScene(const mjModel* model, const mjData* data,
                   mjvScene* scene) const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(residual_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  friend class ResidualFn;
  ResidualFn residual_;
};

}  // namespace humanoid
}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_SKATEBOARD_STEERING_TASK_H_

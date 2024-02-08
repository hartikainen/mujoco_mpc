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
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Steering* task, int current_mode = 0,
                        double reference_time = 0)
        : mjpc::BaseResidualFn(task),
          current_mode_(current_mode),
          reference_time_(reference_time) {}

    // ------- Residuals for humanoid skateboard steering task --------
    //   Number of residuals:
    //     Residual (0): Joint vel: minimise joint velocity
    //     Residual (1): Control: minimise control
    //     Residual (2-11): Steering position: minimise steering position error
    //         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
    //     Residual (11-20): Steering velocity: minimise steering velocity error
    //         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
    //   Number of parameters: 0
    // ----------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

   private:
    friend class Steering;
    int current_mode_;
    double reference_time_;

    //  ============  states updated in Transition()  ============
    // std::array<mjtNum, 3> skateboard_position_ = {0, 0, 0};
    // std::array<mjtNum, 2> skateboard_heading_ = {0, 0};
    // std::array<mjtNum, 2> goal_position_ = {0, 0};

    //  ============  constants, computed in Reset()  ============
    int goal_body_id_ = -1;
    int goal_body_mocap_id_ = -1;
    int goal_geom_id_ = -1;
    int skateboard_body_id_ = -1;

    //  ===================  helper functions  ===================
    std::vector<double> ComputeTrackingResidual(
        const mjModel* model, const mjData* data, const int current_mode_,
        const double reference_time_, std::vector<double> parameters) const;
    std::array<double, 2> ComputeFootPositionsResidual(
        const mjModel* model, const mjData* data,
        std::vector<double> parameters) const;
    std::array<double, 3> ComputeGoalPositionResidual(
        const mjModel* model, const mjData* data,
        std::vector<double> parameters) const;
    std::array<double, 1> ComputeGoalOrientationResidual(
        const mjModel* model, const mjData* data,
        std::vector<double> parameters) const;
  };

  Steering() : residual_(this) {}

  // --------------------- Transition for humanoid task ------------------------
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

}  // namespace humanoid
}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_SKATEBOARD_STEERING_TASK_H_

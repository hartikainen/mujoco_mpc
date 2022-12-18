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

#include "tasks/tasks.h"

#include "tasks/acrobot/acrobot.h"
#include "tasks/cartpole/cartpole.h"
#include "tasks/hand/hand.h"
#include "tasks/humanoid/humanoid.h"
#include "tasks/humanoid_cmu/humanoid_cmu.h"
#include "tasks/particle/particle.h"
#include "tasks/quadrotor/quadrotor.h"
#include "tasks/quadruped/quadruped.h"
#include "tasks/swimmer/swimmer.h"
#include "tasks/walker/walker.h"

namespace mjpc {

namespace {
// Define the array without an explicit size then bind it to an explicitly
// sized reference afterward. This way the compiler enforces equality between
// kNumTasks and the size of the array initializer.
const TaskDefinition kTasksArray[]{
    {
        .name = "HumanoidCMU Tracking",
        .xml_path = "humanoid_cmu/track_sequence/task.xml",
        .residual = &HumanoidCMU::TrackSequence::Residual,
        .transition = &HumanoidCMU::TrackSequence::Transition,
    },
    {
        .name = "Humanoid Tracking",
        .xml_path = "humanoid/task_track_sequence.xml",
        .residual = &Humanoid::ResidualTrackSequence,
        .transition = &Humanoid::TransitionTrackSequence,
    },
    {
        .name = "HumanoidCMU Walk",
        .xml_path = "humanoid_cmu/walk/task.xml",
        .residual = &HumanoidCMU::Walk::Residual,
    },
    {
        .name = "HumanoidCMU Stand",
        .xml_path = "humanoid_cmu/stand/task.xml",
        .residual = &HumanoidCMU::Stand::Residual,
    },
    {
        .name = "Humanoid Walk",
        .xml_path = "humanoid/task_walk.xml",
        .residual = &Humanoid::ResidualWalk,
    },
    {
        .name = "Humanoid Stand",
        .xml_path = "humanoid/task_stand.xml",
        .residual = &Humanoid::ResidualStand,
    },
    {
        .name = "Swimmer",
        .xml_path = "swimmer/task.xml",
        .residual = &Swimmer::Residual,
        .transition = &Swimmer::Transition,
    },
    {
        .name = "Walker",
        .xml_path = "walker/task.xml",
        .residual = &Walker::Residual,
    },
    {
        .name = "Cart-pole",
        .xml_path = "cartpole/task.xml",
        .residual = &Cartpole::Residual,
    },
    {
        .name = "Acrobot",
        .xml_path = "acrobot/task.xml",
        .residual = &Acrobot::Residual,
    },
    {
        .name = "Particle",
        .xml_path = "particle/task.xml",
        .residual = &Particle::Residual,
    },
    {
        .name = "Quadruped Hill",
        .xml_path = "quadruped/task_hill.xml",
        .residual = &Quadruped::Residual,
        .transition = &Quadruped::Transition,
    },
    {
        .name = "Quadruped Flat",
        .xml_path = "quadruped/task_flat.xml",
        .residual = &Quadruped::ResidualFloor,
    },
    {
        .name = "Hand",
        .xml_path = "hand/task.xml",
        .residual = &Hand::Residual,
        .transition = &Hand::Transition,
    },
    {
        .name = "Quadrotor",
        .xml_path = "quadrotor/task.xml",
        .residual = &Quadrotor::Residual,
        .transition = &Quadrotor::Transition,
    },
};
}  // namespace

const TaskDefinition (&kTasks)[kNumTasks] = kTasksArray;

}  // namespace mjpc

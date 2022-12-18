#include "tasks/humanoid_cmu/humanoid_cmu.h"
#include "mujoco/mjmodel.h"

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>

#include <mujoco/mujoco.h>
#include "utilities.h"

namespace mjpc {

// ------------- Residuals for HumanoidCMU tracking task -------------
//   Number of residuals: TODO(hartikainen)
//     Residual (0): TODO(hartikainen)
//   Number of parameters: TODO(hartikainen)
//     Parameter (0): TODO(hartikainen)
// -------------------------------------------------------------------
void HumanoidCMU::ResidualTrackSequence(const double* parameters, const mjModel* model,
                                        const mjData* data, double* residual) {
  int counter = 0;

  // float fps = 30.0;
  // int step_index = std::min((int) (data->time * fps), (model->nkey) - 1);

  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // ----- joint velocity ----- //
  mju_copy(residual + counter, data->qvel, model->nv);
  counter += model->nv;

  std::array<std::string, 16> body_names = {
    "pelvis", "head", "ltoe", "rtoe", "lheel", "rheel", "lknee", "rknee",
    "lhand", "rhand", "lelbow", "relbow", "lshoulder", "rshoulder", "lhip",
    "rhip",
  };

  for (const auto& body_name : body_names) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    std::string pos_sensor_name = "tracking_pos[" + body_name + "]";
    int body_mocapid = model->body_mocapid[mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str())];
    assert(0 <= body_mocapid);

    double mocap_body_pos[3];
    mju_copy3(mocap_body_pos, data->mocap_pos + 3 * body_mocapid);
    // mju_copy3(mocap_body_pos, model->key_mpos + model->nmocap * 3 * step_index + 3 * body_mocapid);

    double* sensor_pos = mjpc::SensorByName(model, data, pos_sensor_name.c_str());

    mju_sub3(&residual[counter],
             mocap_body_pos,
             sensor_pos);
    counter += 3;
  }

  for (const auto& body_name : body_names) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    std::string linvel_sensor_name = "tracking_linvel[" + body_name + "]";
    int body_mocapid = model->body_mocapid[
      mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str())];
    assert(0 <= body_mocapid);

    double mocap_body_linvel[3];
    mju_copy3(mocap_body_linvel, data->mocap_quat + 3 * body_mocapid);
    // double current_mocap_body_pos[3];
    // mju_copy3(current_mocap_body_pos, model->key_mpos + model->nmocap * 3 * (step_index + 0) + 3 * body_mocapid);
    // double next_mocap_body_pos[3];
    // mju_copy3(next_mocap_body_pos, model->key_mpos + model->nmocap * 3 * (step_index + 1) + 3 * body_mocapid);
    // mju_sub3(mocap_body_linvel, next_mocap_body_pos, current_mocap_body_pos);
    // mju_scl3(mocap_body_linvel, mocap_body_linvel, fps);

    double* sensor_linvel = mjpc::SensorByName(model, data, linvel_sensor_name.c_str());

    mju_sub3(&residual[counter],
             mocap_body_linvel,
             sensor_linvel);
    counter += 3;
  }

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i=0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    std::printf("user_sensor_dim=%d, counter=%d", user_sensor_dim, counter);
    mju_error_i("mismatch between total user-sensor dimension"
                "and actual length of residual %d", user_sensor_dim);
  }

}

// -------- Transition for HumanoidCMU tracking task ---------
//   TODO(hartikainen)
// -----------------------------------------------------------
int HumanoidCMU::TransitionTrackSequence(int state, const mjModel* model, mjData* data) {
  // TODO(hartikainen): Add distance-based target transition logic.
  // TODO(hartikainen): is `data->time` the right thing to index here?
  float fps = 30.0;
  // int step_index = std::min((int) (data->time * fps), (model->nkey) - 1);
  int step_index = 0;
  mju_copy(data->mocap_pos,
           model->key_mpos + model->nmocap * 3 * (step_index + 0),
           model->nmocap * 3);

  // velocities
  mju_copy(data->mocap_quat,
           model->key_mpos + model->nmocap * 3 * (step_index + 1),
           model->nmocap * 3);
  mju_subFrom(data->mocap_quat, data->mocap_pos, model->nmocap * 3);
  mju_scl(data->mocap_quat, data->mocap_quat, fps, model->nmocap * 3);

  int new_state = step_index;

  return new_state;
}

}  // namespace mjpc

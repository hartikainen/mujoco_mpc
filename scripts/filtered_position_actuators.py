"""Filtered position actuators for SimpleHumanoid."""

import collections
import pathlib
import textwrap
from typing import Any, Callable, Literal, Optional

from absl import app
from absl import logging
import dm_control
from dm_control import mjcf
from dm_control.locomotion.walkers import scaled_actuators
import numpy as np
import numpy.typing as npt
import tree


Path = pathlib.Path

logging.set_verbosity(logging.INFO)


PositionActuatorParams = collections.namedtuple(
    "PositionActuatorParams", ["name", "forcerange", "kp", "damping"]
)

_POSITION_ACTUATORS = [
    PositionActuatorParams("abdomen_y", [-300, 300], 300, None),
    PositionActuatorParams("abdomen_z", [-180, 180], 180, None),

    PositionActuatorParams("abdomen_x", [-200, 200], 200, None),

    PositionActuatorParams("hip_x_right", [-200, 200], 200, None),
    PositionActuatorParams("hip_z_right", [-200, 200], 200, None),
    PositionActuatorParams("hip_y_right", [-300, 300], 300, None),
    PositionActuatorParams("knee_right", [-160, 160], 160, None),

    PositionActuatorParams("ankle_x_right", [-50, 50], 50, None),
    PositionActuatorParams("ankle_y_right", [-120, 120], 120, None),

    PositionActuatorParams("hip_x_left", [-200, 200], 200, None),
    PositionActuatorParams("hip_z_left", [-200, 200], 200, None),
    PositionActuatorParams("hip_y_left", [-300, 300], 300, None),
    PositionActuatorParams("knee_left", [-160, 160], 160, None),

    PositionActuatorParams("ankle_x_left", [-50, 50], 50, None),
    PositionActuatorParams("ankle_y_left", [-120, 120], 120, None),

    PositionActuatorParams("shoulder1_right", [-120, 120], 120, None),
    PositionActuatorParams("shoulder2_right", [-120, 120], 120, None),
    PositionActuatorParams("elbow_right", [-90, 90], 90, None),

    PositionActuatorParams("shoulder1_left", [-120, 120], 120, None),
    PositionActuatorParams("shoulder2_left", [-120, 120], 120, None),
    PositionActuatorParams("elbow_left", [-90, 90], 90, None),
]


def main(argv):
    mujoco_mpc_root_path = Path(__file__).resolve().parents[1]
    humanoid_xml_path = (
        mujoco_mpc_root_path
        / "mjpc"
        / "tasks"
        / "humanoid"
        / "humanoid.xml"
    )
    mjcf_model = mjcf.from_path(humanoid_xml_path.as_posix())

    mjcf_model.default.general.forcelimited = "true"
    mjcf_model.actuator.motor.clear()

    for actuator_params in _POSITION_ACTUATORS:
        associated_joint = mjcf_model.find("joint", actuator_params.name)
        if hasattr(actuator_params, "damping"):
            associated_joint.damping = actuator_params.damping

        if associated_joint.range is not None:
            associated_joint_range = associated_joint.range
        elif associated_joint.dclass.joint.range is not None:
            associated_joint_range = associated_joint.dclass.joint.range
        elif associated_joint.dclass.parent.joint.range is not None:
            associated_joint_range = associated_joint.dclass.parent.joint.range
        else:
            raise ValueError(
                f"No matching joint range found for joint {associated_joint.name}."
            )

        _ = scaled_actuators.add_position_actuator(
            name=actuator_params.name,
            target=associated_joint,
            kp=actuator_params.kp,
            qposrange=associated_joint_range,
            ctrlrange=(-1, 1),
            forcerange=actuator_params.forcerange,
            dyntype="filter",
            dynprm=[0.03],
        )


    print(textwrap.indent(mjcf_model.actuator.to_xml_string(precision=5, pretty_print=True).replace(' class="/"', ''), '  '))



if __name__ == "__main__":
    app.run(main)

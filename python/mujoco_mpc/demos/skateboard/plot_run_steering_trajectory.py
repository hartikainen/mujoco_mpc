"""Plot trajectory obtained from `run_steering.py`.

Given the result from `run_steering.py` (`mujoco_states.npz`), this function plots the
2D trajectory of the skateboard.
"""

import pathlib

from absl import app
from absl import flags
import numpy as np
import tree
import visualization as visualization_lib


Path = pathlib.Path

_TRIAL_PATH_FLAG = flags.DEFINE_string(
    "trial_path",
    None,
    "Path to the trial data output from `run_steering.py`.",
)

flags.mark_flags_as_required((_TRIAL_PATH_FLAG,))


def main(argv):
    del argv

    trial_path = Path(_TRIAL_PATH_FLAG.value).expanduser()
    print(f"{trial_path=}")

    if not trial_path.exists():
        raise FileNotFoundError(f"{trial_path} does not exist.")

    with (trial_path / "mujoco_states.npz").open("rb") as f:
        data = dict(np.load(f))

    print("data:", tree.map_structure(np.shape, data))

    visualization_lib.plot_trajectory_to_path(
        data["qpos"], trial_path / "trajectory.png"
    )


if __name__ == "__main__":
    app.run(main)

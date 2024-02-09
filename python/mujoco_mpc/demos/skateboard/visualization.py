"""Plot trajectory obtained from `run_steering.py`.

Given the result from `run_steering.py` (`mujoco_states.npz`), this function plots the
2D trajectory of the skateboard.
"""

import pathlib

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

Path = pathlib.Path


def plot_trajectory_to_axis(qpos: npt.ArrayLike, figure: plt.Figure, axis: plt.Axes):
    """Plot the trajectory of the skateboard.

    Args:
        qpos: The joint positions of the skateboard.
        figure: The figure to plot on.
        axis: The axis to plot on.
    """
    del figure
    qpos = np.array(qpos)
    axis.plot(qpos[:, 0], qpos[:, 1], label="Trajectory")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Trajectory of the skateboard")
    axis.legend()


def plot_trajectory_to_path(qpos: npt.ArrayLike, path: Path):
    """Plot the trajectory of the skateboard.

    Args:
        qpos: The joint positions of the skateboard.
        figure: The figure to plot on.
        axis: The axis to plot on.
    """
    figure, axis = plt.subplots(1, 1, figsize=(8, 8))
    plot_trajectory_to_axis(qpos, figure, axis)
    plt.savefig(path)

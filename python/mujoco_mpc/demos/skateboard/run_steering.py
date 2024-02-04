"""Synchronous headless runner for "Humanoid Skateboard Steer" task.

Usage:
```sh
python ./python/mujoco_mpc/demos/skateboard/run_steering.py \
  --time_limit=5.0 \
  --num_planner_steps=10 \
  --cost_weights='{}' \
  --task_parameters='{}' \
  --output_path="/tmp/run_steering" \
  --verbosity=1
```
"""

import dataclasses
import datetime
import json
import pathlib
import subprocess
from typing import Any, Callable, Literal, Optional

from absl import app
from absl import flags
from absl import logging
import mediapy
import mujoco
import numpy as np

# NOTE(hartikainen): We need to import testing here because there's some weird
# deadlock happening when `numpy.tesing` is used after spinning up the gRPC
# server.
from numpy import testing
import numpy.typing as npt

from mujoco_mpc import agent as agent_lib

dataclass = dataclasses.dataclass
Path = pathlib.Path

_TIME_LIMIT_FLAG = flags.DEFINE_float(
    "time_limit",
    5.0,
    "Time limit in seconds.",
)
_COST_WEIGHTS_FLAG = flags.DEFINE_string(
    "cost_weights",
    None,
    "JSON-serializable values for cost weights.",
)
_TASK_PARAMETERS_FLAG = flags.DEFINE_string(
    "task_parameters",
    None,
    "JSON-serializable values for task_parameters.",
)
_NUM_PLANNER_STEPS_FLAG = flags.DEFINE_integer(
    "num_planner_steps",
    1,
    "Number of planner steps to take per environment step.",
)
_PLANNER_STEP_TOLERANCE_FLAG = flags.DEFINE_float(
    "planner_step_tolerance",
    1e-2,
    "Tolerance for early-stopping the planner.",
)
_OUTPUT_PATH_FLAG = flags.DEFINE_string(
    "output_path",
    None,
    "Path to save output.",
)


@dataclass
class TimeStep:
    """Time step information."""

    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    cost_total: float
    cost_terms: dict[str, float]
    frame: np.ndarray


def load_cost_weights(cost_weights_str: Optional[str]) -> dict[str, float]:
    if cost_weights_str is None:
        return {}
    try:
        cost_weights = json.loads(cost_weights_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Unable to parse `cost_weights`: {e}")

    return cost_weights


def set_cost_weights(agent: agent_lib.Agent, cost_weights: dict[str, float]):
    cost_weights_str = json.dumps(cost_weights, indent=2)
    logging.info(f"@set_cost_weights. cost_weights: {cost_weights_str}")
    agent.set_cost_weights(cost_weights)
    agent_cost_weights_str = json.dumps(agent.get_cost_weights(), indent=2)
    logging.info(
        f"@set_cost_weights. agent.get_cost_weights(): {agent_cost_weights_str}"
    )


def load_task_parameters(task_parameters_str: Optional[str]) -> dict[str, float]:
    if task_parameters_str is None:
        return {}
    try:
        task_parameters = json.loads(task_parameters_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Unable to parse `task_parameters`: {e}")

    return task_parameters


def set_task_parameters(agent: agent_lib.Agent, task_parameters: dict[str, float]):
    task_parameters_str = json.dumps(task_parameters, indent=2)
    logging.info(f"@set_task_parameters. task_parameters: {task_parameters_str}")
    agent.set_task_parameters(task_parameters)
    agent_task_parameters_str = json.dumps(agent.get_task_parameters(), indent=2)
    logging.info(
        f"@set_task_parameters. agent.get_task_parameters(): {agent_task_parameters_str}"
    )


def main(argv):
    time_limit = _TIME_LIMIT_FLAG.value
    cost_weights = load_cost_weights(_COST_WEIGHTS_FLAG.value)
    task_parameters = load_task_parameters(_TASK_PARAMETERS_FLAG.value)
    num_planner_steps = _NUM_PLANNER_STEPS_FLAG.value
    planner_step_tolerance = _PLANNER_STEP_TOLERANCE_FLAG.value
    if _OUTPUT_PATH_FLAG.present:
        output_path = Path(_OUTPUT_PATH_FLAG.value)
    else:
        output_path = None

    logging.info(f"{time_limit=}")
    logging.info(f"{cost_weights=}")
    logging.info(f"{task_parameters=}")

    model_path = (
        Path(__file__).parents[5]
        / "mujoco_mpc"
        / "mjpc"
        / "tasks"
        / "humanoid"
        / "skateboard"
        / "steering-task.xml"
    )

    render_width, render_height = 960, 960
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offheight = render_height
    model.vis.global_.offwidth = render_width
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=render_height, width=render_width)

    agent = agent_lib.Agent(
        task_id="Humanoid Skateboard Steer",
        model=model,
        # subprocess_kwargs={
        #     "stdout": subprocess.DEVNULL,
        #     "stderr": subprocess.DEVNULL,
        # },
    )

    set_cost_weights(agent, cost_weights)
    set_task_parameters(agent, task_parameters)

    def render_frame() -> np.ndarray:
        scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(scene_option)
        scene_option.sitegroup[3] = True

        renderer.update_scene(data, camera="back", scene_option=scene_option)
        back_frame = renderer.render()
        renderer.update_scene(data, camera="side", scene_option=scene_option)
        side_frame = renderer.render()

        frame = np.concatenate([back_frame, side_frame], axis=1)
        return frame

    def synchronize_state() -> None:
        """Synchronize the python state with the agent state."""
        agent_state = agent.get_state()
        state_spec = (
            mujoco.mjtState.mjSTATE_TIME
            | mujoco.mjtState.mjSTATE_QPOS
            | mujoco.mjtState.mjSTATE_QVEL
            | mujoco.mjtState.mjSTATE_ACT
            | mujoco.mjtState.mjSTATE_MOCAP_POS
            | mujoco.mjtState.mjSTATE_MOCAP_QUAT
            | mujoco.mjtState.mjSTATE_USERDATA
        )

        state = np.array(
            [
                agent_state.time,
                *agent_state.qpos,
                *agent_state.qvel,
                *agent_state.act,
                *agent_state.mocap_pos,
                *agent_state.mocap_quat,
                *agent_state.userdata,
            ]
        )
        mujoco.mj_setState(model, data, state, state_spec)
        mujoco.mj_forward(model, data)

    def get_time_step() -> TimeStep:
        return TimeStep(
            time=data.time,
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            ctrl=data.ctrl.copy(),
            cost_total=agent.get_total_cost(),
            cost_terms=agent.get_cost_term_values(),
            frame=render_frame(),
        )

    def agent_plan() -> np.ndarray:
        import time

        start = time.time()

        action0 = agent.get_action(data.time)
        for i in range(1, num_planner_steps + 1):
            agent.planner_step()
            action1 = agent.get_action(data.time, averaging_duration=None)
            action_diffs = np.abs(action0 - action1)
            action0 = action1
            if np.all(action_diffs < planner_step_tolerance):
                break

        end = time.time()
        logging.debug(
            f"@agent_plan: {i}steps / {end - start:.3f}s; sps: {i / (end - start):.3f}"
        )

        return action0

    def environment_reset() -> TimeStep:
        agent.reset()

        synchronize_state()

        return get_time_step()

    def environment_step() -> TimeStep:
        action = agent_plan()
        agent.step()

        synchronize_state()

        return action, get_time_step()

    time_steps = [environment_reset()]
    actions = []

    while (time := time_steps[-1].time) < time_limit:
        action, time_step = environment_step()
        actions.append(action)
        time_steps.append(time_step)

    agent.close()

    def dump_parameters(path: Path) -> None:
        parameters = {
            "time_limit": time_limit,
            "cost_weights": cost_weights,
            "task_parameters": task_parameters,
            "num_planner_steps": num_planner_steps,
            "planner_step_tolerance": planner_step_tolerance,
        }

        with path.open("wt") as f:
            json.dump(parameters, f, indent=2, sort_keys=True)

    def dump_video(path: Path) -> None:
        with mediapy.VideoWriter(
            path,
            shape=time_steps[0].frame.shape[0:2],
            fps=1.0 / model.opt.timestep,
        ) as video_writer:
            for time_step in time_steps:
                video_writer.add_image(time_step.frame)

    if output_path is not None:
        time_now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = output_path / time_now_str
        output_dir.mkdir(parents=True, exist_ok=True)

        video_save_path = output_dir / "video.mp4"
        parameters_save_path = output_dir / "parameters.json"

        dump_parameters(parameters_save_path)
        dump_video(video_save_path)


if __name__ == "__main__":
    app.run(main)

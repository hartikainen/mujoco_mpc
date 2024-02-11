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
_CUSTOM_DATA_FLAG = flags.DEFINE_string(
    "custom_data",
    None,
    "JSON-serializable values for custom data. Currently only supports setting setting numeric values.",
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
    qpos: npt.NDArray[np.float_]
    qvel: npt.NDArray[np.float_]
    ctrl: npt.NDArray[np.float_]
    cost_total: float
    cost_terms: dict[str, float]
    frame: npt.NDArray[np.float_]


def load_json(json_str: Optional[str], error_identifier: str) -> dict[str, float]:
    if not json_str:
        return {}
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Unable to parse '{error_identifier}': {e}")

    return data


def set_cost_weights(agent: agent_lib.Agent, cost_weights: dict[str, float]):
    cost_weights_str = json.dumps(cost_weights, indent=2)
    logging.info(f"@set_cost_weights. cost_weights: {cost_weights_str}")
    agent.set_cost_weights(cost_weights)
    agent_cost_weights_str = json.dumps(agent.get_cost_weights(), indent=2)
    logging.info(
        f"@set_cost_weights. agent.get_cost_weights(): {agent_cost_weights_str}"
    )


def set_task_parameters(agent: agent_lib.Agent, task_parameters: dict[str, float]):
    task_parameters_str = json.dumps(task_parameters, indent=2)
    logging.info(f"@set_task_parameters. task_parameters: {task_parameters_str}")
    agent.set_task_parameters(task_parameters)
    agent_task_parameters_str = json.dumps(agent.get_task_parameters(), indent=2)
    logging.info(
        f"@set_task_parameters. agent.get_task_parameters(): {agent_task_parameters_str}"
    )


def set_custom_text_data(model: mujoco.MjModel, name: str, value: str):
    raise NotImplementedError(
        "Can't set text values because the string buffer is read-only")

    def string_slice(string, start):
        return string[start:string.find(b'\x00', start)].decode('utf-8')

    for i in range(model.ntext):
        text_name = string_slice(model.names, model.name_textadr[i])

        if text_name == name:
            text_adr = model.text_adr[i]
            text_size = model.text_size[i]
            model_text_data = bytearray(model.text_data)
            model_text_data[text_adr:text_adr + text_size - 1] = value.encode('utf-8')
            model.text_data = bytes(model_text_data)

            return

    raise ValueError(f"Could not find `{name=}` in model.text_names")


def set_custom_numeric_data(model: mujoco.MjModel, name: str, value: float | int | npt.ArrayLike):
    value = np.array(value)
    # make sure that value is of dtype float or int
    is_floating = np.issubdtype(value.dtype, np.floating)
    is_integer = np.issubdtype(value.dtype, np.integer)
    if not (is_floating or is_integer):
        raise ValueError(f"Expected `{name=}` to be of dtype float or int, got {value.dtype=}, {value=}")

    def string_slice(string, start):
        return string[start:string.find(b'\x00', start)].decode('utf-8')

    for i in range(model.nnumeric):
        numeric_name = string_slice(model.names, model.name_numericadr[i])

        if numeric_name == name:
            numeric_adr = model.numeric_adr[i]
            numeric_size = model.numeric_size[i]
            model.numeric_data[numeric_adr:numeric_adr + numeric_size] = value

            return

    raise ValueError(f"Could not find `{name=}` in model.numeric_names")


def set_custom_data(model: mujoco.MjModel, custom_data: dict[str, Any]):
    for name, value in custom_data.items():
        if isinstance(value, str):
            set_custom_text_data(model, name, value)
        else:
            set_custom_numeric_data(model, name, value)

def main(argv):
    del argv

    time_limit = _TIME_LIMIT_FLAG.value
    cost_weights = load_json(_COST_WEIGHTS_FLAG.value, "cost_weights")
    task_parameters = load_json(_TASK_PARAMETERS_FLAG.value, "task_parameters")
    custom_data = load_json(_CUSTOM_DATA_FLAG.value, "custom_data")
    num_planner_steps = _NUM_PLANNER_STEPS_FLAG.value
    planner_step_tolerance = _PLANNER_STEP_TOLERANCE_FLAG.value
    if _OUTPUT_PATH_FLAG.present and _OUTPUT_PATH_FLAG.value is not None:
        output_path = Path(_OUTPUT_PATH_FLAG.value)
    else:
        output_path = None

    logging.info(f"{time_limit=}")
    logging.info(f"{cost_weights=}")
    logging.info(f"{task_parameters=}")
    logging.info(f"{custom_data=}")

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

    # NOTE(hartikainen): custom data, such as the agent or planner parameters
    # are set directly into the model because the grpc api only allows setting
    # "residual_"-values.
    set_custom_data(model, custom_data)

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

    def render_frame() -> npt.NDArray[np.float_]:
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

    def agent_plan() -> npt.NDArray[np.float_]:
        import time

        start = time.time()

        action0 = agent.get_action(data.time)
        for i in range(1, num_planner_steps + 1):
            agent.planner_step()
            action1 = agent.get_action(data.time, averaging_duration=0)
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
        set_cost_weights(agent, cost_weights)
        set_task_parameters(agent, task_parameters)

        synchronize_state()

        return get_time_step()

    def environment_step() -> tuple[npt.NDArray[np.float_], TimeStep]:
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
            "custom_data": custom_data,
            "num_planner_steps": num_planner_steps,
            "planner_step_tolerance": planner_step_tolerance,
        }

        with (path / "parameters").open("wt") as f:
            json.dump(parameters, f, indent=2, sort_keys=True)

        mujoco.mj_saveLastXML((path / "model.xml").as_posix(), model)

    def dump_video(path: Path) -> None:
        with mediapy.VideoWriter(
            path,
            shape=time_steps[0].frame.shape[0:2],
            fps=1.0 / model.opt.timestep,
        ) as video_writer:
            for time_step in time_steps:
                video_writer.add_image(time_step.frame)

    def dump_mujoco_states(path: Path) -> None:
        states = {
            "time": np.stack([ts.time for ts in time_steps]),
            "qpos": np.stack([ts.qpos for ts in time_steps]),
            "qvel": np.stack([ts.qvel for ts in time_steps]),
            "ctrl": np.stack([ts.ctrl for ts in time_steps]),
            "cost_total": np.stack([ts.cost_total for ts in time_steps]),
            "cost_terms": np.stack([ts.time for ts in time_steps]),
        }

        assert path.suffix == ".npz", path
        with path.open("wb") as f:
            np.savez_compressed(f, **states)

    if output_path is not None:
        time_now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = output_path / time_now_str
        output_dir.mkdir(parents=True, exist_ok=True)

        video_save_path = output_dir / "video.mp4"
        parameters_save_dir = output_dir
        mujoco_states_save_path = output_dir / "mujoco_states.npz"

        dump_parameters(parameters_save_dir)
        dump_video(video_save_path)
        dump_mujoco_states(mujoco_states_save_path)


if __name__ == "__main__":
    app.run(main)

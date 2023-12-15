"""Module documentation."""

import pathlib
import re
import subprocess

import matplotlib.pyplot as plt
import mediapy
import mujoco
import mujoco_mpc
import mujoco_mpc.agent
import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import seaborn as sns
from absl import app, flags, logging

Path = pathlib.Path

logging.set_verbosity(logging.INFO)


def create_plot_image(df, t, plot_size=(750, 250)):
    dpi = 300
    figure, axis = plt.subplots(
        1,
        1,
        figsize=(plot_size[0] / dpi, plot_size[1] / dpi),
        dpi=dpi,
    )
    df = df[["ts", "return_ptp"]].set_index("ts")
    sns.lineplot(df, ax=axis, legend=False, linewidth=0.5)
    axis.scatter(x=df.index[t], y=df.iloc[t], marker="|", c="k", s=100, zorder=1000)

    axis.set_ylim([df.values.min(), df.values.max()])
    axis.set_xlim([df.index.min(), df.index.max()])
    axis.axis("off")
    figure.tight_layout(pad=0.5, w_pad=0, h_pad=0)

    figure.canvas.draw()
    pixels = np.array(figure.canvas.buffer_rgba())
    plt.close(figure)
    plt.clf()
    plt.cla()
    image = PIL.Image.fromarray(pixels)
    return image


def add_planner_errors(image, df, t):
    # Convert to PIL Image
    image = PIL.Image.fromarray(image)
    draw = PIL.ImageDraw.Draw(image)
    try:
        font = PIL.ImageFont.truetype("SFNSMono.ttf", 40)
    except OSError:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 40)
    draw.text(
        (0, 250),
        df.iloc[t][["ts", "return_ptp"]]
        .rename({"return_ptp": "np.ptp(returns)"})
        .to_string(),
        font=font,
        fill=(255, 0, 0),
    )

    # Add plot to image
    plot_image = create_plot_image(df, t)
    image.paste(plot_image, (0, 0))  # Adjust position as needed

    return np.array(image)


def main(argv):
    del argv

    model_path = (
        Path(__file__).parent.parent
        / "mjpc"
        / "tasks"
        / "humanoid"
        / "walk"
        / "task.xml"
    )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offheight = 1080
    model.vis.global_.offwidth = 1920

    data = mujoco.MjData(model)

    np.testing.assert_equal(0.1, 0.1)

    agent = mujoco_mpc.agent.Agent(
        "Humanoid Walk",
        model,
        subprocess_kwargs={
            "stdout": subprocess.PIPE,
            "universal_newlines": True,
            "bufsize": 1,
            "close_fds": True,
        },
    )

    renderer = mujoco.Renderer(
        model,
        width=model.vis.global_.offwidth,
        height=model.vis.global_.offheight,
    )

    def get_observation(model, data):
        del model
        return np.concatenate([data.qpos, data.qvel])

    def environment_step(model, data, action):
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        return get_observation(model, data)

    def environment_reset(model, data):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        return get_observation(model, data)

    def compute_action_and_capture_diagnostics(agent, data):
        """Compute actions with `agent.planner_step` and capture diagnostics.

        The diagnostics are captured by parsing the output of the planner process.
        """
        agent.set_state(
            time=data.time,
            qpos=data.qpos,
            qvel=data.qvel,
            act=data.act,
            mocap_pos=data.mocap_pos,
            mocap_quat=data.mocap_quat,
            userdata=data.userdata,
        )

        action_change_tolerance = 1e-2
        action0 = agent.get_action(data.time)

        stdout_lines = []
        for _ in range(1000):
            agent.planner_step()
            action1 = agent.get_action(data.time)
            action_change = np.abs(action1 - action0)
            converged = (action_change < action_change_tolerance).all()
            action0 = action1

            stdout_lines.extend(
                list(iter(agent.server_process.stdout.readline, "END NOMINAL DEBUG\n"))
            )
            stdout_lines.append("END NOMINAL DEBUG\n")

            if converged:
                break

        start_indices = np.array(
            [
                i
                for i, line in enumerate(stdout_lines)
                if line.strip() == "BEGIN NOMINAL DEBUG"
            ]
        )
        end_indices = np.array(
            [
                i
                for i, line in enumerate(stdout_lines)
                if line.strip() == "END NOMINAL DEBUG"
            ]
        )
        np.testing.assert_equal(start_indices.size, end_indices.size)
        np.testing.assert_equal(start_indices[1:] - 1, end_indices[:-1])
        diagnostic_results = []
        for i, (start, end) in enumerate(zip(start_indices + 1, end_indices)):
            iteration_lines = stdout_lines[start:end]
            iteration_diagnostics = []
            for line in iteration_lines:
                line_match = re.search(
                    r"^j=([0-9]{2}); failure=([01]); return=([0-9\.]+);$", line
                )
                np.testing.assert_equal(len(line_match.groups()), 3)
                j_str, failure_str, return_str = line_match.groups()
                j = int(j_str)
                failure = bool(int(failure_str))
                return_ = float(return_str)
                diagnostic_results.append((i, j, failure, return_))

        diagnostics_dataframe = pd.DataFrame(
            diagnostic_results,
            columns=["planner_step_i", "trajectory_j", "failure", "return"],
        )

        return action0, diagnostics_dataframe

    def render_frame(model, data, renderer):
        del model
        renderer.update_scene(data)
        return renderer.render()

    actions = []
    observations = [environment_reset(model, data)]
    pixels = [render_frame(model, data, renderer)]

    simulation_length_s = 6.0
    num_steps = int(np.ceil(simulation_length_s / model.opt.timestep))
    video_fps = 50.0
    render_every = 1 / (model.opt.timestep * video_fps)
    np.testing.assert_equal(int(render_every), render_every)
    render_every = int(render_every)

    all_diagnostics = []
    render_diagnostics = []
    for t in range(num_steps):
        print(f"{t=}")
        action, diagnostics = compute_action_and_capture_diagnostics(agent, data)
        actions.append(action)
        diagnostics["t"] = t
        observations.append(environment_step(model, data, actions[-1]))
        all_diagnostics.append(diagnostics)

        if t % render_every == 0:
            pixels.append(render_frame(model, data, renderer))
            render_diagnostics.append(diagnostics)

    agent.close()  # Need to close here, otherwise the video writer will hang.

    actions = np.array(actions)
    observations = np.array(observations)
    np.testing.assert_equal(len(observations), len(actions) + 1)

    all_diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    render_diagnostics = pd.concat(render_diagnostics, ignore_index=True)

    render_df_rows = []
    # Modify the diagnostics to include in the video. By default, select the final
    # `planner_step` iteration for each timestep.
    for t in render_diagnostics["t"].unique():
        t_df = render_diagnostics[render_diagnostics["t"] == t]
        # Could also render e.g. the first `planner_step` iteration by selecting:
        # `t_df = t_df[t_df["planner_step_i"] == 0]`.
        t_df = t_df[t_df["planner_step_i"] == t_df["planner_step_i"].max()]
        t_return_ptp = np.ptp(t_df["return"])
        assert t_df["t"].unique().size == 1, t_df["t"]
        ts = t_df["t"].unique()[0] * model.opt.timestep
        render_df_rows.append((ts, t_return_ptp))

    render_df_rows.append(
        render_df_rows[-1]
    )  # Fix final step for which we don't have diagnostics.

    render_df = pd.DataFrame(render_df_rows, columns=["ts", "return_ptp"])

    with mediapy.VideoWriter(
        "/tmp/humanoid_walk.mp4",
        fps=video_fps,
        shape=(renderer.height, renderer.width),
    ) as video:
        for t, frame in enumerate(pixels):
            modified_frame = add_planner_errors(frame, render_df, t)
            video.add_image(modified_frame)


if __name__ == "__main__":
    app.run(main)

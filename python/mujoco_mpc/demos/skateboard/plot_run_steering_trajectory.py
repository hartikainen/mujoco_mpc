"""Plot trajectory obtained from `run_steering.py`.

Given the result from `run_steering.py` (`mujoco_states.npz`), this function plots the
2D trajectory of the skateboard.
"""

import pathlib
import json

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tree

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

    trial_paths = list(trial_path.glob("**/mujoco_states.npz"))

    trials = []

    def load_trial_path(trial_path: Path):
        with trial_path.open("rb") as f:
            data = dict(np.load(f))
        with trial_path.with_name("parameters.json").open("r") as f:
            parameters = json.load(f)
        return {"data": data, "parameters": parameters}

    trials = list(map(load_trial_path, trial_paths))

    stacked_parameters = tree.map_structure(lambda *xs: np.stack(xs), *(t["parameters"] for t in trials))

    def drop_single_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[c for c in df.columns if len(df[c].unique()) == 1])

    categories_df = pd.DataFrame(dict(tree.flatten_with_path({ k: v for k, v in stacked_parameters.items() if k in {"cost_weights", "task_parameters"} })))
    categories_df = drop_single_unique_columns(categories_df)
    categories_df.columns = categories_df.columns.to_flat_index().str.join("/")
    for column in categories_df.columns:
        categories_df[column] = categories_df[column].astype(str)
    useful_columns = categories_df.columns

    if useful_columns.size > 2:
        raise ValueError(f"Too many grouping columns: {useful_columns}")

    visualization_dfs = []
    for trial, (i, trial_category) in zip(trials, categories_df.iterrows()):
        x, y = trial["data"]["qpos"][0::10, 0:2]
        df = pd.DataFrame({"x": x, "y": y})

        for useful_column in useful_columns:
            df[useful_column] = trial_category[useful_column]

        visualization_dfs.append(df)

    visualization_df = pd.concat(visualization_dfs)

    figure, axis = plt.subplots(1, 1, figsize=(8, 8))

    sns.scatterplot(
        data=visualization_df,
        x="x",
        y="y",
        hue=useful_columns[0] if useful_columns.size > 0 else None,
        style=useful_columns[1] if useful_columns.size > 1 else None,
        ax=axis,
    )

    figure_save_path = trial_path / "trajectory.png"
    plt.savefig(figure_save_path)


if __name__ == "__main__":
    app.run(main)

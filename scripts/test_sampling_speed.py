"""Module documentation."""

import argparse
import functools
# import asyncio
from multiprocessing import Pool, TimeoutError
import pathlib
import re
import subprocess
import time

import numpy as np
import pandas as pd


MJPC_HEADLESS_EXECUTABLE_PATH = pathlib.Path(
    "~/github/deepmind/mujoco_mpc/build/bin/main_headless").expanduser()


def sample_sequence(seed=0, mocap_id=None):
    if mocap_id is None:
        mocap_id = "CMU-CMU-02-02_04_poses.xml"

    print(f"Sampling, {seed=}")
    result = subprocess.run(
        [str(MJPC_HEADLESS_EXECUTABLE_PATH),
         "--mocap_id",
         "CMU-CMU-02-02_04_poses.xml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    if result.returncode != 0:
        print(f"{result.stderr=}")
        raise ValueError(result)

    num_timesteps = None
    total_cost = None
    total_cost_per_step = None
    for line in result.stdout.split("\n"):
        if line.startswith("num_timesteps: "):
            match = re.match(r"^num_timesteps: (\d+)$", line)
            assert match is not None
            num_timesteps = int(match.group(1))
        elif line.startswith("total_cost: "):
            float_regex = r"([+-]?(?:\d+(?:\.\d+)?)|(?:\.\d+))"
            match = re.match(
                fr"^total_cost: {float_regex}; total_cost per step: {float_regex}$",
                line)
            total_cost = float(match.group(1))
            total_cost_per_step = float(match.group(2))

    assert num_timesteps is not None
    assert total_cost is not None
    assert total_cost_per_step is not None

    return {
        "num_timesteps": num_timesteps,
        "total_cost": total_cost,
        "total_cost_per_step": total_cost_per_step,
        "mocap_id": mocap_id,
        "seed": seed,
    }


def sample_in_parallel(num_trials_per_run, num_parallel_runs):
    print(f"{num_trials_per_run=}, {num_parallel_runs=}")

    start_time = time.time()
    with Pool(processes=num_parallel_runs) as pool:
        result = pool.map(
            sample_sequence, range(num_trials_per_run * num_parallel_runs))
    end_time = time.time()

    result_df = pd.DataFrame(result)
    # print(result_df)

    total_samples = result_df['num_timesteps'].sum()
    total_runtime_s = end_time - start_time
    samples_per_s = total_samples / total_runtime_s

    print(f"{total_samples=}, {total_runtime_s=}, {samples_per_s=}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num_trials_per_run", type=int, required=True)
    parser.add_argument(
        "-P", "--num_parallel_runs", type=int, nargs="+", required=True)
    args = parser.parse_args()

    # Build the binary before running
    subprocess.check_call(
        [
            "cmake",
            "--build",
            "~/github/deepmind/mujoco_mpc/build",
            "--config", "Debug",
            "--target", "all",
            "-j", "18",
            "--"
        ],
    )

    for num_parallel_runs in args.num_parallel_runs:
        sample_in_parallel(args.num_trials_per_run, num_parallel_runs)


if __name__ == "__main__":
    main()

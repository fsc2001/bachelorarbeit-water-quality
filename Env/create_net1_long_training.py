import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "NeuralSurrogateKalmanChlorineEstimation"
DATA_DIR = REPO / "data"

os.chdir(REPO)
sys.path.insert(0, str(REPO))

from create_data import LeakdDbChlorineInjectionEnv


N_STEPS = 5000
OUTPUT_STEM = "net1_randDemand=True_training_5000"


def main():
    scada_data = None
    control_actions = []

    with LeakdDbChlorineInjectionEnv(
        use_net1=True,
        randomized_demands=True,
    ) as env:

        env.action_space.seed(12345)
        env.reset()

        for step in range(N_STEPS):
            action = env.action_space.sample()

            control_actions.append(action)

            _, _, terminated, _, info = env.step(
                action
            )

            current_scada = info["scada_data"]

            if scada_data is None:
                scada_data = current_scada
            else:
                scada_data.concatenate(
                    current_scada
                )

            if terminated:
                print(
                    f"Simulation terminated after "
                    f"{step + 1} steps."
                )
                break

            if (step + 1) % 500 == 0:
                print(
                    f"{step + 1}/{N_STEPS} steps"
                )

    if scada_data is None:
        raise RuntimeError(
            "No SCADA data generated."
        )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scada_path = (
        DATA_DIR
        / f"{OUTPUT_STEM}.epytflow_scada_data"
    )

    actions_path = (
        DATA_DIR
        / f"{OUTPUT_STEM}.npz"
    )

    scada_data.save_to_file(
        str(scada_path)
    )

    np.savez(
        str(actions_path),
        control_actions=np.asarray(
            control_actions
        ),
    )

    print("\nSaved:")
    print(scada_path)
    print(actions_path)


if __name__ == "__main__":
    main()
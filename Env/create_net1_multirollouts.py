"""
This module creates multiple Net1 training rollouts for surrogate model training.
"""
import os
import sys
from pathlib import Path
from create_data import create_data_set

PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_REPO = PROJECT_DIR / "NeuralSurrogateKalmanChlorineEstimation"
os.chdir(REFERENCE_REPO)
sys.path.insert(0, str(REFERENCE_REPO))

ROLLOUTS = 5

def main():
    for rollout_index in range(ROLLOUTS):
        output_name = f"net1_randDemand=True_training_rollout{rollout_index}"

        print(f"Create net1 rollout {rollout_index + 1}/{ROLLOUTS}")

        create_data_set(
            use_net1=True,
            randomized_demands=True,
            file_out=output_name,
            path_out="data",
        )

    print("Net1 rollouts created")


if __name__ == "__main__":
    main()
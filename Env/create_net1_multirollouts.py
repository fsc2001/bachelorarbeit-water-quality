import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "NeuralSurrogateKalmanChlorineEstimation"

os.chdir(REPO)
sys.path.insert(0, str(REPO))

from create_data import create_data_set


N_ROLLOUTS = 5


def main() -> None:
    for rollout_idx in range(N_ROLLOUTS):
        file_out = (
            "net1_randDemand=True_"
            f"training_rollout{rollout_idx}"
        )

        print("\n" + "=" * 72)
        print(
            f"NET1 TRAINING ROLLOUT "
            f"{rollout_idx + 1}/{N_ROLLOUTS}"
        )
        print("=" * 72)

        create_data_set(
            use_net1=True,
            randomized_demands=True,
            file_out=file_out,
            path_out="data",
        )

    print("\nAlle Rollouts erzeugt.")


if __name__ == "__main__":
    main()
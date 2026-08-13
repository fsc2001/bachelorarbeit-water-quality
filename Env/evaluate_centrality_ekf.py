import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "NeuralSurrogateKalmanChlorineEstimation"
DATA_DIR = REPO / "data"
RESULT_DIR = ROOT / "results"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))


from Env.network_config import NETWORKS

from Env.fixed_placement_state_estimation import (
    run_state_estimation_fixed,
)


LOWER_BOUND = 0.3
UPPER_BOUND = 2.0

NETWORK_NAMES = [
    "net1",
    "hanoi",
    "cydbp",
]


def compute_metrics(
    true_values,
    predicted_values,
):
    error = (
        predicted_values
        - true_values
    )

    abs_error = np.abs(
        error
    )

    true_safe = (
        (true_values >= LOWER_BOUND)
        & (true_values <= UPPER_BOUND)
    )

    predicted_safe = (
        (predicted_values >= LOWER_BOUND)
        & (predicted_values <= UPPER_BOUND)
    )

    false_safe = (
        predicted_safe
        & ~true_safe
    )

    false_unsafe = (
        ~predicted_safe
        & true_safe
    )

    n_true_unsafe = int(
        np.sum(~true_safe)
    )

    n_true_safe = int(
        np.sum(true_safe)
    )

    return {
        "mae": float(
            np.mean(abs_error)
        ),

        "median_ae": float(
            np.median(abs_error)
        ),

        "rmse": float(
            np.sqrt(
                np.mean(error ** 2)
            )
        ),

        "max_ae": float(
            np.max(abs_error)
        ),

        "negative_fraction": float(
            np.mean(
                predicted_values < 0
            )
        ),

        "safe_accuracy": float(
            np.mean(
                predicted_safe
                == true_safe
            )
        ),

        "false_safe_given_unsafe": (
            float(
                np.sum(false_safe)
                / n_true_unsafe
            )
            if n_true_unsafe > 0
            else float("nan")
        ),

        "false_unsafe_given_safe": (
            float(
                np.sum(false_unsafe)
                / n_true_safe
            )
            if n_true_safe > 0
            else float("nan")
        ),
    }


def print_metrics(
    title,
    metrics,
):
    print()
    print(title)

    for name, value in metrics.items():
        print(
            f"  {name:<28}"
            f"{value:.6f}"
        )


if __name__ == "__main__":

    placement_path = (
        RESULT_DIR
        / "centrality_sensor_placements.json"
    )

    with placement_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        placements = json.load(
            file
        )

    for network_name in NETWORK_NAMES:

        network = NETWORKS[
            network_name
        ]

        placement = placements[
            network_name
        ]

        node_indices = placement[
            "node_indices"
        ]

        link_indices = placement[
            "link_indices"
        ]

        print()
        print("=" * 72)

        print(
            network.model_name,
            "CENTRALITY EKF"
        )

        print("=" * 72)

        print(
            "Node sensors:",
            len(node_indices)
        )

        print(
            "Link sensors:",
            len(link_indices)
        )

        prefix = (
            f"{network_name}_"
            "randDemand=True"
        )

        scada_path = (
            DATA_DIR
            / (
                f"{prefix}_"
                "test.epytflow_scada_data"
            )
        )

        actions_path = (
            DATA_DIR
            / f"{prefix}_test.npz"
        )

        model_path = (
            DATA_DIR
            / network.surrogate_filename
        )

        (
            chlorine_predictions,
            chlorine_true,
        ) = run_state_estimation_fixed(
            net_desc=network.model_name,
            scada_file_in=str(
                scada_path
            ),
            control_actions_file_in=str(
                actions_path
            ),
            state_transition_model_file_in=str(
                model_path
            ),
            node_indices=node_indices,
            link_indices=link_indices,
        )

        pred = np.concatenate(
            chlorine_predictions,
            axis=0,
        )

        true = np.concatenate(
            chlorine_true,
            axis=0,
        )

        n_nodes = network.n_nodes

        print(
            "Prediction min/max:",
            float(pred.min()),
            float(pred.max()),
        )

        all_metrics = compute_metrics(
            true,
            pred,
        )

        node_metrics = compute_metrics(
            true[:, :n_nodes],
            pred[:, :n_nodes],
        )

        link_metrics = compute_metrics(
            true[:, n_nodes:],
            pred[:, n_nodes:],
        )

        print_metrics(
            "All chlorine states:",
            all_metrics,
        )

        print_metrics(
            "Node chlorine:",
            node_metrics,
        )

        print_metrics(
            "Link chlorine:",
            link_metrics,
        )
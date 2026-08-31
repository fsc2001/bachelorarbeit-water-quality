"""
This module evaluates EKF state estimation using centrality-based sensor placements.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_REPO = PROJECT_DIR / "NeuralSurrogateKalmanChlorineEstimation"
DATA_DIR = REFERENCE_REPO / "data"
RESULTS_DIR = PROJECT_DIR / "results"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REFERENCE_REPO))

from Env.network_config import NETWORKS
from Env.ekf_state_estimation import run_state_estimation


LOWER_BOUND = 0.3
UPPER_BOUND = 2.0

NETWORK_NAMES = [
    "net1",
    "hanoi",
    "cydbp",
]


def compute_metrics(true_values, predicted_values):
    errors = predicted_values - true_values

    true_safe = (
        (true_values >= LOWER_BOUND)
        & (true_values <= UPPER_BOUND)
    )

    predicted_safe = (
        (predicted_values >= LOWER_BOUND)
        & (predicted_values <= UPPER_BOUND)
    )

    false_safe = predicted_safe & ~true_safe
    false_unsafe = ~predicted_safe & true_safe

    n_unsafe = np.sum(~true_safe)
    n_safe = np.sum(true_safe)

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "safe_accuracy": float(np.mean(predicted_safe == true_safe)),
        "false_safe_given_unsafe": (
            float(np.sum(false_safe) / n_unsafe)
            if n_unsafe > 0
            else float("nan")
        ),
        "false_unsafe_given_safe": (
            float(np.sum(false_unsafe) / n_safe)
            if n_safe > 0
            else float("nan")
        ),
    }


def print_metrics(name, metrics):
    print(
        f"{name}: "
        f"MAE={metrics['mae']:.4f}, "
        f"RMSE={metrics['rmse']:.4f}, "
        f"safe accuracy={metrics['safe_accuracy']:.4f}, "
        f"false safe={metrics['false_safe_given_unsafe']:.4f}, "
        f"false unsafe={metrics['false_unsafe_given_safe']:.4f}"
    )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    placement_path = RESULTS_DIR / "centrality_sensor_placements.json"

    with placement_path.open("r", encoding="utf-8") as file:
        placements = json.load(file)

    results = []

    for network_name in NETWORK_NAMES:
        network = NETWORKS[network_name]
        placement = placements[network_name]

        node_indices = placement["node_indices"]
        link_indices = placement["link_indices"]

        file_prefix = f"{network_name}_randDemand=True"

        scada_path = DATA_DIR / f"{file_prefix}_test.epytflow_scada_data"
        actions_path = DATA_DIR / f"{file_prefix}_test.npz"
        model_path = DATA_DIR / network.surrogate_filename

        chlorine_predictions, chlorine_true = run_state_estimation(
            net_desc=network.model_name,
            scada_file_in=str(scada_path),
            control_actions_file_in=str(actions_path),
            state_transition_model_file_in=str(model_path),
            node_indices=node_indices,
            link_indices=link_indices,
        )

        predicted = np.concatenate(chlorine_predictions, axis=0)
        true = np.concatenate(chlorine_true, axis=0)

        n_nodes = network.n_nodes

        metrics_by_scope = {
            "all": compute_metrics(true, predicted),
            "nodes": compute_metrics(
                true[:, :n_nodes],
                predicted[:, :n_nodes],
            ),
            "links": compute_metrics(
                true[:, n_nodes:],
                predicted[:, n_nodes:],
            ),
        }

        print(f"\n{network.model_name} centrality EKF")

        for scope, metrics in metrics_by_scope.items():
            print_metrics(scope.capitalize(), metrics)

            results.append(
                {
                    "network": network_name,
                    "scope": scope,
                    "node_sensors": len(node_indices),
                    "link_sensors": len(link_indices),
                    **metrics,
                }
            )

    output_path = RESULTS_DIR / "centrality_ekf_metrics.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
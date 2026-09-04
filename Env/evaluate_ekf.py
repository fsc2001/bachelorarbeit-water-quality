"""
This module evaluates EKF state estimation for the final
random and centrality-based sensor placements.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = (
    PROJECT_DIR
    / "NeuralSurrogateKalmanChlorineEstimation"
    / "data"
)
RESULTS_DIR = PROJECT_DIR / "results"

sys.path.insert(0, str(PROJECT_DIR))

from Env.network_config import NETWORKS
from Env.ekf_state_estimation import run_state_estimation


NETWORK_NAMES = ("net1", "hanoi", "cydbp")
PLACEMENT_TYPES = ("random", "centrality")

DATA_SPLIT = "test"

LOWER_BOUND = 0.3
UPPER_BOUND = 2.0


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
        "safe_accuracy": float(
            np.mean(predicted_safe == true_safe)
        ),
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


def evaluate_placement(
    network_name,
    placement_type,
    placements,
):
    network = NETWORKS[network_name]
    placement = placements[network_name][placement_type]

    node_indices = placement["node_indices"]
    link_indices = placement["link_indices"]

    file_prefix = f"{network_name}_randDemand=True"

    scada_path = (
        DATA_DIR
        / f"{file_prefix}_{DATA_SPLIT}.epytflow_scada_data"
    )

    actions_path = (
        DATA_DIR
        / f"{file_prefix}_{DATA_SPLIT}.npz"
    )

    surrogate_path = (
        DATA_DIR
        / network.surrogate_filename
    )

    chlorine_predictions, chlorine_true = run_state_estimation(
        network_name=network.model_name,
        scada_path=str(scada_path),
        actions_path=str(actions_path),
        surrogate_path=str(surrogate_path),
        node_indices=node_indices,
        link_indices=link_indices,
    )

    predicted = np.concatenate(
        chlorine_predictions,
        axis=0,
    )

    true = np.concatenate(
        chlorine_true,
        axis=0,
    )

    n_nodes = network.n_nodes

    metrics = {
        "all": compute_metrics(
            true,
            predicted,
        ),
        "nodes": compute_metrics(
            true[:, :n_nodes],
            predicted[:, :n_nodes],
        ),
        "links": compute_metrics(
            true[:, n_nodes:],
            predicted[:, n_nodes:],
        ),
    }

    return metrics, node_indices, link_indices


def print_metrics(
    network_name,
    placement_type,
    metrics,
):
    print(
        f"\n{NETWORKS[network_name].model_name} "
        f"{placement_type} EKF"
    )

    for scope in ("all", "nodes", "links"):
        values = metrics[scope]

        print(
            f"{scope.capitalize()}: "
            f"MAE={values['mae']:.4f}, "
            f"RMSE={values['rmse']:.4f}, "
            f"safe accuracy={values['safe_accuracy']:.4f}, "
            f"false safe={values['false_safe_given_unsafe']:.4f}, "
            f"false unsafe={values['false_unsafe_given_safe']:.4f}"
        )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    placement_path = (
        RESULTS_DIR
        / "ppo_sensor_placements.json"
    )

    with placement_path.open("r", encoding="utf-8") as file:
        placements = json.load(file)

    rows = []

    for network_name in NETWORK_NAMES:
        for placement_type in PLACEMENT_TYPES:
            (
                metrics,
                node_indices,
                link_indices,
            ) = evaluate_placement(
                network_name,
                placement_type,
                placements,
            )

            print_metrics(
                network_name,
                placement_type,
                metrics,
            )

            for scope, values in metrics.items():
                rows.append(
                    {
                        "network": network_name,
                        "placement": placement_type,
                        "split": DATA_SPLIT,
                        "n_node_sensors": len(node_indices),
                        "n_link_sensors": len(link_indices),
                        "scope": scope,
                        **values,
                    }
                )

    output_path = (
        RESULTS_DIR
        / f"ekf_final_placements_{DATA_SPLIT}.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
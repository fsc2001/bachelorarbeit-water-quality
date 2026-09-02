"""
Evaluate EKF state estimation for the final random and centrality
sensor placements used in the PPO experiments.
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

PLACEMENT_PATH = RESULTS_DIR / "ppo_sensor_placements.json"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REFERENCE_REPO))

from Env.network_config import NETWORKS
from Env.ekf_state_estimation import run_state_estimation


NETWORK_NAMES = ("net1", "hanoi", "cydbp")
PLACEMENT_TYPES = ("random", "centrality")

DATA_SPLIT = "test"
RANDOMIZED_DEMANDS = True

LOWER_BOUND = 0.3
UPPER_BOUND = 2.0


def load_sensor_placements():
    if not PLACEMENT_PATH.exists():
        raise FileNotFoundError(
            f"Sensor placement file not found: {PLACEMENT_PATH}"
        )

    with PLACEMENT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def get_sensor_indices(
    placements,
    network_name,
    placement_type,
):
    try:
        placement = placements[
            network_name
        ][placement_type]
    except KeyError as exc:
        raise KeyError(
            f"Missing placement '{placement_type}' "
            f"for network '{network_name}' in {PLACEMENT_PATH}."
        ) from exc

    node_indices = list(placement["node_indices"])
    link_indices = list(placement["link_indices"])

    return node_indices, link_indices


def evaluate_placement(
    network_name,
    placement_type,
    placements,
):
    network = NETWORKS[network_name]

    node_indices, link_indices = get_sensor_indices(
        placements,
        network_name,
        placement_type,
    )

    file_prefix = (
        f"{network_name}_randDemand={RANDOMIZED_DEMANDS}"
    )

    scada_path = DATA_DIR / (
        f"{file_prefix}_{DATA_SPLIT}.epytflow_scada_data"
    )

    actions_path = DATA_DIR / (
        f"{file_prefix}_{DATA_SPLIT}.npz"
    )

    model_path = DATA_DIR / network.surrogate_filename

    for path in (scada_path, actions_path, model_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    chlorine_predictions, chlorine_true = run_state_estimation(
        net_desc=network.model_name,
        scada_file_in=str(scada_path),
        control_actions_file_in=str(actions_path),
        state_transition_model_file_in=str(model_path),
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

    if predicted.shape != true.shape:
        raise RuntimeError(
            f"Prediction shape {predicted.shape} does not match "
            f"ground truth shape {true.shape}."
        )

    if not np.all(np.isfinite(predicted)):
        raise RuntimeError(
            f"EKF produced non-finite predictions for "
            f"{network_name} / {placement_type}."
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


def save_results(rows):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main():
    placements = load_sensor_placements()
    rows = []

    print(
        f"EKF evaluation using final PPO sensor placements "
        f"({DATA_SPLIT} split)"
    )

    for network_name in NETWORK_NAMES:
        for placement_type in PLACEMENT_TYPES:
            (
                metrics,
                node_indices,
                link_indices,
            ) = evaluate_placement(
                network_name=network_name,
                placement_type=placement_type,
                placements=placements,
            )

            print_metrics(
                network_name,
                placement_type,
                metrics,
            )

            for scope, scope_metrics in metrics.items():
                rows.append(
                    {
                        "network": network_name,
                        "placement": placement_type,
                        "split": DATA_SPLIT,
                        "n_node_sensors": len(node_indices),
                        "n_link_sensors": len(link_indices),
                        "node_sensor_indices": ";".join(
                            map(str, node_indices)
                        ),
                        "link_sensor_indices": ";".join(
                            map(str, link_indices)
                        ),
                        "scope": scope,
                        **scope_metrics,
                    }
                )

    output_path = save_results(rows)

    print(
        f"\nSaved results to: {output_path}"
    )


if __name__ == "__main__":
    main()

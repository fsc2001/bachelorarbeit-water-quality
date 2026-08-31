"""
This module evaluates EKF state estimation using random sensor placements.
"""

import csv
import random
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


NETWORK_NAME = "hanoi"
DATA_SPLIT = "validation"

LOWER_BOUND = 0.3
UPPER_BOUND = 2.0

SEEDS = list(range(5))
SENSOR_COUNTS = [10]

NETWORK = NETWORKS[NETWORK_NAME]


def get_random_sensor_indices(n_sensors, seed):
    rng = random.Random(seed)

    node_indices = rng.sample(
        range(NETWORK.n_nodes),
        k=n_sensors,
    )

    link_indices = rng.sample(
        range(NETWORK.n_links),
        k=n_sensors,
    )

    node_indices.sort()
    link_indices.sort()

    return node_indices, link_indices


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


def run_evaluation(n_sensors, seed):
    if n_sensors > NETWORK.n_nodes or n_sensors > NETWORK.n_links:
        raise ValueError(
            f"{n_sensors} sensors are not available for {NETWORK.model_name}."
        )

    node_indices, link_indices = get_random_sensor_indices(
        n_sensors,
        seed,
    )

    file_prefix = f"{NETWORK_NAME}_randDemand=True"

    scada_path = DATA_DIR / (
        f"{file_prefix}_{DATA_SPLIT}.epytflow_scada_data"
    )

    actions_path = DATA_DIR / f"{file_prefix}_{DATA_SPLIT}.npz"
    model_path = DATA_DIR / NETWORK.surrogate_filename

    chlorine_predictions, chlorine_true = run_state_estimation(
        net_desc=NETWORK.model_name,
        scada_file_in=str(scada_path),
        control_actions_file_in=str(actions_path),
        state_transition_model_file_in=str(model_path),
        node_indices=node_indices,
        link_indices=link_indices,
    )

    predicted = np.concatenate(chlorine_predictions, axis=0)
    true = np.concatenate(chlorine_true, axis=0)

    if predicted.shape != true.shape:
        raise RuntimeError(
            f"Prediction shape {predicted.shape} does not match "
            f"ground truth shape {true.shape}."
        )

    if not np.all(np.isfinite(predicted)):
        raise RuntimeError("EKF produced non-finite predictions.")

    n_nodes = NETWORK.n_nodes

    metrics = {
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

    return metrics, node_indices, link_indices


def aggregate_metrics(runs, scope):
    aggregated = {}

    for metric_name in runs[0][scope]:
        values = np.array(
            [run[scope][metric_name] for run in runs],
            dtype=float,
        )

        aggregated[metric_name] = {
            "mean": float(np.nanmean(values)),
            "std": (
                float(np.nanstd(values, ddof=1))
                if len(values) > 1
                else 0.0
            ),
        }

    return aggregated


def print_metrics(name, metrics):
    values = []

    for metric_name, result in metrics.items():
        values.append(
            f"{metric_name}={result['mean']:.4f} ± {result['std']:.4f}"
        )

    print(f"{name}: " + ", ".join(values))


def save_results(rows):
    RESULTS_DIR.mkdir(exist_ok=True)

    output_path = (
        RESULTS_DIR
        / f"{NETWORK_NAME}_ekf_random_sensors_{DATA_SPLIT}.csv"
    )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main():
    rows = []

    print(f"{NETWORK.model_name} random sensor EKF")

    for n_sensors in SENSOR_COUNTS:
        runs = []

        print(
            f"\nEvaluating {n_sensors} node sensors and "
            f"{n_sensors} link sensors"
        )

        for seed in SEEDS:
            print(f"Seed {seed}")

            metrics, node_indices, link_indices = run_evaluation(
                n_sensors,
                seed,
            )

            runs.append(metrics)

            row = {
                "network": NETWORK_NAME,
                "n_sensors": n_sensors,
                "seed": seed,
                "node_sensor_indices": ";".join(map(str, node_indices)),
                "link_sensor_indices": ";".join(map(str, link_indices)),
            }

            for scope, scope_metrics in metrics.items():
                for metric_name, value in scope_metrics.items():
                    row[f"{scope}_{metric_name}"] = value

            rows.append(row)

        print_metrics(
            "All states",
            aggregate_metrics(runs, "all"),
        )

        print_metrics(
            "Nodes",
            aggregate_metrics(runs, "nodes"),
        )

        print_metrics(
            "Links",
            aggregate_metrics(runs, "links"),
        )

    output_path = save_results(rows)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
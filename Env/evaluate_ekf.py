import csv
import random
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
from run_exp_state_estimation import run_state_estimation


# ============================================================
# Configuration
# ============================================================

NETWORK_NAME = "hanoi"
NETWORK = NETWORKS[NETWORK_NAME]

RANDOMIZED_DEMANDS = True
DATA_SPLIT = "validation"

LOWER_BOUND = 0.3
UPPER_BOUND = 2.0

SEEDS = list(range(5))

# Net1:  z. B. [1, 2, 3]
# Hanoi: z. B. [3, 5, 10]
# CY-DBP: [25, 50, 75]
SENSOR_COUNTS = [10]

N_NODES = NETWORK.n_nodes
N_LINKS = NETWORK.n_links

DATASET_PREFIX = (
    f"{NETWORK_NAME}_randDemand={RANDOMIZED_DEMANDS}"
)


# ============================================================
# Sensor placement logging
# ============================================================

def get_sensor_indices(
    n_sensors: int,
    seed: int,
):
    """
    Reproduces exactly the random sensor selection used by
    create_random_sensor_placement() in the provided repository:

        nodes_idx = random.sample(...)
        links_idx = random.sample(...)

    A local Random instance is used here so that logging the
    placement does not modify the global random state.
    """
    rng = random.Random(seed)

    nodes_idx = rng.sample(
        range(N_NODES),
        k=n_sensors,
    )

    links_idx = rng.sample(
        range(N_LINKS),
        k=n_sensors,
    )

    nodes_idx.sort()
    links_idx.sort()

    return nodes_idx, links_idx


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> dict:

    error = predicted_values - true_values
    abs_error = np.abs(error)

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

    true_unsafe = ~true_safe

    n_true_unsafe = int(
        np.sum(true_unsafe)
    )

    n_true_safe = int(
        np.sum(true_safe)
    )

    false_safe_given_unsafe = (
        float(
            np.sum(false_safe)
            / n_true_unsafe
        )
        if n_true_unsafe > 0
        else float("nan")
    )

    false_unsafe_given_safe = (
        float(
            np.sum(false_unsafe)
            / n_true_safe
        )
        if n_true_safe > 0
        else float("nan")
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
                predicted_values < 0.0
            )
        ),

        "safe_classification_accuracy": float(
            np.mean(
                predicted_safe == true_safe
            )
        ),

        "false_safe_fraction": float(
            np.mean(false_safe)
        ),

        "false_unsafe_fraction": float(
            np.mean(false_unsafe)
        ),

        "false_safe_given_unsafe": (
            false_safe_given_unsafe
        ),

        "false_unsafe_given_safe": (
            false_unsafe_given_safe
        ),
    }


# ============================================================
# Single EKF run
# ============================================================

def run_single_evaluation(
    n_sensors: int,
    seed: int,
) -> dict:

    if n_sensors > N_NODES:
        raise ValueError(
            f"{n_sensors} node sensors requested, "
            f"but {NETWORK_NAME} only has "
            f"{N_NODES} nodes."
        )

    if n_sensors > N_LINKS:
        raise ValueError(
            f"{n_sensors} link sensors requested, "
            f"but {NETWORK_NAME} only has "
            f"{N_LINKS} links."
        )

    # Determine the placement for logging only.
    # This uses the same sampling sequence as the
    # professor's create_random_sensor_placement().
    (
        node_sensor_indices,
        link_sensor_indices,
    ) = get_sensor_indices(
        n_sensors=n_sensors,
        seed=seed,
    )

    # The provided repository uses Python's global
    # random module inside create_random_sensor_placement().
    random.seed(seed)

    # Currently not required by sensor placement, but keeps
    # the complete evaluation deterministic if NumPy randomness
    # is introduced elsewhere.
    np.random.seed(seed)

    scada_path = (
            DATA_DIR
            / (
                f"{DATASET_PREFIX}_"
                f"{DATA_SPLIT}.epytflow_scada_data"
            )
    )

    actions_path = (
            DATA_DIR
            / f"{DATASET_PREFIX}_{DATA_SPLIT}.npz"
    )

    model_path = (
        DATA_DIR
        / NETWORK.surrogate_filename
    )

    (
        _,
        chlorine_predictions,
        chlorine_true,
        _,
        _,
        _,
        _,
    ) = run_state_estimation(
        net_desc=NETWORK.model_name,

        scada_file_in=str(
            scada_path
        ),

        control_actions_file_in=str(
            actions_path
        ),

        state_transition_model_file_in=str(
            model_path
        ),

        n_node_quality_sensors=n_sensors,
        n_link_sensors=n_sensors,
    )

    predictions = np.concatenate(
        chlorine_predictions,
        axis=0,
    )

    true_values = np.concatenate(
        chlorine_true,
        axis=0,
    )

    if predictions.shape != true_values.shape:
        raise RuntimeError(
            "Prediction and ground truth have "
            "different shapes: "
            f"{predictions.shape} vs. "
            f"{true_values.shape}"
        )

    expected_states = (
        N_NODES + N_LINKS
    )

    if predictions.shape[1] != expected_states:
        raise RuntimeError(
            "Unexpected number of chlorine states: "
            f"{predictions.shape[1]}, "
            f"expected {expected_states}"
        )

    if not np.all(
        np.isfinite(predictions)
    ):
        raise RuntimeError(
            "EKF produced non-finite predictions."
        )

    node_predictions = predictions[
        :,
        :N_NODES,
    ]

    node_true = true_values[
        :,
        :N_NODES,
    ]

    link_predictions = predictions[
        :,
        N_NODES:,
    ]

    link_true = true_values[
        :,
        N_NODES:,
    ]

    return {
        "all": compute_metrics(
            true_values,
            predictions,
        ),

        "nodes": compute_metrics(
            node_true,
            node_predictions,
        ),

        "links": compute_metrics(
            link_true,
            link_predictions,
        ),

        "pred_min": float(
            predictions.min()
        ),

        "pred_max": float(
            predictions.max()
        ),

        "n_time_steps": int(
            predictions.shape[0]
        ),

        "node_sensor_indices": (
            node_sensor_indices
        ),

        "link_sensor_indices": (
            link_sensor_indices
        ),
    }


# ============================================================
# Aggregation
# ============================================================

def aggregate_runs(
    runs: list,
    section: str,
) -> dict:

    metric_names = runs[
        0
    ][section].keys()

    result = {}

    for metric_name in metric_names:

        values = np.asarray(
            [
                run[section][metric_name]
                for run in runs
            ],
            dtype=np.float64,
        )

        mean = float(
            np.nanmean(values)
        )

        # ddof=1 requires at least two runs.
        # For a one-seed diagnostic run, std=0 avoids
        # the NumPy warning and the meaningless "nan".
        if len(values) > 1:
            std = float(
                np.nanstd(
                    values,
                    ddof=1,
                )
            )
        else:
            std = 0.0

        result[metric_name] = {
            "mean": mean,
            "std": std,
        }

    return result


def print_section(
    title: str,
    aggregated: dict,
) -> None:

    print(f"\n{title}:")

    for (
        metric_name,
        values,
    ) in aggregated.items():

        print(
            f"  {metric_name:<30}"
            f"{values['mean']:.6f}"
            f" ± {values['std']:.6f}"
        )


# ============================================================
# Result storage
# ============================================================

def save_results(
    rows: list,
) -> Path:

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
            RESULT_DIR
            / (
                f"{NETWORK_NAME}_"
                f"ekf_random_sensors_{DATA_SPLIT}.csv"
            )
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    return output_path


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        f"{NETWORK.model_name} "
        "EKF evaluation"
    )

    print(
        "Seeds:",
        SEEDS,
    )

    print(
        "Sensor counts:",
        SENSOR_COUNTS,
    )

    print(
        "Surrogate:",
        NETWORK.surrogate_filename,
    )

    raw_rows = []

    for n_sensors in SENSOR_COUNTS:

        print(
            "\n" + "=" * 72
        )

        print(
            f"{n_sensors} node sensor(s) + "
            f"{n_sensors} link sensor(s)"
        )

        print(
            "=" * 72
        )

        runs = []

        for seed in SEEDS:

            print(
                f"Starting seed {seed} ..."
            )

            result = (
                run_single_evaluation(
                    n_sensors=n_sensors,
                    seed=seed,
                )
            )

            runs.append(
                result
            )

            print(
                "  Prediction min/max:",
                f"{result['pred_min']:.4f}",
                f"{result['pred_max']:.4f}",
            )

            print(
                "  Time steps:",
                result["n_time_steps"],
            )

            print(
                "  Node sensors:",
                result[
                    "node_sensor_indices"
                ],
            )

            print(
                "  Link sensors:",
                result[
                    "link_sensor_indices"
                ],
            )

            row = {
                "network": NETWORK_NAME,
                "n_sensors": n_sensors,
                "seed": seed,

                "pred_min": (
                    result["pred_min"]
                ),

                "pred_max": (
                    result["pred_max"]
                ),

                # Semicolon-separated values are easier
                # to read again from the CSV.
                "node_sensor_indices": (
                    ";".join(
                        map(
                            str,
                            result[
                                "node_sensor_indices"
                            ],
                        )
                    )
                ),

                "link_sensor_indices": (
                    ";".join(
                        map(
                            str,
                            result[
                                "link_sensor_indices"
                            ],
                        )
                    )
                ),
            }

            for section in (
                "all",
                "nodes",
                "links",
            ):

                for (
                    metric_name,
                    value,
                ) in result[
                    section
                ].items():

                    row[
                        f"{section}_{metric_name}"
                    ] = value

            raw_rows.append(
                row
            )

        print_section(
            "All chlorine states",
            aggregate_runs(
                runs,
                "all",
            ),
        )

        print_section(
            "Node chlorine",
            aggregate_runs(
                runs,
                "nodes",
            ),
        )

        print_section(
            "Link chlorine",
            aggregate_runs(
                runs,
                "links",
            ),
        )

    output_path = save_results(
        raw_rows
    )

    print(
        "\nResults saved:"
    )

    print(
        output_path
    )
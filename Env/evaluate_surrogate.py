"""
This module evaluates the surrogate model on validation and test data.
"""

import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from epyt_flow.simulation import ScadaData


PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_REPO = PROJECT_DIR / "NeuralSurrogateKalmanChlorineEstimation"
DATA_DIR = REFERENCE_REPO / "data"
RESULTS_DIR = PROJECT_DIR / "results"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REFERENCE_REPO))

from Env.network_config import NETWORKS
from run_exp_state_estimation import get_state_transition_model


NETWORK_NAME = "net1"
RANDOMIZED_DEMANDS = True

LOWER_BOUND = 0.3
UPPER_BOUND = 2.0

NETWORK = NETWORKS[NETWORK_NAME]
N_CHLORINE = NETWORK.n_nodes + NETWORK.n_links


def compute_metrics(true_values, predicted_values):
    errors = predicted_values - true_values

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }


def print_metrics(name, metrics):
    print(
        f"{name}: "
        f"MAE={metrics['mae']:.4f} mg/L, "
        f"RMSE={metrics['rmse']:.4f} mg/L"
    )

def diagnose_predictions(
    split: str,
    true_nodes: np.ndarray,
    predicted_nodes: np.ndarray,
    true_links: np.ndarray,
    predicted_links: np.ndarray,
) -> None:

    print("\n" + "-" * 70)
    print(f"DIAGNOSTICS - {split.upper()}")
    print("-" * 70)

    print("Number of transitions:", true_nodes.shape[0])

    def describe(name, values):
        print(
            f"{name}: "
            f"min={values.min():.4f}, "
            f"max={values.max():.4f}, "
            f"mean={values.mean():.4f}, "
            f"std={values.std():.4f}"
        )

    describe("True nodes     ", true_nodes)
    describe("Predicted nodes", predicted_nodes)
    describe("True links     ", true_links)
    describe("Predicted links", predicted_links)

    node_error = predicted_nodes - true_nodes

    print(
        "Node mean error / bias:",
        float(np.mean(node_error)),
    )

    # Drei Nodes mit der größten zeitlichen Variation auswählen.
    node_variance = np.var(true_nodes, axis=0)

    selected_nodes = np.argsort(
        node_variance
    )[-3:]

    n_plot_steps = min(
        100,
        true_nodes.shape[0],
    )

    for node_idx in selected_nodes:
        plt.figure()

        plt.plot(
            true_nodes[:n_plot_steps, node_idx],
            label="True",
        )

        plt.plot(
            predicted_nodes[:n_plot_steps, node_idx],
            label="Predicted",
        )

        plt.xlabel("Time step")
        plt.ylabel("Chlorine concentration [mg/L]")
        plt.title(
            f"{split}: Node {node_idx}"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

def evaluate_dataset(split):
    dataset_name = (
        f"{NETWORK_NAME}_randDemand={RANDOMIZED_DEMANDS}_{split}"
    )

    scada_path = DATA_DIR / f"{dataset_name}.epytflow_scada_data"
    actions_path = DATA_DIR / f"{dataset_name}.npz"
    model_path = DATA_DIR / NETWORK.surrogate_filename

    if not scada_path.exists():
        raise FileNotFoundError(f"SCADA dataset not found: {scada_path}")

    if not actions_path.exists():
        raise FileNotFoundError(f"Action dataset not found: {actions_path}")

    scada_data = ScadaData.load_from_file(str(scada_path))

    control_actions = np.load(
        actions_path,
        allow_pickle=False,
    )["control_actions"]

    node_quality = np.asarray(scada_data.get_data_nodes_quality())
    link_quality = np.asarray(scada_data.get_data_links_quality())
    flows = np.asarray(scada_data.get_data_flows())

    current_states = np.concatenate(
        (
            node_quality[:-1],
            link_quality[:-1],
            flows[1:],
        ),
        axis=1,
    )

    control_actions = control_actions[:current_states.shape[0]]

    true_next_chlorine = np.concatenate(
        (
            node_quality[1:],
            link_quality[1:],
        ),
        axis=1,
    )

    persistence_prediction = np.concatenate(
        (
            node_quality[:-1],
            link_quality[:-1],
        ),
        axis=1,
    )

    model = get_state_transition_model(
        NETWORK.model_name,
        str(model_path),
    )

    model.n_missing_flows = N_CHLORINE
    model._normalize_input_output = False

    model_input = np.concatenate(
        (
            current_states,
            control_actions,
        ),
        axis=1,
    )

    scaled_input = model._scaler.transform(model_input)
    scaled_states = scaled_input[:, :current_states.shape[1]]

    scaled_controls = scaled_input[
                      :,
                      current_states.shape[1]:
                      ]

    scaled_prediction = np.asarray(
        model.predict(
            scaled_states,
            scaled_controls,
        )
    )

    prediction_with_controls = np.concatenate(
        (
            scaled_prediction,
            np.zeros(
                (
                    scaled_prediction.shape[0],
                    control_actions.shape[1],
                )
            ),
        ),
        axis=1,
    )

    prediction = model._scaler.inverse_transform(
        prediction_with_controls
    )[:, :current_states.shape[1]]

    predicted_chlorine = prediction[:, :N_CHLORINE]

    n_nodes = NETWORK.n_nodes

    true_nodes = true_next_chlorine[:, :n_nodes]
    true_links = true_next_chlorine[:, n_nodes:]

    predicted_nodes = predicted_chlorine[:, :n_nodes]
    predicted_links = predicted_chlorine[:, n_nodes:]

    diagnose_predictions(
        split,
        true_nodes,
        predicted_nodes,
        true_links,
        predicted_links,
    )

    model_metrics = {
        "all": compute_metrics(
            true_next_chlorine,
            predicted_chlorine,
        ),
        "nodes": compute_metrics(
            true_nodes,
            predicted_nodes,
        ),
        "links": compute_metrics(
            true_links,
            predicted_links,
        ),
    }

    persistence_metrics = {
        "all": compute_metrics(
            true_next_chlorine,
            persistence_prediction,
        ),
        "nodes": compute_metrics(
            true_next_chlorine[:, :n_nodes],
            persistence_prediction[:, :n_nodes],
        ),
        "links": compute_metrics(
            true_next_chlorine[:, n_nodes:],
            persistence_prediction[:, n_nodes:],
        ),
    }

    control_mask = (
        (true_next_chlorine >= LOWER_BOUND)
        & (true_next_chlorine <= UPPER_BOUND)
    )

    control_metrics = compute_metrics(
        true_next_chlorine[control_mask],
        predicted_chlorine[control_mask],
    )

    print(f"\n{NETWORK.model_name} surrogate - {split}")
    print_metrics("All states", model_metrics["all"])
    print_metrics("Nodes", model_metrics["nodes"])
    print_metrics("Links", model_metrics["links"])
    print_metrics("Persistence baseline", persistence_metrics["all"])
    print_metrics("Control-relevant range", control_metrics)

    return model_metrics, persistence_metrics, control_metrics


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []

    training_path = (
            DATA_DIR
            / f"{NETWORK_NAME}_randDemand={RANDOMIZED_DEMANDS}_training.epytflow_scada_data"
    )

    training_scada = ScadaData.load_from_file(
        str(training_path)
    )

    training_steps = np.asarray(
        training_scada.get_data_nodes_quality()
    ).shape[0]

    print(
        "\nTraining time steps:",
        training_steps,
    )

    print(
        "Training transitions:",
        training_steps - 1,
    )

    for split in ("training", "validation", "test"):
        model_metrics, persistence_metrics, control_metrics = (
            evaluate_dataset(split)
        )

        for scope in ("all", "nodes", "links"):
            rows.append(
                {
                    "network": NETWORK_NAME,
                    "split": split,
                    "model": "surrogate",
                    "scope": scope,
                    **model_metrics[scope],
                }
            )

            rows.append(
                {
                    "network": NETWORK_NAME,
                    "split": split,
                    "model": "persistence",
                    "scope": scope,
                    **persistence_metrics[scope],
                }
            )

        rows.append(
            {
                "network": NETWORK_NAME,
                "split": split,
                "model": "surrogate",
                "scope": "control_range",
                **control_metrics,
            }
        )

    output_path = RESULTS_DIR / f"{NETWORK_NAME}_surrogate_evaluation.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
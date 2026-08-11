import sys
from pathlib import Path

import numpy as np
from epyt_flow.simulation import ScadaData


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "NeuralSurrogateKalmanChlorineEstimation"
DATA_DIR = REPO / "data"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

from Env.network_config import NETWORKS
from run_exp_state_estimation import get_state_transition_model


# ============================================================
# Configuration
# ============================================================

NETWORK_NAME = "net1"
NETWORK = NETWORKS[NETWORK_NAME]

RANDOMIZED_DEMANDS = True

DATASET_PREFIX = (
    f"{NETWORK_NAME}_randDemand={RANDOMIZED_DEMANDS}"
)

N_NODES = NETWORK.n_nodes
N_LINKS = NETWORK.n_links
N_CHLORINE = N_NODES + N_LINKS


# ============================================================
# Metrics
# ============================================================

def print_metrics(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    error = y_pred - y_true

    mae = np.mean(np.abs(error))
    median_ae = np.median(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    max_ae = np.max(np.abs(error))

    print(f"\n{name}:")
    print(f"  MAE:       {mae:.6f} mg/L")
    print(f"  Median AE: {median_ae:.6f} mg/L")
    print(f"  RMSE:      {rmse:.6f} mg/L")
    print(f"  Max AE:    {max_ae:.6f} mg/L")


# ============================================================
# Evaluation
# ============================================================

def evaluate_dataset(split: str) -> None:
    dataset_name = f"{DATASET_PREFIX}_{split}"

    scada_path = (
        DATA_DIR
        / f"{dataset_name}.epytflow_scada_data"
    )

    actions_path = (
        DATA_DIR
        / f"{dataset_name}.npz"
    )

    model_path = (
            DATA_DIR
            / NETWORK.surrogate_filename
    )

    if not scada_path.exists():
        raise FileNotFoundError(
            f"SCADA dataset not found: {scada_path}"
        )

    if not actions_path.exists():
        raise FileNotFoundError(
            f"Action dataset not found: {actions_path}"
        )

    scada = ScadaData.load_from_file(
        str(scada_path)
    )

    actions = np.asarray(
        np.load(
            actions_path,
            allow_pickle=False,
        )["control_actions"]
    )

    node_quality = np.asarray(
        scada.get_data_nodes_quality()
    )

    link_quality = np.asarray(
        scada.get_data_links_quality()
    )

    flows = np.asarray(
        scada.get_data_flows()
    )

    n_time_steps = node_quality.shape[0]

    # Gleicher Aufbau wie im Referenzcode:
    # Chlor(t) sowie bereits bekannte Flows(t+1).
    current_state = np.concatenate(
        (
            node_quality[:-1],
            link_quality[:-1],
            flows[1:],
        ),
        axis=1,
    )

    control = actions[:n_time_steps - 1]

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

    # Skalierung wie im offiziellen EKF-Code
    # explizit selbst durchführen.
    model._normalize_input_output = False

    scaler_input = np.concatenate(
        (
            current_state,
            control,
        ),
        axis=1,
    )

    scaled_input = model._scaler.transform(
        scaler_input
    )

    scaled_state = scaled_input[
        :,
        :current_state.shape[1],
    ]

    scaled_prediction = np.asarray(
        model.predict(
            scaled_state,
            control,
        )
    )

    inverse_input = np.concatenate(
        (
            scaled_prediction,
            np.zeros(
                (
                    scaled_prediction.shape[0],
                    control.shape[1],
                )
            ),
        ),
        axis=1,
    )

    prediction = model._scaler.inverse_transform(
        inverse_input
    )[:, :current_state.shape[1]]

    predicted_chlorine = prediction[
        :,
        :N_CHLORINE,
    ]

    predicted_nodes = predicted_chlorine[
        :,
        :N_NODES,
    ]

    predicted_links = predicted_chlorine[
        :,
        N_NODES:,
    ]

    true_nodes = true_next_chlorine[
        :,
        :N_NODES,
    ]

    true_links = true_next_chlorine[
        :,
        N_NODES:,
    ]

    print("\n" + "=" * 70)
    print(
        f"{NETWORK_NAME.upper()} - {split.upper()}"
    )
    print("=" * 70)

    print(
        "Current state shape:",
        current_state.shape,
    )

    print(
        "Control shape:",
        control.shape,
    )

    print(
        "Prediction shape:",
        prediction.shape,
    )

    print(
        "Prediction finite:",
        bool(np.all(np.isfinite(prediction))),
    )

    print(
        "Predicted chlorine min/max:",
        float(predicted_chlorine.min()),
        float(predicted_chlorine.max()),
    )

    print(
        "True chlorine min/max:",
        float(true_next_chlorine.min()),
        float(true_next_chlorine.max()),
    )

    print_metrics(
        "Node chlorine",
        true_nodes,
        predicted_nodes,
    )

    print_metrics(
        "Link chlorine",
        true_links,
        predicted_links,
    )

    print_metrics(
        "All chlorine states",
        true_next_chlorine,
        predicted_chlorine,
    )

    print_metrics(
        "Persistence baseline: all chlorine states",
        true_next_chlorine,
        persistence_prediction,
    )

    print_metrics(
        "Persistence baseline: node chlorine",
        true_nodes,
        persistence_prediction[:, :N_NODES],
    )

    print_metrics(
        "Persistence baseline: link chlorine",
        true_links,
        persistence_prediction[:, N_NODES:],
    )

    # Für unsere Regelungsaufgabe besonders relevanter Bereich.
    safety_relevant_mask = (
        (true_next_chlorine >= 0.3)
        & (true_next_chlorine <= 2.0)
    )

    safety_error = (
        predicted_chlorine[safety_relevant_mask]
        - true_next_chlorine[safety_relevant_mask]
    )

    print("\nControl-relevant concentration range [0.3, 2.0] mg/L:")
    print(
        "  Anzahl Werte:",
        safety_error.size,
    )

    if safety_error.size > 0:
        print(
            "  MAE:",
            float(
                np.mean(
                    np.abs(safety_error)
                )
            ),
        )

        print(
            "  RMSE:",
            float(
                np.sqrt(
                    np.mean(
                        safety_error ** 2
                    )
                )
            ),
        )
    else:
        print(
            "  Keine Werte in diesem Bereich."
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    evaluate_dataset("validation")
    evaluate_dataset("test")
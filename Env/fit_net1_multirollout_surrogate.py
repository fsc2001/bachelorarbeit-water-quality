"""
This module trains the Net1 surrogate model using multiple training rollouts.
"""
import os
import sys
from pathlib import Path

import numpy as np
from epyt_flow.simulation import ScadaData


PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_REPO = PROJECT_DIR / "NeuralSurrogateKalmanChlorineEstimation"
DATA_DIR = REFERENCE_REPO / "data"

os.chdir(REFERENCE_REPO)
sys.path.insert(0, str(REFERENCE_REPO))

from fit_surrogates import get_mlp_state_transition_model


N_ROLLOUTS = 5
OUTPUT_PATH = DATA_DIR / "net1_randDemand=True_surrogate_multi5.pt"


def load_rollout(rollout_index):
    file_stem = f"net1_randDemand=True_training_rollout{rollout_index}"

    scada_path = DATA_DIR / f"{file_stem}.epytflow_scada_data"
    actions_path = DATA_DIR / f"{file_stem}.npz"

    if not scada_path.exists():
        raise FileNotFoundError(f"Missing SCADA file: {scada_path}")

    if not actions_path.exists():
        raise FileNotFoundError(f"Missing actions file: {actions_path}")

    scada_data = ScadaData.load_from_file(str(scada_path))

    control_actions = np.load(
        actions_path,
        allow_pickle=False,
    )["control_actions"]

    flows = np.asarray(scada_data.get_data_flows())
    node_quality = np.asarray(scada_data.get_data_nodes_quality())
    link_quality = np.asarray(scada_data.get_data_links_quality())

    n_time_steps = flows.shape[0]

    current_state = np.concatenate(
        (
            node_quality[:-1],
            link_quality[:-1],
        ),
        axis=1,
    )

    flow_action_inputs = np.concatenate(
        (
            flows[1:],
            control_actions[:n_time_steps - 1],
        ),
        axis=1,
    )

    next_state = np.concatenate(
        (
            node_quality[1:],
            link_quality[1:],
        ),
        axis=1,
    )

    return current_state, flow_action_inputs, next_state


def main():
    current_state_batches = []
    input_batches = []
    next_state_batches = []

    for rollout_index in range(N_ROLLOUTS):
        current_state, flow_action_inputs, next_state = load_rollout(
            rollout_index
        )

        current_state_batches.append(current_state)
        input_batches.append(flow_action_inputs)
        next_state_batches.append(next_state)

        print(f"Loaded rollout {rollout_index + 1}/{N_ROLLOUTS}")

    current_states = np.concatenate(current_state_batches, axis=0)
    flow_action_inputs = np.concatenate(input_batches, axis=0)
    next_states = np.concatenate(next_state_batches, axis=0)

    if not (
        np.all(np.isfinite(current_states))
        and np.all(np.isfinite(flow_action_inputs))
        and np.all(np.isfinite(next_states))
    ):
        raise RuntimeError("Training data contains non-finite values.")

    model = get_mlp_state_transition_model("Net1")

    first_rollout_path = (
        DATA_DIR
        / "net1_randDemand=True_training_rollout0.epytflow_scada_data"
    )
    first_scada = ScadaData.load_from_file(str(first_rollout_path))

    input_size = current_states.shape[1] + flow_action_inputs.shape[1]
    state_size = next_states.shape[1]

    model.init(
        first_scada.network_topo,
        input_size,
        state_size,
    )

    print("Training Net1 surrogate...")

    model.fit(
        current_states,
        flow_action_inputs,
        next_states,
    )

    model.save_to_file(str(OUTPUT_PATH))

    print(f"Saved model to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
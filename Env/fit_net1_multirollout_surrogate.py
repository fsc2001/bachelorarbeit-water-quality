import os
import sys
from pathlib import Path

import numpy as np
from epyt_flow.simulation import ScadaData


ROOT = Path(__file__).resolve().parents[1]

REPO = (
    ROOT
    / "NeuralSurrogateKalmanChlorineEstimation"
)

DATA_DIR = REPO / "data"

os.chdir(REPO)
sys.path.insert(0, str(REPO))

from fit_surrogates import get_mlp_state_transition_model


N_ROLLOUTS = 5

OUTPUT_PATH = (
    DATA_DIR
    / "net1_randDemand=True_surrogate_multi5.pt"
)


def load_rollout(rollout_idx: int):
    stem = (
        "net1_randDemand=True_"
        f"training_rollout{rollout_idx}"
    )

    scada_path = (
        DATA_DIR
        / f"{stem}.epytflow_scada_data"
    )

    actions_path = (
        DATA_DIR
        / f"{stem}.npz"
    )

    if not scada_path.exists():
        raise FileNotFoundError(
            f"Missing SCADA file: {scada_path}"
        )

    if not actions_path.exists():
        raise FileNotFoundError(
            f"Missing actions file: {actions_path}"
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

    flows = np.asarray(
        scada.get_data_flows()
    )

    nodes_quality = np.asarray(
        scada.get_data_nodes_quality()
    )

    links_quality = np.asarray(
        scada.get_data_links_quality()
    )

    n_time_steps = flows.shape[0]

    # Exakt wie WaterQualityStateTransitionSurrogate.fit_to_scada
    cur_state = np.concatenate(
        (
            nodes_quality[:n_time_steps - 1],
            links_quality[:n_time_steps - 1],
        ),
        axis=1,
    )

    next_time_varying_quantity = np.concatenate(
        (
            flows[1:],
            actions[:n_time_steps - 1],
        ),
        axis=1,
    )

    next_state = np.concatenate(
        (
            nodes_quality[1:],
            links_quality[1:],
        ),
        axis=1,
    )

    return (
        cur_state,
        next_time_varying_quantity,
        next_state,
    )


def main() -> None:
    all_cur_state = []
    all_next_quantities = []
    all_next_state = []

    for rollout_idx in range(N_ROLLOUTS):
        (
            cur_state,
            next_quantities,
            next_state,
        ) = load_rollout(
            rollout_idx
        )

        print(
            f"Rollout {rollout_idx}: "
            f"{cur_state.shape[0]} transitions"
        )

        all_cur_state.append(
            cur_state
        )

        all_next_quantities.append(
            next_quantities
        )

        all_next_state.append(
            next_state
        )

    # Erst fertige Übergänge zusammenführen.
    cur_state = np.concatenate(
        all_cur_state,
        axis=0,
    )

    next_quantities = np.concatenate(
        all_next_quantities,
        axis=0,
    )

    next_state = np.concatenate(
        all_next_state,
        axis=0,
    )

    print("\nCombined training data:")
    print(
        "cur_state:",
        cur_state.shape,
    )
    print(
        "next quantities:",
        next_quantities.shape,
    )
    print(
        "next_state:",
        next_state.shape,
    )

    if not (
        np.all(np.isfinite(cur_state))
        and np.all(np.isfinite(next_quantities))
        and np.all(np.isfinite(next_state))
    ):
        raise RuntimeError(
            "Training data contains non-finite values."
        )

    model = get_mlp_state_transition_model(
        "Net1"
    )

    # Das DNN muss vor dem direkten fit()-Aufruf
    # initialisiert werden. Normalerweise geschieht
    # dies über WaterQualityStateTransitionSurrogate.
    first_scada = ScadaData.load_from_file(
        str(
            DATA_DIR
            / (
                "net1_randDemand=True_"
                "training_rollout0.epytflow_scada_data"
            )
        )
    )

    input_size = (
            cur_state.shape[1]
            + next_quantities.shape[1]
    )

    state_size = next_state.shape[1]

    print(
        "Initializing model:",
        f"input_size={input_size},",
        f"state_size={state_size}",
    )

    model.init(
        first_scada.network_topo,
        input_size,
        state_size,
    )

    print("\nTraining Net1 multi-rollout surrogate ...")

    model.fit(
        cur_state,
        next_quantities,
        next_state,
    )

    model.save_to_file(
        str(OUTPUT_PATH)
    )

    print("\nTraining finished.")
    print("Saved model:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
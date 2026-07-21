"""
This module fits the surrogate models for predicting the chlorine concentration states.
"""
import os
import numpy as np
from epyt_flow.simulation import ScadaData
from epyt_control.signal_processing.state_forecasting import DnnStateTransitionModel, \
    WaterQualityStateTransitionSurrogate


def get_mlp_state_transition_model(net_desc: str):
    if net_desc == "Net1":
        return DnnStateTransitionModel(hidden_layers_size=[512, 512, 128],
                                       activation="relu", last_layer_activation=None,
                                       max_iter=2000, normalization_layer=False,
                                       normalize_input_output=True,
                                       dropout=.01, batch_size=1024)
    elif net_desc == "Hanoi":
        return DnnStateTransitionModel(hidden_layers_size=[1028, 512, 128],
                                       activation="relu", last_layer_activation=None,
                                       max_iter=2000, normalization_layer=False,
                                       normalize_input_output=True, dropout=.05,
                                       batch_size=1024)
    elif net_desc == "CY-DBP":
        return DnnStateTransitionModel(hidden_layers_size=[1028, 512, 128],
                                       activation="relu", last_layer_activation=None,
                                       max_iter=2000, normalization_layer=False,
                                       normalize_input_output=True, dropout=.05,
                                       batch_size=1024)
    else:
        raise ValueError(f"Unknown network '{net_desc}'")


def fit_surrogate(net_desc: str, scada_file_in: str, control_actions_file_in: str, file_out: str):
    scada_data = ScadaData.load_from_file(scada_file_in)
    control_actions = np.load(control_actions_file_in)["control_actions"]

    mlp_state_transition_model = get_mlp_state_transition_model(net_desc)
    surrogate = WaterQualityStateTransitionSurrogate(scada_data.network_topo,
                                                     n_actuators=1 if net_desc != "CY-DBP" else 1,  # TODO: Change if old CY-DBP is used
                                                     state_transition_model=mlp_state_transition_model)

    surrogate.fit_to_scada(scada_data, control_actions=control_actions)
    mlp_state_transition_model.save_to_file(file_out)


if __name__ == "__main__":
    fit_surrogate(
        net_desc="Hanoi",
        scada_file_in=os.path.join(
            "data",
            "hanoi_randDemand=True_training.epytflow_scada_data"
        ),
        control_actions_file_in=os.path.join(
            "data",
            "hanoi_randDemand=True_training.npz"
        ),
        file_out=os.path.join(
            "data",
            "hanoi_randDemand=True_surrogate.pt"
        )
    )

# if __name__ == "__main__":
#     # CY-DBP
#     fit_surrogate(net_desc="CY-DBP",
#                   scada_file_in=os.path.join("data", "cydbp_randDemand=False_training.epytflow_scada_data"),
#                   control_actions_file_in=os.path.join("data", "cydbp_randDemand=False_training.npz"),
#                   file_out=os.path.join("data", "cydbp_randDemand=False_surrogate.pt"))
#
#     fit_surrogate(net_desc="CY-DBP",
#                   scada_file_in=os.path.join("data", "cydbp_randDemand=True_training.epytflow_scada_data"),
#                   control_actions_file_in=os.path.join("data", "cydbp_randDemand=True_training.npz"),
#                   file_out=os.path.join("data", "cydbp_randDemand=True_surrogate.pt"))
#
#     # Net1
#     # fit_surrogate(net_desc="Net1",
#     #               scada_file_in=os.path.join("data", "net1_randDemand=False_training.epytflow_scada_data"),
#     #               control_actions_file_in=os.path.join("data", "net1_randDemand=False_training.npz"),
#     #               file_out=os.path.join("data", "net1_randDemand=False_surrogate.pt"))
#     #
#     # fit_surrogate(net_desc="Net1",
#     #               scada_file_in=os.path.join("data", "net1_randDemand=True_training.epytflow_scada_data"),
#     #               control_actions_file_in=os.path.join("data", "net1_randDemand=True_training.npz"),
#     #               file_out=os.path.join("data", "net1_randDemand=True_surrogate.pt"))
#     # Hanoi
#     fit_surrogate(net_desc="Hanoi",
#                   scada_file_in=os.path.join("data", "hanoi_randDemand=False_training.epytflow_scada_data"),
#                   control_actions_file_in=os.path.join("data", "hanoi_randDemand=False_training.npz"),
#                   file_out=os.path.join("data", "hanoi_randDemand=False_surrogate.pt"))
#     fit_surrogate(net_desc="Hanoi",
#                   scada_file_in=os.path.join("data", "hanoi_randDemand=True_training.epytflow_scada_data"),
#                   control_actions_file_in=os.path.join("data", "hanoi_randDemand=True_training.npz"),
#                   file_out=os.path.join("data", "hanoi_randDemand=True_surrogate.pt"))


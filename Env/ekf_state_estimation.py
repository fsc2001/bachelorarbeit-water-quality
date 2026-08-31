"""
This module uses an EKF for chlorine state estimation with fixed sensor placements.
"""

import sys
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_REPO = PROJECT_DIR / "NeuralSurrogateKalmanChlorineEstimation"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REFERENCE_REPO))

from epyt_flow.simulation import ScadaData
from epyt_control.signal_processing.state_estimation import (
    TimeVaryingExtendedKalmanFilter,
)
from run_exp_state_estimation import get_state_transition_model


def create_measurement_matrix(
    node_indices,
    link_indices,
    n_nodes,
    n_links,
    state_dim,
):
    node_indices = sorted(node_indices)
    link_indices = sorted(link_indices)

    observation_dim = len(node_indices) + 2 * len(link_indices)
    measurement_matrix = np.zeros((observation_dim, state_dim))

    measured_flow_indices = []
    row = 0

    for node_index in node_indices:
        measurement_matrix[row, node_index] = 1
        row += 1

    for link_index in link_indices:
        measurement_matrix[row, n_nodes + link_index] = 1
        row += 1

    for link_index in link_indices:
        flow_index = n_nodes + n_links + link_index
        measured_flow_indices.append(flow_index)

        measurement_matrix[row, flow_index] = 1
        row += 1

    return measurement_matrix, measured_flow_indices


def inverse_scale_state(state, scaler, n_controls):
    state_with_controls = np.concatenate(
        (
            state.reshape(1, -1),
            np.zeros((1, n_controls)),
        ),
        axis=1,
    )

    return scaler.inverse_transform(state_with_controls).flatten()[:state.size]


def run_state_estimation(
    net_desc,
    scada_file_in,
    control_actions_file_in,
    state_transition_model_file_in,
    node_indices,
    link_indices,
):
    scada_data = ScadaData.load_from_file(scada_file_in)

    control_actions = np.load(
        control_actions_file_in
    )["control_actions"]

    flows = scada_data.get_data_flows()
    node_quality = scada_data.get_data_nodes_quality()
    link_quality = scada_data.get_data_links_quality()

    n_time_steps = flows.shape[0]

    current_node_quality = node_quality[:-1]
    current_link_quality = link_quality[:-1]
    next_flows = flows[1:]

    next_states = np.concatenate(
        (
            node_quality[1:],
            link_quality[1:],
        ),
        axis=1,
    )

    current_states = np.concatenate(
        (
            current_node_quality,
            current_link_quality,
            next_flows,
        ),
        axis=1,
    )

    control_actions = control_actions[:n_time_steps - 1]

    state_dim = current_states.shape[1]
    n_chlorine_values = next_states.shape[1]

    n_nodes = current_node_quality.shape[1]
    n_links = current_link_quality.shape[1]

    state_transition_model = get_state_transition_model(
        net_desc,
        state_transition_model_file_in,
    )

    state_transition_model.n_missing_flows = next_states.shape[1]
    state_transition_model._normalize_input_output = False

    states_with_controls = np.concatenate(
        (
            current_states,
            control_actions,
        ),
        axis=1,
    )

    current_states = state_transition_model._scaler.transform(
        states_with_controls
    )[:, :state_dim]

    measurement_matrix, measured_flow_indices = create_measurement_matrix(
        node_indices=node_indices,
        link_indices=link_indices,
        n_nodes=n_nodes,
        n_links=n_links,
        state_dim=state_dim,
    )

    observation_dim = measurement_matrix.shape[0]

    def measurement_function(state):
        return measurement_matrix @ state.flatten()

    def measurement_jacobian(_):
        return measurement_matrix

    def get_measurement_func(_):
        return measurement_function

    def get_measurement_func_grad(_):
        return measurement_jacobian

    def get_control_signal(time_step):
        return control_actions[time_step + 1].reshape(1, -1)

    def get_state_transition_func(time_step):
        control_signal = get_control_signal(time_step)

        def predict(state):
            return state_transition_model.predict(
                state.reshape(1, -1),
                control_signal,
            ).flatten()

        return predict

    def get_state_transition_func_grad(time_step):
        control_signal = get_control_signal(time_step)

        def get_jacobian(current_state):
            jacobian = state_transition_model.compute_jacobian(
                current_state.reshape(1, -1),
                control_signal,
            )

            jacobian = jacobian.reshape(
                jacobian.shape[1],
                jacobian.shape[3],
            )

            return jacobian[:, :state_dim]

        return get_jacobian

    ekf = TimeVaryingExtendedKalmanFilter(
        state_dim=state_dim,
        obs_dim=observation_dim,
        init_state=current_states[0],
        get_state_transition_func=get_state_transition_func,
        get_state_transition_func_grad=get_state_transition_func_grad,
        get_measurement_func=get_measurement_func,
        get_measurement_func_grad=get_measurement_func_grad,
    )

    chlorine_predictions = []
    chlorine_true = []

    for time_step in range(1, current_states.shape[0]):
        current_state = current_states[time_step]

        # Measured flows are inserted directly, following the reference implementation.
        for flow_index in measured_flow_indices:
            ekf._x[flow_index] = current_state[flow_index]

        observation = measurement_function(current_state)

        predicted_state, _ = ekf.step(observation)

        predicted_state = inverse_scale_state(
            predicted_state,
            state_transition_model._scaler,
            control_actions.shape[1],
        )

        true_state = inverse_scale_state(
            current_state,
            state_transition_model._scaler,
            control_actions.shape[1],
        )

        chlorine_predictions.append(
            predicted_state[:n_chlorine_values].reshape(1, -1)
        )

        chlorine_true.append(
            true_state[:n_chlorine_values].reshape(1, -1)
        )

    return chlorine_predictions, chlorine_true
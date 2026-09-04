"""
This module performs EKF-based chlorine state estimation
for fixed sensor locations.
"""

import sys
from pathlib import Path

import numpy as np
from epyt_control.signal_processing.state_estimation import (
    TimeVaryingExtendedKalmanFilter,
)
from epyt_flow.simulation import ScadaData


PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_REPO = PROJECT_DIR / "NeuralSurrogateKalmanChlorineEstimation"

sys.path.insert(0, str(REFERENCE_REPO))

from run_exp_state_estimation import get_state_transition_model


def create_measurement_matrix(
    node_indices,
    link_indices,
    n_nodes,
    n_links,
    state_size,
):
    node_indices = sorted(node_indices)
    link_indices = sorted(link_indices)

    n_measurements = len(node_indices) + 2 * len(link_indices)
    measurement_matrix = np.zeros((n_measurements, state_size))

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

        measurement_matrix[row, flow_index] = 1
        measured_flow_indices.append(flow_index)

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

    state = scaler.inverse_transform(
        state_with_controls
    )

    return state.flatten()[:state_with_controls.shape[1] - n_controls]


def run_state_estimation(
    network_name,
    scada_path,
    actions_path,
    surrogate_path,
    node_indices,
    link_indices,
):
    scada_data = ScadaData.load_from_file(scada_path)

    control_actions = np.load(
        actions_path
    )["control_actions"]

    node_quality = scada_data.get_data_nodes_quality()
    link_quality = scada_data.get_data_links_quality()
    flows = scada_data.get_data_flows()

    current_states = np.concatenate(
        (
            node_quality[:-1],
            link_quality[:-1],
            flows[1:],
        ),
        axis=1,
    )

    next_chlorine = np.concatenate(
        (
            node_quality[1:],
            link_quality[1:],
        ),
        axis=1,
    )

    control_actions = control_actions[:current_states.shape[0]]

    n_nodes = node_quality.shape[1]
    n_links = link_quality.shape[1]
    state_size = current_states.shape[1]
    n_chlorine_values = next_chlorine.shape[1]

    surrogate = get_state_transition_model(
        network_name,
        surrogate_path,
    )

    surrogate.n_missing_flows = n_chlorine_values
    surrogate._normalize_input_output = False

    states_with_controls = np.concatenate(
        (
            current_states,
            control_actions,
        ),
        axis=1,
    )

    scaled_states = surrogate._scaler.transform(
        states_with_controls
    )[:, :state_size]

    measurement_matrix, measured_flow_indices = (
        create_measurement_matrix(
            node_indices=node_indices,
            link_indices=link_indices,
            n_nodes=n_nodes,
            n_links=n_links,
            state_size=state_size,
        )
    )

    def measurement_function(state):
        return measurement_matrix @ state.flatten()

    def measurement_jacobian(_):
        return measurement_matrix

    def get_prediction_function(time_step):
        control = control_actions[
            time_step + 1
        ].reshape(1, -1)

        def predict(state):
            return surrogate.predict(
                state.reshape(1, -1),
                control,
            ).flatten()

        return predict

    def get_prediction_jacobian(time_step):
        control = control_actions[
            time_step + 1
        ].reshape(1, -1)

        def jacobian(state):
            result = surrogate.compute_jacobian(
                state.reshape(1, -1),
                control,
            )

            result = result.reshape(
                result.shape[1],
                result.shape[3],
            )

            return result[:, :state_size]

        return jacobian

    ekf = TimeVaryingExtendedKalmanFilter(
        state_dim=state_size,
        obs_dim=measurement_matrix.shape[0],
        init_state=scaled_states[0],
        get_state_transition_func=get_prediction_function,
        get_state_transition_func_grad=get_prediction_jacobian,
        get_measurement_func=lambda _: measurement_function,
        get_measurement_func_grad=lambda _: measurement_jacobian,
    )

    chlorine_predictions = []
    chlorine_true = []

    for time_step in range(1, scaled_states.shape[0]):
        true_state = scaled_states[time_step]

        # Insert measured flows directly as in the reference implementation.
        for flow_index in measured_flow_indices:
            ekf._x[flow_index] = true_state[flow_index]

        measurement = measurement_function(true_state)

        predicted_state, _ = ekf.step(
            measurement
        )

        predicted_state = inverse_scale_state(
            predicted_state,
            surrogate._scaler,
            control_actions.shape[1],
        )

        true_state = inverse_scale_state(
            true_state,
            surrogate._scaler,
            control_actions.shape[1],
        )

        chlorine_predictions.append(
            predicted_state[:n_chlorine_values].reshape(1, -1)
        )

        chlorine_true.append(
            true_state[:n_chlorine_values].reshape(1, -1)
        )

    return chlorine_predictions, chlorine_true
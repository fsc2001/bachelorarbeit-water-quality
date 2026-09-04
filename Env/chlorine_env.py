"""
This module defines the chlorine control environment used for PPO training.
"""

import sys
from pathlib import Path

import numpy as np
from epyt_control.envs import EpanetControlEnv
from epyt_control.envs.actions import ChemicalInjectionAction
from epyt_control.signal_processing.state_estimation import (
    TimeVaryingExtendedKalmanFilter,
)
from epyt_flow.simulation import EpanetConstants, ScenarioConfig, ScadaData
from epyt_flow.uncertainty import ModelUncertainty


PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_REPO = PROJECT_DIR / "NeuralSurrogateKalmanChlorineEstimation"
DATA_DIR = REFERENCE_REPO / "data"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REFERENCE_REPO))

from run_exp_state_estimation import get_state_transition_model


LOWER_CHLORINE_BOUND = 0.3
UPPER_CHLORINE_BOUND = 2.0

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
        link_quality_index = n_nodes + link_index
        measurement_matrix[row, link_quality_index] = 1
        row += 1

    for link_index in link_indices:
        flow_index = n_nodes + n_links + link_index

        measurement_matrix[row, flow_index] = 1
        measured_flow_indices.append(flow_index)
        row += 1

    return measurement_matrix, measured_flow_indices


def create_network_scenario(
    network_config,
    split=None,
    uncertainty_seed=None,
):
    scenario_name = network_config.get_scenario_stem(split)

    config = ScenarioConfig.load_from_file(
        DATA_DIR / f"{scenario_name}.epytflow_scenario_config"
    )

    config._ScenarioConfig__f_inp_in = str(
        DATA_DIR / f"{scenario_name}.inp"
    )

    if uncertainty_seed is not None and config.model_uncertainty is not None:
        uncertainty_settings = config.model_uncertainty.get_attributes()
        uncertainty_settings["seed"] = int(uncertainty_seed)

        config = ScenarioConfig(
            scenario_config=config,
            model_uncertainty=ModelUncertainty(**uncertainty_settings),
        )

    return config


class ChlorineControlEnv(EpanetControlEnv):
    def __init__(
        self,
        scenario_config,
        network_config,
        use_estimated_state=True,
        chlorine_penalty_weight=0.01,
        sensor_node_indices=None,
        sensor_link_indices=None,
    ):
        self._network = network_config
        self._use_estimated_state = use_estimated_state
        self._chlorine_penalty_weight = chlorine_penalty_weight

        if (sensor_node_indices is None) != (sensor_link_indices is None):
            raise ValueError(
                "Node and link sensor indices must both be provided."
            )

        if use_estimated_state and sensor_node_indices is None:
            raise ValueError(
                "Sensor indices are required when using EKF state estimation."
            )

        self._sensor_node_indices = sensor_node_indices
        self._sensor_link_indices = sensor_link_indices

        self._ekf = None
        self._surrogate = None
        self._state_size = None
        self._measurement_matrix = None
        self._measured_flow_indices = None

        self._last_action = np.zeros((1, 1), dtype=float)

        super().__init__(
            scenario_config=scenario_config,
            chemical_injection_actions=[
                ChemicalInjectionAction(
                    node_id=self._network.injection_node_id,
                    pattern_id=self._network.injection_pattern_id,
                    source_type_id=EpanetConstants.EN_CONCEN,
                    upper_bound=5.0,
                )
            ],
            autoreset=True,
            reload_scenario_when_reset=False,
        )

        self.observation_space.low[self._network.demand_slice] = -np.inf

    def _extract_state(self, observation):
        node_chlorine = observation[self._network.node_chlorine_slice]
        link_chlorine = observation[self._network.link_chlorine_slice]
        flows = observation[self._network.flow_slice]

        return np.concatenate(
            (node_chlorine, link_chlorine, flows)
        )

    def _scale_state(self, state):
        state_with_action = np.concatenate(
            (
                state.reshape(1, -1),
                self._last_action,
            ),
            axis=1,
        )

        return self._surrogate._scaler.transform(
            state_with_action
        )[0, :self._state_size]

    def _inverse_scale_state(self, state):
        state_with_action = np.concatenate(
            (
                state.reshape(1, -1),
                np.zeros((1, 1)),
            ),
            axis=1,
        )

        return self._surrogate._scaler.inverse_transform(
            state_with_action
        )[0, :self._state_size]

    def _prepare_observation(self, observation):
        observation = np.asarray(
            observation,
            dtype=np.float32,
        ).copy()

        chlorine_slice = self._network.chlorine_slice

        observation[chlorine_slice] = np.maximum(
            observation[chlorine_slice],
            0.0,
        )

        return observation

    def _build_agent_observation(
        self,
        simulator_observation,
        estimated_state,
    ):
        pressures = simulator_observation[self._network.pressure_slice]
        demands = simulator_observation[self._network.demand_slice]

        node_chlorine = estimated_state[
            self._network.state_node_chlorine_slice
        ]

        link_chlorine = estimated_state[
            self._network.state_link_chlorine_slice
        ]

        flows = estimated_state[
            self._network.state_flow_slice
        ]

        observation = np.concatenate(
            (
                pressures,
                flows,
                demands,
                node_chlorine,
                link_chlorine,
            )
        )

        return self._prepare_observation(observation)

    def _initialize_ekf(self, observation):
        self._surrogate = get_state_transition_model(
            self._network.model_name,
            str(DATA_DIR / self._network.surrogate_filename),
        )

        self._surrogate.n_missing_flows = (
            self._network.n_nodes + self._network.n_links
        )

        self._surrogate._normalize_input_output = False

        initial_state = self._extract_state(observation)

        self._state_size = initial_state.size

        scaled_initial_state = self._scale_state(initial_state)

        (
            self._measurement_matrix,
            self._measured_flow_indices,
        ) = create_measurement_matrix(
            node_indices=self._sensor_node_indices,
            link_indices=self._sensor_link_indices,
            n_nodes=self._network.n_nodes,
            n_links=self._network.n_links,
            state_size=self._state_size,
        )

        def measurement_function(state):
            return self._measurement_matrix @ state.flatten()

        def measurement_jacobian(_):
            return self._measurement_matrix

        def get_prediction_function(_):
            control_action = self._last_action.copy()

            def predict(state):
                return self._surrogate.predict(
                    state.reshape(1, -1),
                    control_action,
                ).flatten()

            return predict

        def get_prediction_jacobian(_):
            control_action = self._last_action.copy()

            def jacobian(state):
                result = self._surrogate.compute_jacobian(
                    state.reshape(1, -1),
                    control_action,
                )

                result = result.reshape(
                    result.shape[1],
                    result.shape[3],
                )

                return result[:, :self._state_size]

            return jacobian

        self._ekf = TimeVaryingExtendedKalmanFilter(
            state_dim=self._state_size,
            obs_dim=self._measurement_matrix.shape[0],
            init_state=scaled_initial_state,
            get_state_transition_func=get_prediction_function,
            get_state_transition_func_grad=get_prediction_jacobian,
            get_measurement_func=lambda _: measurement_function,
            get_measurement_func_grad=lambda _: measurement_jacobian,
        )

    def _estimate_state(self, observation):
        state = self._extract_state(observation)
        scaled_state = self._scale_state(state)

        # flows are inserted directly, as in the reference implementation.
        for flow_index in self._measured_flow_indices:
            self._ekf._x[flow_index] = scaled_state[flow_index]

        sensor_measurement = (
            self._measurement_matrix @ scaled_state
        )

        estimated_state, _ = self._ekf.step(
            sensor_measurement
        )

        return self._inverse_scale_state(estimated_state)

    def reset(self, **kwargs):
        self._last_action = np.zeros((1, 1), dtype=float)

        observation, info = super().reset(**kwargs)

        if not self._use_estimated_state:
            return self._prepare_observation(observation), info

        self._initialize_ekf(observation)

        initial_estimate = self._inverse_scale_state(
            self._ekf._x
        )

        agent_observation = self._build_agent_observation(
            observation,
            initial_estimate,
        )

        return agent_observation, info

    def step(self, action):
        self._last_action = np.asarray(
            action,
            dtype=float,
        ).reshape(1, -1)

        observation, reward, terminated, truncated, info = (
            super().step(action)
        )

        info["physical_chlorine_action"] = float(
            self._last_action[0, 0]
        )

        if self._use_estimated_state:
            estimated_state = self._estimate_state(observation)

            agent_observation = self._build_agent_observation(
                observation,
                estimated_state,
            )
        else:
            agent_observation = self._prepare_observation(
                observation
            )

        return (
            agent_observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def _compute_reward_function(self, scada_data):
        node_chlorine = np.asarray(
            scada_data.get_data_nodes_quality(),
            dtype=float,
        )

        lower_violation = np.maximum(
            LOWER_CHLORINE_BOUND - node_chlorine,
            0.0,
        )

        upper_violation = np.maximum(
            node_chlorine - UPPER_CHLORINE_BOUND,
            0.0,
        )

        violation_penalty = np.sum(
            lower_violation + upper_violation
        )

        chlorine_action = float(self._last_action[0, 0])

        chlorine_penalty = (
            self._chlorine_penalty_weight
            * chlorine_action
        )

        return -float(
            violation_penalty + chlorine_penalty
        )

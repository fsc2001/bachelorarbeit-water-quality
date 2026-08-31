import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from epyt_control.envs import EpanetControlEnv
from epyt_control.envs.actions import ChemicalInjectionAction
from epyt_control.signal_processing.state_estimation import (
    TimeVaryingExtendedKalmanFilter,
)
from epyt_flow.simulation import EpanetConstants, ScenarioConfig, ScadaData
from epyt_flow.uncertainty import ModelUncertainty


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_REPO = ROOT / "NeuralSurrogateKalmanChlorineEstimation"
DATA_PATH = REFERENCE_REPO / "data"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REFERENCE_REPO))

from Env.network_config import NETWORKS, NetworkConfig
from run_exp_state_estimation import (
    create_random_sensor_placement,
    get_state_transition_model,
)


SENSOR_SEED = 0


def create_fixed_sensor_placement(
    node_indices,
    link_indices,
    n_nodes,
    n_links,
    state_dim,
):
    node_indices = sorted(node_indices)
    link_indices = sorted(link_indices)

    if len(set(node_indices)) != len(node_indices):
        raise ValueError("Duplicate node sensor indices.")

    if len(set(link_indices)) != len(link_indices):
        raise ValueError("Duplicate link sensor indices.")

    if not all(0 <= idx < n_nodes for idx in node_indices):
        raise ValueError("Invalid node sensor index.")

    if not all(0 <= idx < n_links for idx in link_indices):
        raise ValueError("Invalid link sensor index.")

    measurement_matrix = np.zeros(
        (
            len(node_indices) + 2 * len(link_indices),
            state_dim,
        )
    )

    flow_indices = []
    row = 0

    for idx in node_indices:
        measurement_matrix[row, idx] = 1
        row += 1

    for idx in link_indices:
        measurement_matrix[row, n_nodes + idx] = 1
        row += 1

    for idx in link_indices:
        flow_idx = n_nodes + n_links + idx
        flow_indices.append(flow_idx)
        measurement_matrix[row, flow_idx] = 1
        row += 1

    return measurement_matrix, flow_indices


def create_network_scenario(
    network_config: NetworkConfig,
    n_sensors: int = 3,
    split: Optional[str] = None,
    uncertainty_seed: Optional[int] = None,
):
    scenario_stem = network_config.get_scenario_stem(split=split)

    config = ScenarioConfig.load_from_file(
        DATA_PATH / f"{scenario_stem}.epytflow_scenario_config"
    )

    config._ScenarioConfig__f_inp_in = str(
        DATA_PATH / f"{scenario_stem}.inp"
    )

    if uncertainty_seed is not None and config.model_uncertainty is not None:
        uncertainty_args = config.model_uncertainty.get_attributes()
        uncertainty_args["seed"] = int(uncertainty_seed)

        config = ScenarioConfig(
            scenario_config=config,
            model_uncertainty=ModelUncertainty(**uncertainty_args),
        )

    return config, n_sensors


def create_hanoi_scenario(n_sensors: int = 3):
    return create_network_scenario(
        network_config=NETWORKS["hanoi"],
        n_sensors=n_sensors,
    )


class ChlorineControlEnv(EpanetControlEnv):
    def __init__(
        self,
        scenario_config: ScenarioConfig,
        network_config: NetworkConfig,
        n_sensors: int = 3,
        use_estimated_state: bool = True,
        chlorine_penalty_weight: float = 0.01,
        sensor_node_indices=None,
        sensor_link_indices=None,
    ):
        self._network = network_config
        self._n_sensors = n_sensors
        self._use_estimated_state = use_estimated_state
        self._chlorine_penalty_weight = chlorine_penalty_weight

        if (sensor_node_indices is None) != (sensor_link_indices is None):
            raise ValueError(
                "Node and link sensor indices must either both be "
                "provided or both be None."
            )

        self._sensor_node_indices = sensor_node_indices
        self._sensor_link_indices = sensor_link_indices

        self._kalman = None
        self._surrogate = None
        self._state_dim = None
        self._M = None
        self._flows_idx = None
        self._last_action = np.array([[0.0]], dtype=np.float64)

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

    def _extract_state(self, obs: np.ndarray) -> np.ndarray:
        node_chlorine = obs[self._network.node_chlorine_slice]
        link_chlorine = obs[self._network.link_chlorine_slice]
        flows = obs[self._network.flow_slice]

        return np.concatenate((node_chlorine, link_chlorine, flows))

    def _scale_state(self, state: np.ndarray) -> np.ndarray:
        scaler_input = np.concatenate(
            (
                state.reshape(1, -1),
                self._last_action.reshape(1, -1),
            ),
            axis=1,
        )

        return self._surrogate._scaler.transform(
            scaler_input
        )[0, :self._state_dim]

    def _build_agent_observation(
        self,
        obs: np.ndarray,
        estimated_state: np.ndarray,
    ) -> np.ndarray:
        pressures = obs[self._network.pressure_slice]
        demands = obs[self._network.demand_slice]

        node_chlorine = estimated_state[
            self._network.state_node_chlorine_slice
        ]
        link_chlorine = estimated_state[
            self._network.state_link_chlorine_slice
        ]
        flows = estimated_state[
            self._network.state_flow_slice
        ]

        agent_obs = np.concatenate(
            (
                pressures,
                flows,
                demands,
                node_chlorine,
                link_chlorine,
            )
        )

        return self._prepare_observation(agent_obs)

    def _init_kalman(self, obs: np.ndarray) -> None:
        self._surrogate = get_state_transition_model(
            self._network.model_name,
            str(DATA_PATH / self._network.surrogate_filename),
        )

        self._surrogate.n_missing_flows = (
            self._network.n_nodes + self._network.n_links
        )
        self._surrogate._normalize_input_output = False

        raw_state = self._extract_state(obs)
        self._state_dim = raw_state.shape[0]

        init_state = self._scale_state(raw_state)

        if self._sensor_node_indices is not None:
            self._M, self._flows_idx = create_fixed_sensor_placement(
                node_indices=self._sensor_node_indices,
                link_indices=self._sensor_link_indices,
                n_nodes=self._network.n_nodes,
                n_links=self._network.n_links,
                state_dim=self._state_dim,
            )
        else:
            python_random_state = random.getstate()
            numpy_random_state = np.random.get_state()

            try:
                random.seed(SENSOR_SEED)
                np.random.seed(SENSOR_SEED)

                self._M, self._flows_idx = create_random_sensor_placement(
                    self._n_sensors,
                    self._n_sensors,
                    self._network.n_nodes,
                    self._network.n_links,
                    self._state_dim,
                )
            finally:
                random.setstate(python_random_state)
                np.random.set_state(numpy_random_state)

        def measurement_func(x: np.ndarray) -> np.ndarray:
            return self._M @ x.flatten()

        def measurement_func_grad(_: np.ndarray) -> np.ndarray:
            return self._M

        def get_control_signal() -> np.ndarray:
            return self._last_action.reshape(1, -1)

        def get_state_transition_func(_t: int):
            control = get_control_signal()

            return lambda x: self._surrogate.predict(
                x.reshape(1, -1),
                control,
            ).flatten()

        def get_state_transition_func_grad(_t: int):
            control = get_control_signal()

            def get_jacobian(current_state: np.ndarray) -> np.ndarray:
                jacobian = self._surrogate.compute_jacobian(
                    current_state.reshape(1, -1),
                    control,
                )

                jacobian = jacobian.reshape(
                    jacobian.shape[1],
                    jacobian.shape[3],
                )

                return jacobian[:, :self._state_dim]

            return get_jacobian

        self._kalman = TimeVaryingExtendedKalmanFilter(
            state_dim=self._state_dim,
            obs_dim=self._M.shape[0],
            init_state=init_state,
            get_state_transition_func=get_state_transition_func,
            get_state_transition_func_grad=get_state_transition_func_grad,
            get_measurement_func=lambda _t: measurement_func,
            get_measurement_func_grad=lambda _t: measurement_func_grad,
        )

    def _prepare_observation(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).copy()
        chlorine_slice = self._network.chlorine_slice
        obs[chlorine_slice] = np.maximum(obs[chlorine_slice], 0.0)
        return obs

    def _inverse_scale_state(
        self,
        scaled_state: np.ndarray,
    ) -> np.ndarray:
        scaler_input = np.concatenate(
            (
                scaled_state.reshape(1, -1),
                np.zeros((1, 1)),
            ),
            axis=1,
        )

        return self._surrogate._scaler.inverse_transform(
            scaler_input
        )[0, :self._state_dim]

    def reset(self, **kwargs):
        self._last_action = np.zeros((1, 1), dtype=np.float64)

        obs, info = super().reset(**kwargs)

        if not self._use_estimated_state:
            return self._prepare_observation(obs), info

        self._init_kalman(obs)

        initial_estimate = self._inverse_scale_state(self._kalman._x)
        agent_obs = self._build_agent_observation(
            obs,
            initial_estimate,
        )

        return agent_obs, info

    def step(self, action: np.ndarray):
        self._last_action = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(1, -1)

        obs, reward, terminated, truncated, info = super().step(action)

        info["physical_chlorine_action"] = float(
            self._last_action.reshape(-1)[0]
        )

        if self._use_estimated_state:
            agent_obs = self._kalman_obs(obs)
        else:
            agent_obs = self._prepare_observation(obs)

        return agent_obs, reward, terminated, truncated, info

    def _kalman_obs(self, obs: np.ndarray) -> np.ndarray:
        raw_state = self._extract_state(obs)
        scaled_state = self._scale_state(raw_state)

        for idx in self._flows_idx:
            self._kalman._x[idx] = scaled_state[idx]

        observation = self._M @ scaled_state
        scaled_estimate, _ = self._kalman.step(observation)

        state_estimate = self._inverse_scale_state(scaled_estimate)

        return self._build_agent_observation(
            obs,
            state_estimate,
        )

    def _compute_reward_function(
        self,
        scada_data: ScadaData,
    ) -> float:
        lower_bound = 0.3
        upper_bound = 2.0

        node_quality = np.asarray(
            scada_data.get_data_nodes_quality(),
            dtype=np.float64,
        )

        lower_violation = np.maximum(
            lower_bound - node_quality,
            0.0,
        )
        upper_violation = np.maximum(
            node_quality - upper_bound,
            0.0,
        )

        violation_penalty = np.sum(
            lower_violation + upper_violation
        )

        chlorine_action = float(
            self._last_action.flatten()[0]
        )

        chlorine_penalty = (
            self._chlorine_penalty_weight
            * chlorine_action
        )

        return -float(
            violation_penalty + chlorine_penalty
        )
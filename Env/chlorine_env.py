import random
from typing import Optional

import numpy as np
import sys
from pathlib import Path
from epyt_flow.uncertainty import ModelUncertainty

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_REPO = (
    ROOT / "NeuralSurrogateKalmanChlorineEstimation"
)

# Für Imports wie: from Env.network_config import ...
sys.path.insert(0, str(ROOT))

# Für Imports aus dem verschachtelten Referenzprojekt,
# z. B. run_exp_state_estimation
sys.path.insert(0, str(REFERENCE_REPO))

DATA_PATH = REFERENCE_REPO / "data"
from Env.network_config import (
    NETWORKS,
    NetworkConfig,
)
from epyt_flow.simulation import EpanetConstants, ScenarioConfig, ScadaData
from epyt_control.envs import EpanetControlEnv
from epyt_control.envs.actions import ChemicalInjectionAction
from epyt_control.signal_processing.state_estimation import TimeVaryingExtendedKalmanFilter
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

    if not all(0 <= i < n_nodes for i in node_indices):
        raise ValueError("Invalid node sensor index.")

    if not all(0 <= i < n_links for i in link_indices):
        raise ValueError("Invalid link sensor index.")

    M = np.zeros(
        (
            len(node_indices)
            + 2 * len(link_indices),
            state_dim,
        )
    )

    flows_idx = []

    i = 0

    for idx in node_indices:
        M[i, idx] = 1
        i += 1

    for idx in link_indices:
        M[i, n_nodes + idx] = 1
        i += 1

    for idx in link_indices:
        flow_idx = (
            n_nodes
            + n_links
            + idx
        )

        flows_idx.append(flow_idx)
        M[i, flow_idx] = 1
        i += 1

    return M, flows_idx

def create_network_scenario(
    network_config: NetworkConfig,
    n_sensors: int = 3,
    split: Optional[str] = None,
    uncertainty_seed: Optional[int] = None,
):
    scenario_stem = (
        network_config.get_scenario_stem(
            split=split
        )
    )

    config = ScenarioConfig.load_from_file(
        DATA_PATH
        / f"{scenario_stem}.epytflow_scenario_config"
    )

    config._ScenarioConfig__f_inp_in = str(
        DATA_PATH
        / f"{scenario_stem}.inp"
    )

    if uncertainty_seed is not None:
        model_uncertainty = (
            config.model_uncertainty
        )

        if model_uncertainty is not None:
            uncertainty_args = (
                model_uncertainty.get_attributes()
            )

            uncertainty_args["seed"] = int(
                uncertainty_seed
            )

            seeded_model_uncertainty = (
                ModelUncertainty(
                    **uncertainty_args
                )
            )

            # ScenarioConfig besitzt dafür keinen
            # öffentlichen Setter. Daher erzeugen
            # wir sauber eine neue Config auf Basis
            # der vorhandenen Config.
            config = ScenarioConfig(
                scenario_config=config,
                model_uncertainty=(
                    seeded_model_uncertainty
                ),
            )

    return config, n_sensors

def create_hanoi_scenario(
    n_sensors: int = 3,
):
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

        if (
                (sensor_node_indices is None)
                != (sensor_link_indices is None)
        ):
            raise ValueError(
                "Node and link sensor indices must "
                "either both be provided or both be None."
            )

        self._sensor_node_indices = sensor_node_indices
        self._sensor_link_indices = sensor_link_indices

        self._sparse_node_indices = None
        self._kalman = None
        self._surrogate = None
        self._last_action = np.array(
            [[0.0]],
            dtype=np.float64,
        )

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

        # Im Hanoi-Netz kann der Demand eines Einspeiseknotens
        # negativ sein.
        self.observation_space.low[
            self._network.demand_slice
        ] = -np.inf


    def _init_kalman(self, obs: np.ndarray) -> None:
        self._surrogate = get_state_transition_model(
            self._network.model_name,
            str(
                DATA_PATH
                / self._network.surrogate_filename
            ),
        )

        self._surrogate.n_missing_flows = (
                self._network.n_nodes
                + self._network.n_links
        )

        self._surrogate._normalize_input_output = False

        cl_nodes = obs[
            self._network.node_chlorine_slice
        ]
        cl_links = obs[
            self._network.link_chlorine_slice
        ]
        flows = obs[
            self._network.flow_slice
        ]

        raw_state = np.concatenate((cl_nodes, cl_links, flows))
        state_dim = raw_state.shape[0]

        self._state_dim = state_dim

        scaler_input = np.concatenate(
            (
                raw_state.reshape(1, -1),
                self._last_action.reshape(1, -1)
            ),
            axis=1
        )

        init_state = self._surrogate._scaler.transform(
            scaler_input
        )[0, :state_dim]

        if self._sensor_node_indices is not None:

            self._M, self._flows_idx = (
                create_fixed_sensor_placement(
                    node_indices=self._sensor_node_indices,
                    link_indices=self._sensor_link_indices,
                    n_nodes=self._network.n_nodes,
                    n_links=self._network.n_links,
                    state_dim=state_dim,
                )
            )

        else:
            # Fallback für alte Experimente:
            # gleiche Random-Logik wie bisher.
            python_random_state = random.getstate()
            numpy_random_state = np.random.get_state()

            try:
                random.seed(SENSOR_SEED)
                np.random.seed(SENSOR_SEED)

                self._M, self._flows_idx = (
                    create_random_sensor_placement(
                        self._n_sensors,
                        self._n_sensors,
                        self._network.n_nodes,
                        self._network.n_links,
                        state_dim,
                    )
                )

            finally:
                random.setstate(python_random_state)
                np.random.set_state(
                    numpy_random_state
                )

        obs_dim = self._M.shape[0]

        def measurement_func(x: np.ndarray) -> np.ndarray:
            return self._M @ x.flatten()

        def measurement_func_grad(_: np.ndarray) -> np.ndarray:
            return self._M

        def get_control_signal() -> np.ndarray:
            return self._last_action.reshape(1, -1)

        def get_state_transition_func(_t: int):
            x_control = get_control_signal()

            return lambda x: self._surrogate.predict(
                x.reshape(1, -1),
                x_control
            ).flatten()

        def get_state_transition_func_grad(_t: int):
            x_control = get_control_signal()

            def get_jac(x_cur_state: np.ndarray) -> np.ndarray:
                jac = self._surrogate.compute_jacobian(
                    x_cur_state.reshape(1, -1),
                    x_control
                )

                # Batch-Dimensionen entfernen.
                jac = jac.reshape(jac.shape[1], jac.shape[3])

                # Ableitungen nach dem Control-Eingang entfernen.
                return jac[:, :state_dim]

            return get_jac

        self._kalman = TimeVaryingExtendedKalmanFilter(
            state_dim=state_dim,
            obs_dim=obs_dim,
            init_state=init_state,
            get_state_transition_func=get_state_transition_func,
            get_state_transition_func_grad=get_state_transition_func_grad,
            get_measurement_func=lambda _t: measurement_func,
            get_measurement_func_grad=lambda _t: measurement_func_grad,
        )

    def _prepare_observation(
            self,
            obs: np.ndarray,
    ) -> np.ndarray:
        obs = np.asarray(
            obs,
            dtype=np.float32,
        ).copy()

        chlorine_slice = self._network.chlorine_slice

        obs[chlorine_slice] = np.maximum(
            obs[chlorine_slice],
            0.0,
        )

        return obs

    def _inverse_scale_state(self, scaled_state: np.ndarray) -> np.ndarray:
        # Der Scaler erwartet wieder 100 Zustände + 1 Control-Wert.
        scaler_input = np.concatenate(
            (
                scaled_state.reshape(1, -1),
                np.zeros((1, 1))
            ),
            axis=1
        )

        return self._surrogate._scaler.inverse_transform(
            scaler_input
        )[0, :self._state_dim]

    def reset(self, **kwargs):
        self._last_action = np.zeros(
            (1, 1),
            dtype=np.float64,
        )

        obs, info = super().reset(**kwargs)

        # Oracle-Variante:
        # Der Agent erhält direkt die vollständige
        # Simulatorbeobachtung.
        if not self._use_estimated_state:
            return self._prepare_observation(obs), info

        # EKF-Variante:
        self._init_kalman(obs)

        initial_estimate = self._inverse_scale_state(
            self._kalman._x
        )

        pressures = obs[
            self._network.pressure_slice
        ]

        demands = obs[
            self._network.demand_slice
        ]

        estimated_node_chlorine = initial_estimate[
            self._network.state_node_chlorine_slice
        ]

        estimated_link_chlorine = initial_estimate[
            self._network.state_link_chlorine_slice
        ]

        estimated_flows = initial_estimate[
            self._network.state_flow_slice
        ]

        new_obs = np.concatenate(
            (
                pressures,
                estimated_flows,
                demands,
                estimated_node_chlorine,
                estimated_link_chlorine,
            )
        )

        return self._prepare_observation(new_obs), info

    def step(self, action: np.ndarray):
        self._last_action = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(1, -1)

        obs, reward, terminated, truncated, info = (
            super().step(action)
        )

        info["physical_chlorine_action"] = float(
            self._last_action.reshape(-1)[0]
        )

        if self._use_estimated_state:
            agent_obs = self._kalman_obs(obs)
        else:
            agent_obs = self._prepare_observation(obs)

        return agent_obs, reward, terminated, truncated, info

    def _kalman_obs(self, obs: np.ndarray) -> np.ndarray:
        cl_nodes = obs[
            self._network.node_chlorine_slice
        ]

        cl_links = obs[
            self._network.link_chlorine_slice
        ]

        flows = obs[
            self._network.flow_slice
        ]

        raw_state = np.concatenate((cl_nodes, cl_links, flows))

        # Wie beim Training und im Originalexperiment skalieren.
        scaler_input = np.concatenate(
            (
                raw_state.reshape(1, -1),
                self._last_action.reshape(1, -1)
            ),
            axis=1
        )

        scaled_state = self._surrogate._scaler.transform(
            scaler_input
        )[0, :self._state_dim]

        # Direkt beobachtete Flows in den Filterzustand einsetzen.
        for idx in self._flows_idx:
            self._kalman._x[idx] = scaled_state[idx]

        # Nur die durch M ausgewählten Messwerte an den EKF geben.
        observation = self._M @ scaled_state

        scaled_estimate, _ = self._kalman.step(observation)

        # EKF arbeitet skaliert; der RL-Agent erhält reale Einheiten.
        state_estimate = self._inverse_scale_state(scaled_estimate)

        pressures = obs[
            self._network.pressure_slice
        ]

        demands = obs[
            self._network.demand_slice
        ]

        estimated_node_chlorine = state_estimate[
            self._network.state_node_chlorine_slice
        ]

        estimated_link_chlorine = state_estimate[
            self._network.state_link_chlorine_slice
        ]

        estimated_flows = state_estimate[
            self._network.state_flow_slice
        ]

        new_obs = np.concatenate(
            (
                pressures,
                estimated_flows,
                demands,
                estimated_node_chlorine,
                estimated_link_chlorine,
            )
        )

        return self._prepare_observation(new_obs)

    def _compute_reward_function(
            self,
            scada_data: ScadaData,
    ) -> float:
        lower_cl_bound = 0.3
        upper_cl_bound = 2.0

        nodes_quality = np.asarray(
            scada_data.get_data_nodes_quality(),
            dtype=np.float64,
        )

        upper_violation = np.maximum(
            nodes_quality - upper_cl_bound,
            0.0,
        )

        lower_violation = np.maximum(
            lower_cl_bound - nodes_quality,
            0.0,
        )

        violation_penalty = (
                np.sum(upper_violation)
                + np.sum(lower_violation)
        )

        chlorine_action = float(
            self._last_action.flatten()[0]
        )

        chlorine_usage_penalty = (
                self._chlorine_penalty_weight
                * chlorine_action
        )

        total_penalty = (
                violation_penalty
                + chlorine_usage_penalty
        )

        return -float(total_penalty)


if __name__ == "__main__":
    network = NETWORKS["cydbp"]

    config, n_sensors = create_network_scenario(
        network_config=network,
        n_sensors=10,
    )

    with ChlorineControlEnv(
        scenario_config=config,
        network_config=network,
        n_sensors=n_sensors,
        use_estimated_state=True,
    ) as env:
        obs, info = env.reset()

        print(
            "Observation finite:",
            np.all(np.isfinite(obs)),
        )

        api = env._scenario_sim.epanet_api

        node_idx = api.get_node_idx(
            network.injection_node_id
        )

        print(
            "Pattern-Index:",
            api.getpatternindex(
                network.injection_pattern_id
            )
        )

        print(
            "SourcePattern:",
            api.getnodevalue(
                node_idx,
                EpanetConstants.EN_SOURCEPAT
            )
        )

        print(
            "SourceType:",
            api.getnodevalue(
                node_idx,
                EpanetConstants.EN_SOURCETYPE
            )
        )

        print(
            "SourceQuality:",
            api.getnodevalue(
                node_idx,
                EpanetConstants.EN_SOURCEQUAL
            )
        )

        print("Observation shape:", obs.shape)
        print("Injection node index:", node_idx)
        print("Quality:", api.getqualinfo())

        print(
            "Pattern-Index:",
            api.getpatternindex(
                network.injection_pattern_id
            )
        )

        for step in range(10):
            obs, reward, terminated, truncated, info = (
                env.step(
                    np.array([5.0], dtype=np.float32)
                )
            )

            if not np.all(np.isfinite(obs)):
                raise RuntimeError(
                    f"Nicht-endliche Beobachtung "
                    f"in Schritt {step + 1}"
                )

            node_quality = np.asarray(
                info[
                    "scada_data"
                ].get_data_nodes_quality()
            ).reshape(-1)

            print(
                f"Schritt {step + 1}: "
                f"Reward={reward:.4f}, "
                f"Cl min={node_quality.min():.4f}, "
                f"Cl max={node_quality.max():.4f}"
            )
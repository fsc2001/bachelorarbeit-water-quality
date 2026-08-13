import os
import random
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "NeuralSurrogateKalmanChlorineEstimation"))
DATA_PATH = ROOT / "NeuralSurrogateKalmanChlorineEstimation" / "data"

from epyt_flow.simulation import EpanetConstants, ScenarioConfig, ScadaData
from epyt_control.envs import EpanetControlEnv
from epanet_plus import EPyT
from epyt_control.envs.actions import ChemicalInjectionAction
from epyt_control.signal_processing.state_estimation import TimeVaryingExtendedKalmanFilter
from run_exp_state_estimation import (
    create_random_sensor_placement,
    get_state_transition_model,
)


PRESSURES_START = 0
PRESSURES_END = 32

FLOWS_START = 32
FLOWS_END = 66

DEMANDS_START = 66
DEMANDS_END = 98

CL_NODES_START = 98
CL_NODES_END = 130

CL_LINKS_START = 130
CL_LINKS_END = 164

INJECTION_NODE_ID = "1"
INJECTION_PATTERN_ID = "my-chl-injection"

N_NODES = CL_NODES_END - CL_NODES_START   # 32
N_LINKS = CL_LINKS_END - CL_LINKS_START   # 34
N_CL_ITEMS = N_NODES + N_LINKS            # 66

def create_hanoi_scenario(n_sensors: int = 3):
    config = ScenarioConfig.load_from_file(
        DATA_PATH
        / (
            "control_cl_injection_scenario-"
            "Net1=False_randDemand=True."
            "epytflow_scenario_config"
        )
    )

    config._ScenarioConfig__f_inp_in = str(
        DATA_PATH
        / (
            "control_cl_injection_scenario-"
            "Net1=False_randDemand=True.inp"
        )
    )

    return config, n_sensors


class ChlorineControlEnv(EpanetControlEnv):
    def __init__(self, scenario_config: ScenarioConfig, n_sensors: int = 3):
        self._n_sensors = n_sensors
        self._sparse_node_indices = None
        self._kalman = None
        self._surrogate = None
        self._last_action = np.array([[0.]])

        super().__init__(
            scenario_config=scenario_config,
            chemical_injection_actions=[
                ChemicalInjectionAction(
                    node_id=INJECTION_NODE_ID,
                    pattern_id=INJECTION_PATTERN_ID,
                    source_type_id=EpanetConstants.EN_CONCEN,
                    upper_bound=5.0,
                )
            ],
            autoreset=True,
        )

    def _init_kalman(self, obs: np.ndarray) -> None:
        self._surrogate = get_state_transition_model(
            "Hanoi",
            str(DATA_PATH / "hanoi_randDemand=True_surrogate.pt")
        )

        # 32 Knotenqualitäten + 34 Linkqualitäten = 66 Chlorzustände
        self._surrogate.n_missing_flows = N_CL_ITEMS
        self._surrogate._normalize_input_output = False

        cl_nodes = obs[CL_NODES_START:CL_NODES_END]
        cl_links = obs[CL_LINKS_START:CL_LINKS_END]
        flows = obs[FLOWS_START:FLOWS_END]

        raw_state = np.concatenate((cl_nodes, cl_links, flows))
        state_dim = raw_state.shape[0]  # 32 + 34 + 34 = 100

        self._state_dim = state_dim

        # Der Scaler wurde auf Zustand + Control-Action trainiert:
        # 100 Zustandswerte + 1 Aktuator = 101 Werte.
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

        # Wie in der Referenzimplementierung:
        # n_sensors Knotensensoren und n_sensors Linksensoren.
        self._M, self._flows_idx = create_random_sensor_placement(
            self._n_sensors,
            self._n_sensors,
            N_NODES,
            N_LINKS,
            state_dim
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
            dtype=np.float64
        )

        obs, info = super().reset(**kwargs)

        # Das Szenario wurde von super().reset() neu geladen.
        # Jetzt fehlendes Pattern und Source reparieren,
        # bevor die erste Action ausgeführt wird.

        self._init_kalman(obs)

        initial_estimate = self._inverse_scale_state(
            self._kalman._x
        )

        pressures = obs[
                    PRESSURES_START:PRESSURES_END
                    ]
        demands = obs[
                  DEMANDS_START:DEMANDS_END
                  ]

        return np.concatenate(
            (
                pressures,
                demands,
                initial_estimate
            )
        ), info

    def step(self, action: np.ndarray):
        self._last_action = action.reshape(1, -1)
        obs, reward, terminated, truncated, info = super().step(action)
        return self._kalman_obs(obs), reward, terminated, truncated, info

    def _kalman_obs(self, obs: np.ndarray) -> np.ndarray:
        cl_nodes = obs[CL_NODES_START:CL_NODES_END]
        cl_links = obs[CL_LINKS_START:CL_LINKS_END]
        flows = obs[FLOWS_START:FLOWS_END]

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

        pressures = obs[PRESSURES_START:PRESSURES_END]
        demands = obs[DEMANDS_START:DEMANDS_END]

        return np.concatenate((pressures, demands, state_estimate))

    def _compute_reward_function(self, scada_data: ScadaData) -> float:
        lower_cl_bound = 0.3
        upper_cl_bound = 2.0

        nodes_quality = np.asarray(
            scada_data.get_data_nodes_quality()
        )

        upper_violation = np.maximum(
            nodes_quality - upper_cl_bound,
            0.0
        )

        lower_violation = np.maximum(
            lower_cl_bound - nodes_quality,
            0.0
        )

        violation_penalty = np.sum(upper_violation) + np.sum(lower_violation)

        return -float(violation_penalty)


if __name__ == "__main__":
    config, n_sensors = create_hanoi_scenario(3)

    with ChlorineControlEnv(
        config,
        n_sensors
    ) as env:
        print("Verwendete INP:", env._scenario_config.f_inp_in)
        obs, info = env.reset()

        api = env._scenario_sim.epanet_api
        node_idx = api.get_node_idx(
            INJECTION_NODE_ID
        )

        print("Quality:", api.getqualinfo())

        print(
            "Pattern-Index:",
            api.getpatternindex(
                INJECTION_PATTERN_ID
            )
        )

        print(
            "SourcePattern:",
            api.getnodevalue(
                node_idx,
                EpanetConstants.EN_SOURCEPAT
            )
        )

        for step_idx in range(10):
            action = np.array(
                [5.0],
                dtype=np.float32
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            node_quality = np.asarray(
                info["scada_data"].get_data_nodes_quality()
            )

            print(
                f"Schritt {step_idx + 1}: "
                f"Reward={reward:.4f}, "
                f"Cl min={node_quality.min():.4f}, "
                f"Cl max={node_quality.max():.4f}"
            )

            if terminated or truncated:
                break
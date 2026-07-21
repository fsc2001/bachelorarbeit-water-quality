import os
import numpy as np
from stable_baselines3 import PPO
from chlorine_env import ChlorineControlEnv, create_hanoi_scenario
from epyt_flow.simulation import EpanetConstants, ScenarioConfig, ScadaData
from epyt_control.envs import EpanetControlEnv
from epyt_control.envs.actions import ChemicalInjectionAction

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

class BaselineEnv(EpanetControlEnv):
    """RL-Env mit ALLEN Sensoren – kein Kalman, echte Werte"""
    def __init__(self, scenario_config: ScenarioConfig):
        super().__init__(
            scenario_config=scenario_config,
            chemical_injection_actions=[ChemicalInjectionAction(
                node_id="1",
                pattern_id="my-chl-injection",
                source_type_id=EpanetConstants.EN_CONCEN,
                upper_bound=5.)],
            autoreset=True
        )

    def _compute_reward_function(self, scada_data: ScadaData) -> float:
        upper_cl_bound = 2.
        lower_cl_bound = 0.3
        reward = 0.

        nodes_quality = scada_data.get_data_nodes_quality()

        upper_bound_violation_idx = nodes_quality > upper_cl_bound
        reward += np.sum(nodes_quality[upper_bound_violation_idx] - upper_cl_bound)

        lower_bound_violation_idx = nodes_quality < lower_cl_bound
        reward += -1. * np.sum(nodes_quality[lower_bound_violation_idx] - lower_cl_bound)

        return reward


if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(base)
    print("Env liegt in:", base)
    print("Parent:", parent)
    print("Inhalt Parent:", os.listdir(parent))
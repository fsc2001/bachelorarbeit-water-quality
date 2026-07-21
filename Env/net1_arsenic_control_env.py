import os
from pathlib import Path
import random
import numpy as np
from epyt_flow.data.benchmarks import load_leakdb_scenarios
from epyt_flow.simulation import ScenarioSimulator, EpanetConstants, ModelUncertainty, ScenarioConfig, ScadaData, SensorConfig
from epyt_flow.simulation.events import SpeciesInjectionEvent, AbruptLeakage
from epyt_flow.uncertainty import RelativeUniformUncertainty, AbsoluteGaussianUncertainty
from epyt_flow.utils import to_seconds

from epyt_control.envs import AdvancedQualityControlEnv
from epyt_control.envs.actions import SpeciesInjectionAction


def create_hanoi_scenario(s_id: str = ["0"]):
    config, = load_leakdb_scenarios(scenarios_id=s_id, use_net1=True)
    config = ScenarioConfig(scenario_config=config,
                            f_msx_in="net1_arsenic_contamination.msx")

    with ScenarioSimulator(scenario_config=config) as sim:
        # Set simulation duration to 21 days
        sim.set_general_parameters(simulation_duration=to_seconds(days=21))

        # Specify uncertainties -- similar to the one already implemented in LeakDB
        my_uncertainties = {"global_pipe_length_uncertainty": RelativeUniformUncertainty(low=0.8, high=1.2),
                            "global_pipe_roughness_uncertainty": RelativeUniformUncertainty(low=0.8, high=1.2),
                            "global_base_demand_uncertainty": RelativeUniformUncertainty(low=0.8, high=1.2),
                            "global_demand_pattern_uncertainty": AbsoluteGaussianUncertainty(mean=0, scale=.1)}
        sim.set_model_uncertainty(ModelUncertainty(**my_uncertainties))

        # Place some chlorine sensors and also keep track of the contaminant
        all_nodes = sim.sensor_config.nodes
        sim.set_bulk_species_node_sensors({"Chlorine": all_nodes,
                                           # Also: Keep track of the contaminant
                                           "AsIII": all_nodes})   # Arsenite

        # Create a 1-day contamination event --
        # i.e. injection of Arsenite (100mg/L) at a specific node
        # TODO: Vary place, strength, and time
        # contamination_event = SpeciesInjectionEvent(species_id="AsIII", node_id="3",
        #                                            profile=np.array([10000]),
        #                                            source_type=EpanetConstants.EN_MASS,
        #                                            start_time=to_seconds(days=3),
        #                                            end_time=to_seconds(days=4))
        # sim.add_system_event(contamination_event)

        # Return scenario configuration
        return sim.get_scenario_config()


class MyEnv(AdvancedQualityControlEnv):
    """
    A simple environment for controlling the chlorine injection.
    """
    def __init__(self, s_config: ScenarioConfig, cl_source_id: str):
        self._cl_species_id = "Chlorine"

        # Create scenario and set autoreset=True
        super().__init__(scenario_config=s_config,
                         action_space=[SpeciesInjectionAction(node_id=cl_source_id,
                                                              species_id=self._cl_species_id,
                                                              pattern_id="CL2PAT",
                                                              source_type_id=EpanetConstants.EN_MASS,
                                                              upper_bound=10.)],
                         rerun_hydraulics_when_reset=False, # Do not re-run hydraulic simulation when reseting the environment.
                         autoreset=True)

        self.__sensor_config_reward = None

    def step(self, action: np.ndarray):
        # Scaling of Cl injection
        return super().step(action * 1000)

    def _compute_reward_function(self, scada_data: ScadaData) -> float:
        """
        Computes the current reward based on the current sensors readings (i.e. SCADA data).

        Parameters
        ----------
        :class:`epyt_flow.simulation.ScadaData`
            Current sensor readings.

        Returns
        -------
        `float`
            Current reward.
        """
        # TODO: Replace with smth. more reasonable!
        # Sum up (negative) residuals for out of bounds Cl concentrations at nodes -- i.e.
        # reward of zero means everythings is okay, while a negative reward
        # denotes Cl concentration bound violations
        reward = 0.

        # Regulation Limits
        upper_cl_bound = 2.  # (mg/l)
        lower_cl_bound = .3  # (mg/l)

        if self.__sensor_config_reward is None:
            self.__sensor_config_reward = SensorConfig.create_empty_sensor_config(scada_data.sensor_config)   # TODO: Move to constructor
            self.__sensor_config_reward.bulk_species_node_sensors = {self._cl_species_id: scada_data.sensor_config.nodes}
        scada_data.change_sensor_config(self.__sensor_config_reward)

        nodes_quality = scada_data.get_data_bulk_species_node_concentration({self._cl_species_id: scada_data.sensor_config.nodes})

        upper_bound_violation_idx = nodes_quality > upper_cl_bound
        reward += np.sum(nodes_quality[upper_bound_violation_idx] - upper_cl_bound)

        lower_bound_violation_idx = nodes_quality < lower_cl_bound
        reward += -1. * np.sum(nodes_quality[lower_bound_violation_idx] - lower_cl_bound)

        return reward


if __name__ == "__main__":
    s_config = create_hanoi_scenario()

    with MyEnv(s_config, cl_source_id="2") as env:  # TODO: cl_source_id="2" in the case of Net1
        obs, info = env.reset()
        print(obs)

        for _ in range(10): # Run 10 iterations (time steps)
            act = env.action_space.sample()     # TODO: RL-agent logic goes here
            obs, reward, terminated, _, info = env.step(act)
            print(obs)

            if terminated is True:
                break

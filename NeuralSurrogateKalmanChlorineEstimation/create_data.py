"""
This modules creates (by running simulations) the data sets used in the experiments.
"""
import os
from pathlib import Path
import numpy as np
from water_benchmark_hub import load
from epyt_flow.data.benchmarks import load_leakdb_scenarios
from epyt_flow.simulation import ScenarioSimulator, EpanetConstants, ModelUncertainty, \
    ScenarioConfig, ScadaData, SensorConfig
from epyt_flow.uncertainty import RelativeUniformUncertainty, AbsoluteGaussianUncertainty
from epyt_flow.utils import to_seconds
from epyt_control.envs import HydraulicControlEnv
from epyt_control.envs.actions import ChemicalInjectionAction


path_to_scenarios = "data"


def create_leakdb_scenario(use_net1: bool = False, randomized_demands: bool = False) -> None:
    # Create scenarios based on the LeakDB Hanoi
    [scenario_config] = load_leakdb_scenarios(scenarios_id=list(range(1)), use_net1=use_net1)
    with ScenarioSimulator(scenario_config=scenario_config) as sim:
        sim.set_general_parameters(simulation_duration=to_seconds(days=120))

        if randomized_demands is True:
            sim.randomize_demands()

        # Enable chlorine simulation and place a chlorine injection pump at the reservoir
        sim.enable_chemical_analysis()

        reservoid_node_id = sim.epanet_api.get_all_reservoirs_id()[0]
        sim.add_quality_source(node_id=reservoid_node_id,
                                pattern=np.array([1.]),
                                source_type=EpanetConstants.EN_CONCEN,
                                pattern_id="my-chl-injection")

        # Set initial concentration and simple (constant) reactions
        for node_idx in sim.epanet_api.get_all_nodes_idx():
            sim.epanet_api.set_node_init_quality(node_idx, 0)
        for link_idx in sim.epanet_api.get_all_links_idx():
            sim.epanet_api.setlinkvalue(link_idx, EpanetConstants.EN_BULKORDER, -.5)
            sim.epanet_api.setlinkvalue(link_idx, EpanetConstants.EN_WALLORDER, -.01)

        # Set flow and chlorine sensors everywhere
        sim.sensor_config = SensorConfig.create_empty_sensor_config(sim.sensor_config)
        sim.set_pressure_sensors(sim.sensor_config.nodes)
        sim.set_demand_sensors(sim.sensor_config.nodes)
        sim.set_flow_sensors(sim.sensor_config.links)
        sim.set_node_quality_sensors(sim.sensor_config.nodes)
        sim.set_link_quality_sensors(sim.sensor_config.links)

        # Specify uncertainties -- similar to the one already implemented in LeakDB
        my_uncertainties = {"global_pipe_length_uncertainty": RelativeUniformUncertainty(low=0.8, high=1.8),
                            "global_pipe_roughness_uncertainty": RelativeUniformUncertainty(low=0.8, high=1.8),
                            "global_base_demand_uncertainty": RelativeUniformUncertainty(low=0.8, high=1.8),
                            "global_demand_pattern_uncertainty": AbsoluteGaussianUncertainty(mean=0, scale=.02)}
        sim.set_model_uncertainty(ModelUncertainty(**my_uncertainties))

        # Export scenario
        Path(path_to_scenarios).mkdir(exist_ok=True)
        sim.save_to_epanet_file(os.path.join(path_to_scenarios, f"control_cl_injection_scenario-Net1={use_net1}_randDemand={randomized_demands}.inp"))
        sim.get_scenario_config().save_to_file(os.path.join(path_to_scenarios, f"control_cl_injection_scenario-Net1={use_net1}_randDemand={randomized_demands}"))


class LeakdDbChlorineInjectionEnv(HydraulicControlEnv):
        def __init__(self, use_net1: bool = False, randomized_demands: bool = False):
            # Load scenario and set autoreset=True
            scenario_config_file_in = os.path.join(path_to_scenarios,
                                                   f"control_cl_injection_scenario-Net1={use_net1}_randDemand={randomized_demands}.epytflow_scenario_config")

            injection_node_id = "1"
            if use_net1 is True:
                injection_node_id = "9"

            super().__init__(scenario_config=ScenarioConfig.load_from_file(scenario_config_file_in),
                            chemical_injection_actions=[ChemicalInjectionAction(node_id=injection_node_id,
                                                                                pattern_id="my-chl-injection",
                                                                                source_type_id=EpanetConstants.EN_CONCEN,
                                                                                upper_bound=5.)],
                            autoreset=False,
                            reload_scenario_when_reset=False)

        def _compute_reward_function(self, scada_data: ScadaData) -> float:
            return 0

def create_data_set(use_net1: bool, randomized_demands: bool, file_out: str, path_out: str = "data") -> None:
    scada_data = None
    control_actions = []
    with LeakdDbChlorineInjectionEnv(use_net1, randomized_demands) as env:
        env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            control_actions.append(action)
            _, _, terminated, _, info = env.step(action)
            if terminated is True:
                break

            current_scada_data = info["scada_data"]
            if scada_data is None:
                scada_data = current_scada_data
            else:
                scada_data.concatenate(current_scada_data)

        env.close()

    Path(path_out).mkdir(exist_ok=True)
    scada_data.save_to_file(os.path.join(path_out, f"{file_out}.epytflow_scada_data"))
    np.savez(os.path.join(path_out, f"{file_out}.npz"), control_actions=control_actions)


def create_cydbp_scenario(randomized_demands: bool = False) -> None:
    with ScenarioSimulator(f_inp_in="CY-DBP_dist_stream.inp") as sim:
        sim.set_general_parameters(simulation_duration=to_seconds(days=120),
                                   hydraulic_time_step=1800,
                                   quality_time_step=300)

        if randomized_demands is True:
            sim.randomize_demands()

        # Enable chlorine simulation and place a chlorine injection pump at the reservoir
        sim.enable_chemical_analysis()

        for reservoid_node_id in sim.epanet_api.get_all_reservoirs_id():
            sim.add_quality_source(node_id=reservoid_node_id,
                                   pattern=np.array([1.]),
                                   source_type=EpanetConstants.EN_CONCEN,
                                   pattern_id=f"my-chl-inj-{reservoid_node_id}")

        # Set initial concentration and simple (constant) reactions
        for node_idx in sim.epanet_api.get_all_nodes_idx():
            sim.epanet_api.set_node_init_quality(node_idx, 0)
        for link_idx in sim.epanet_api.get_all_links_idx():
            sim.epanet_api.setlinkvalue(link_idx, EpanetConstants.EN_BULKORDER, -.5)
            sim.epanet_api.setlinkvalue(link_idx, EpanetConstants.EN_WALLORDER, -.01)

        # Set flow and chlorine sensors everywhere
        sim.sensor_config = SensorConfig.create_empty_sensor_config(sim.sensor_config)
        sim.set_pressure_sensors(sim.sensor_config.nodes)
        sim.set_demand_sensors(sim.sensor_config.nodes)
        sim.set_flow_sensors(sim.sensor_config.links)
        sim.set_node_quality_sensors(sim.sensor_config.nodes)
        sim.set_link_quality_sensors(sim.sensor_config.links)

        # Specify uncertainties -- similar to the one already implemented in LeakDB
        my_uncertainties = {"global_pipe_length_uncertainty": RelativeUniformUncertainty(low=0.8, high=1.8),
                            "global_pipe_roughness_uncertainty": RelativeUniformUncertainty(low=0.8, high=1.8),
                            "global_base_demand_uncertainty": RelativeUniformUncertainty(low=0.8, high=1.8),
                            "global_demand_pattern_uncertainty": AbsoluteGaussianUncertainty(mean=0, scale=.02)}
        sim.set_model_uncertainty(ModelUncertainty(**my_uncertainties))

        # Export scenario
        Path(path_to_scenarios).mkdir(exist_ok=True)
        sim.save_to_epanet_file(os.path.join(path_to_scenarios, f"control_cl_injection_scenario-CYDBP_randDemand={randomized_demands}.inp"))
        sim.get_scenario_config().save_to_file(os.path.join(path_to_scenarios, f"control_cl_injection_scenario-CYDBP_randDemand={randomized_demands}"))


class CydbpChlorineInjectionEnv(HydraulicControlEnv):
        def __init__(self, randomized_demands: bool = False):
            # Load scenario and set autoreset=True
            scenario_config_file_in = os.path.join(path_to_scenarios,
                                                   f"control_cl_injection_scenario-CYDBP_randDemand={randomized_demands}.epytflow_scenario_config")

            #injection_nodes_id = ["WTP", "Desalination"]
            injection_nodes_id = ["T_Zone"]
            chemical_injection_actions = []
            for injection_node_id in injection_nodes_id:
                chemical_injection_actions.append(ChemicalInjectionAction(node_id=injection_node_id,
                                                                          pattern_id=f"my-chl-inj-{injection_node_id}",
                                                                          source_type_id=EpanetConstants.EN_CONCEN,
                                                                          upper_bound=5.))

            super().__init__(scenario_config=ScenarioConfig.load_from_file(scenario_config_file_in),
                             chemical_injection_actions=chemical_injection_actions,
                             autoreset=False,
                             reload_scenario_when_reset=False)

        def _compute_reward_function(self, scada_data: ScadaData) -> float:
            return 0


def create_cydbp_dataset(randomized_demands: bool, file_out: str, path_out: str = "data") -> None:
    scada_data = None
    control_actions = []
    with CydbpChlorineInjectionEnv(randomized_demands) as env:
        env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            control_actions.append(action)
            _, _, terminated, _, info = env.step(action)
            if terminated is True:
                break

            current_scada_data = info["scada_data"]
            if scada_data is None:
                scada_data = current_scada_data
            else:
                scada_data.concatenate(current_scada_data)

        env.close()

    Path(path_out).mkdir(exist_ok=True)
    scada_data.save_to_file(os.path.join(path_out, f"{file_out}.epytflow_scada_data"))
    np.savez(os.path.join(path_out, f"{file_out}.npz"), control_actions=control_actions)



if __name__ == "__main__":
    # CY-DBP
    create_cydbp_scenario(randomized_demands=False)
    create_cydbp_scenario(randomized_demands=True)

    create_cydbp_dataset(randomized_demands=False, file_out="cydbp_randDemand=False_training")
    create_cydbp_dataset(randomized_demands=False, file_out="cydbp_randDemand=False_validation")
    create_cydbp_dataset(randomized_demands=False, file_out="cydbp_randDemand=False_test")

    create_cydbp_dataset(randomized_demands=True, file_out="cydbp_randDemand=True_training")
    create_cydbp_dataset(randomized_demands=True, file_out="cydbp_randDemand=True_validation")
    create_cydbp_dataset(randomized_demands=True, file_out="cydbp_randDemand=True_test")

    # Hanoi
    create_leakdb_scenario(use_net1=False, randomized_demands=False)
    create_data_set(use_net1=False, randomized_demands=False, file_out="hanoi_randDemand=False_training")
    create_data_set(use_net1=False, randomized_demands=False, file_out="hanoi_randDemand=False_validation")
    create_data_set(use_net1=False, randomized_demands=False, file_out="hanoi_randDemand=False_test")

    create_leakdb_scenario(use_net1=False, randomized_demands=True)
    create_data_set(use_net1=False, randomized_demands=True, file_out="hanoi_randDemand=True_training")
    create_data_set(use_net1=False, randomized_demands=True, file_out="hanoi_randDemand=True_validation")
    create_data_set(use_net1=False, randomized_demands=True, file_out="hanoi_randDemand=True_test")

    # Net1
    create_leakdb_scenario(use_net1=True, randomized_demands=False)
    create_data_set(use_net1=True, randomized_demands=False, file_out="net1_randDemand=False_training")
    create_data_set(use_net1=True, randomized_demands=False, file_out="net1_randDemand=False_validation")
    create_data_set(use_net1=True, randomized_demands=False, file_out="net1_randDemand=False_test")

    create_leakdb_scenario(use_net1=True, randomized_demands=True)
    create_data_set(use_net1=True, randomized_demands=True, file_out="net1_randDemand=True_training")
    create_data_set(use_net1=True, randomized_demands=True, file_out="net1_randDemand=True_validation")
    create_data_set(use_net1=True, randomized_demands=True, file_out="net1_randDemand=True_test")

import numpy as np

from epyt_flow.data.benchmarks import load_leakdb_scenarios
from epyt_flow.simulation import (
    ScenarioSimulator,
    ScenarioConfig,
    EpanetConstants,
    ScadaData,
    SensorConfig,
    ModelUncertainty,
)
from epyt_flow.utils import to_seconds
from epyt_control.envs import EpanetControlEnv
from epyt_control.envs.actions import ChemicalInjectionAction


SCENARIO_NAME = "minimal_chlorine_scenario"
PATTERN_ID = "my-chl-injection"


def create_scenario() -> None:
    [config] = load_leakdb_scenarios(
        scenarios_id=[0],
        use_net1=False,
    )

    with ScenarioSimulator(scenario_config=config) as sim:
        sim.set_general_parameters(
            simulation_duration=to_seconds(days=2)
        )

        sim.enable_chemical_analysis()

        reservoir_id = sim.epanet_api.get_all_reservoirs_id()[0]

        sim.add_quality_source(
            node_id=reservoir_id,
            pattern=np.array([1.0]),
            source_type=EpanetConstants.EN_CONCEN,
            pattern_id=PATTERN_ID,
        )

        sim.sensor_config = SensorConfig.create_empty_sensor_config(
            sim.sensor_config
        )
        sim.set_node_quality_sensors(sim.sensor_config.nodes)

        # Nur für den isolierten Test:
        # LeakDB bringt eine nicht serialisierbare lokale Klasse mit.
        sim.set_model_uncertainty(ModelUncertainty())

        sim.save_to_epanet_file(f"{SCENARIO_NAME}.inp")
        sim.get_scenario_config().save_to_file(SCENARIO_NAME)


class MinimalChlorineEnv(EpanetControlEnv):
    def __init__(self):
        super().__init__(
            scenario_config=ScenarioConfig.load_from_file(
                f"{SCENARIO_NAME}.epytflow_scenario_config"
            ),
            chemical_injection_actions=[
                ChemicalInjectionAction(
                    node_id="1",
                    pattern_id=PATTERN_ID,
                    source_type_id=EpanetConstants.EN_CONCEN,
                    upper_bound=5.0,
                )
            ],
            autoreset=False,
            reload_scenario_when_reset=False,
        )

    def _compute_reward_function(
        self,
        scada_data: ScadaData,
    ) -> float:
        return 0.0


if __name__ == "__main__":
    create_scenario()

    with MinimalChlorineEnv() as env:
        obs, info = env.reset()
        print("Reset funktioniert:", obs.shape)

        for action_value in [0.0, 1.0, 3.0]:
            action = np.array([action_value], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            quality = np.asarray(
                info["scada_data"].get_data_nodes_quality()
            )

            print(
                "Action:",
                action_value,
                "Cl min/max:",
                float(quality.min()),
                float(quality.max()),
            )
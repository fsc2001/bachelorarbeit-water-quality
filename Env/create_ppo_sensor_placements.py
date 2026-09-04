"""
This module combines the random and centrality-based sensor placements
used for the PPO experiments.
"""

import json
import random
import sys
from pathlib import Path
from epyt_flow.simulation.scada import ScadaData

PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_REPO = PROJECT_DIR / "NeuralSurrogateKalmanChlorineEstimation"
DATA_DIR = REFERENCE_REPO / "data"
RESULTS_DIR = PROJECT_DIR / "results"

sys.path.insert(0, str(PROJECT_DIR))

from Env.network_config import NETWORKS

def create_random_placement(network_name, n_sensors, seed):
    network = NETWORKS[network_name]

    rng = random.Random(seed)

    node_indices = sorted(
        rng.sample(range(network.n_nodes), k=n_sensors)
    )

    link_indices = sorted(
        rng.sample(range(network.n_links), k=n_sensors)
    )

    scada_path = (
            DATA_DIR
            / f"{network_name}_randDemand=True_training.epytflow_scada_data"
    )

    scada_data = ScadaData.load_from_file(str(scada_path))
    topology = scada_data.network_topo

    node_ids = [
        str(node_id)
        for node_id in topology.get_all_nodes()
    ]

    link_ids = [
        str(link[0])
        for link in topology.get_all_links()
    ]

    return {
        "seed": seed,
        "n_node_sensors": n_sensors,
        "n_link_sensors": n_sensors,
        "node_indices": node_indices,
        "node_ids": [node_ids[index] for index in node_indices],
        "link_indices": link_indices,
        "link_ids": [link_ids[index] for index in link_indices],
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    random_selection_path = (
            RESULTS_DIR / "selected_random_ppo_placements.json"
    )

    centrality_path = (
            RESULTS_DIR / "centrality_sensor_placements.json"
    )

    with random_selection_path.open("r", encoding="utf-8") as file:
        random_selections = json.load(file)

    with centrality_path.open("r", encoding="utf-8") as file:
        centrality_placements = json.load(file)

    placements = {}

    for network_name, selection in random_selections.items():
        random_placement = create_random_placement(
            network_name=network_name,
            n_sensors=selection["n_sensors"],
            seed=selection["seed"],
        )

        placements[network_name] = {
            "random": random_placement,
            "centrality": centrality_placements[network_name],
        }

        print(
            f"{network_name}: "
            f"random seed {selection['seed']}, "
            f"{selection['n_sensors']} node and link sensors"
        )

    output_path = RESULTS_DIR / "ppo_sensor_placements.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            placements,
            file,
            indent=2,
        )

    print(f"Saved placements to: {output_path}")


if __name__ == "__main__":
    main()

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "NeuralSurrogateKalmanChlorineEstimation"
DATA_DIR = REPO / "data"
RESULT_DIR = ROOT / "results"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

from Env.network_config import NETWORKS
from epyt_flow.simulation.scada import ScadaData


SELECTED_RANDOM = {
    "net1": {
        "n_sensors": 3,
        "seed": 0,
    },
    "hanoi": {
        "n_sensors": 10,
        "seed": 0,
    },
    "cydbp": {
        "n_sensors": 75,
        "seed": 4,
    },
}


centrality_path = (
    RESULT_DIR
    / "centrality_sensor_placements.json"
)

with centrality_path.open(
    "r",
    encoding="utf-8",
) as file:
    centrality = json.load(file)


result = {}


for network_name, config in SELECTED_RANDOM.items():

    network = NETWORKS[network_name]

    n_sensors = config["n_sensors"]
    seed = config["seed"]

    # Exact same sampling order as the professor's function:
    # nodes first, then links.
    rng = random.Random(seed)

    node_indices = sorted(
        rng.sample(
            range(network.n_nodes),
            k=n_sensors,
        )
    )

    link_indices = sorted(
        rng.sample(
            range(network.n_links),
            k=n_sensors,
        )
    )

    # Get physical IDs for documentation.
    scada_path = (
        DATA_DIR
        / (
            f"{network_name}_"
            "randDemand=True_training."
            "epytflow_scada_data"
        )
    )

    scada = ScadaData.load_from_file(
        str(scada_path)
    )

    topo = scada.network_topo

    node_ids_all = [
        str(x)
        for x in topo.get_all_nodes()
    ]

    link_ids_all = [
        str(x[0])
        for x in topo.get_all_links()
    ]

    random_placement = {
        "seed": seed,

        "n_node_sensors": n_sensors,
        "n_link_sensors": n_sensors,

        "node_indices": node_indices,
        "node_ids": [
            node_ids_all[i]
            for i in node_indices
        ],

        "link_indices": link_indices,
        "link_ids": [
            link_ids_all[i]
            for i in link_indices
        ],
    }

    result[network_name] = {
        "random": random_placement,
        "centrality": centrality[
            network_name
        ],
    }

    print()
    print("=" * 60)
    print(network_name.upper())
    print("=" * 60)

    print(
        "Random seed:",
        seed,
    )

    print(
        "Random node indices:",
        node_indices,
    )

    print(
        "Random link indices:",
        link_indices,
    )

    print(
        "Centrality node indices:",
        centrality[
            network_name
        ]["node_indices"],
    )

    print(
        "Centrality link indices:",
        centrality[
            network_name
        ]["link_indices"],
    )


output_path = (
    RESULT_DIR
    / "ppo_sensor_placements.json"
)

with output_path.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        result,
        file,
        indent=2,
    )


print()
print("Saved:")
print(output_path)
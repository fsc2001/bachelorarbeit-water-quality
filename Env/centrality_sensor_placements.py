"""
This module generates centrality-based sensor placements for all networks.
"""

import json
import sys
from pathlib import Path

import networkx as nx
from epyt_flow.simulation.scada import ScadaData


PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_REPO = PROJECT_DIR / "NeuralSurrogateKalmanChlorineEstimation"
DATA_DIR = REFERENCE_REPO / "data"
RESULTS_DIR = PROJECT_DIR / "results"

sys.path.insert(0, str(PROJECT_DIR))

from Env.network_config import NETWORKS


SENSOR_COUNTS = {
    "net1": 3,
    "hanoi": 10,
    "cydbp": 75,
}


def compute_placement(network_name):
    network = NETWORKS[network_name]
    n_sensors = SENSOR_COUNTS[network_name]

    scada_path = (
        DATA_DIR
        / f"{network_name}_randDemand=True_training.epytflow_scada_data"
    )

    scada_data = ScadaData.load_from_file(str(scada_path))
    topology = scada_data.network_topo

    node_records = topology.get_all_nodes()
    link_records = topology.get_all_links()

    node_ids = [str(node_id) for node_id in node_records]
    link_ids = [str(link[0]) for link in link_records]

    if len(node_ids) != network.n_nodes:
        raise RuntimeError(
            f"{network_name}: expected {network.n_nodes} nodes, "
            f"got {len(node_ids)}."
        )

    if len(link_ids) != network.n_links:
        raise RuntimeError(
            f"{network_name}: expected {network.n_links} links, "
            f"got {len(link_ids)}."
        )

    node_indices = {
        node_id: index
        for index, node_id in enumerate(node_ids)
    }

    link_indices = {
        link_id: index
        for index, link_id in enumerate(link_ids)
    }

    graph = nx.MultiGraph()
    graph.add_nodes_from(node_ids)

    for link_id, endpoints in link_records:
        if len(endpoints) != 2:
            raise RuntimeError(
                f"Unexpected endpoints for link {link_id}: {endpoints}"
            )

        start_node = str(endpoints[0])
        end_node = str(endpoints[1])
        link_id = str(link_id)

        graph.add_edge(
            start_node,
            end_node,
            key=link_id,
        )

    node_centrality = nx.betweenness_centrality(
        graph,
        normalized=True,
        weight=None,
    )

    ranked_nodes = sorted(
        node_ids,
        key=lambda node_id: (
            -node_centrality[node_id],
            node_indices[node_id],
        ),
    )

    selected_node_ids = ranked_nodes[:n_sensors]

    selected_node_indices = sorted(
        node_indices[node_id]
        for node_id in selected_node_ids
    )

    raw_edge_centrality = nx.edge_betweenness_centrality(
        graph,
        normalized=True,
        weight=None,
    )

    link_centrality = {
        str(edge[2]): float(score)
        for edge, score in raw_edge_centrality.items()
    }

    ranked_links = sorted(
        link_ids,
        key=lambda link_id: (
            -link_centrality[link_id],
            link_indices[link_id],
        ),
    )

    selected_link_ids = ranked_links[:n_sensors]

    selected_link_indices = sorted(
        link_indices[link_id]
        for link_id in selected_link_ids
    )

    return {
        "network": network_name,
        "n_node_sensors": n_sensors,
        "n_link_sensors": n_sensors,
        "node_indices": selected_node_indices,
        "node_ids": [
            node_ids[index]
            for index in selected_node_indices
        ],
        "link_indices": selected_link_indices,
        "link_ids": [
            link_ids[index]
            for index in selected_link_indices
        ],
        "node_centrality": {
            node_id: float(node_centrality[node_id])
            for node_id in selected_node_ids
        },
        "link_centrality": {
            link_id: float(link_centrality[link_id])
            for link_id in selected_link_ids
        },
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    placements = {}

    for network_name in SENSOR_COUNTS:
        placements[network_name] = compute_placement(network_name)

        print(
            f"{network_name}: "
            f"{SENSOR_COUNTS[network_name]} node and link sensors"
        )

    output_path = RESULTS_DIR / "centrality_sensor_placements.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            placements,
            file,
            indent=2,
        )

    print(f"Saved placements to: {output_path}")


if __name__ == "__main__":
    main()
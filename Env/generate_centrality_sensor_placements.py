import json
import sys
from pathlib import Path

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "NeuralSurrogateKalmanChlorineEstimation"
DATA_DIR = REPO / "data"
RESULT_DIR = ROOT / "results"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

from Env.network_config import NETWORKS
from epyt_flow.simulation.scada import ScadaData


SENSOR_COUNTS = {
    "net1": 3,
    "hanoi": 10,
    "cydbp": 75,
}


def compute_placement(network_name):
    network = NETWORKS[network_name]
    n_sensors = SENSOR_COUNTS[network_name]

    # We only need the network topology.
    # Use training rather than test data for cleaner methodology.
    scada_path = (
        DATA_DIR
        / f"{network_name}_randDemand=True_training.epytflow_scada_data"
    )

    scada = ScadaData.load_from_file(
        str(scada_path)
    )

    topo = scada.network_topo

    # --------------------------------------------------------
    # Ordered physical IDs
    # --------------------------------------------------------

    node_records = topo.get_all_nodes()
    link_records = topo.get_all_links()

    # get_all_nodes():
    # ['10', '11', ...]
    node_ids = [
        str(node_id)
        for node_id in node_records
    ]

    # get_all_links():
    # [('10', ['10', '11']), ...]
    link_ids = [
        str(record[0])
        for record in link_records
    ]

    if len(node_ids) != network.n_nodes:
        raise RuntimeError(
            f"{network_name}: expected "
            f"{network.n_nodes} nodes, "
            f"got {len(node_ids)}"
        )

    if len(link_ids) != network.n_links:
        raise RuntimeError(
            f"{network_name}: expected "
            f"{network.n_links} links, "
            f"got {len(link_ids)}"
        )

    if len(set(node_ids)) != len(node_ids):
        raise RuntimeError(
            f"{network_name}: duplicate node IDs"
        )

    if len(set(link_ids)) != len(link_ids):
        raise RuntimeError(
            f"{network_name}: duplicate link IDs"
        )

    node_index = {
        node_id: idx
        for idx, node_id in enumerate(node_ids)
    }

    link_index = {
        link_id: idx
        for idx, link_id in enumerate(link_ids)
    }

    # --------------------------------------------------------
    # Build topology graph
    # --------------------------------------------------------

    # MultiGraph is necessary because CY-DBP contains
    # parallel physical links.
    graph = nx.MultiGraph()

    graph.add_nodes_from(node_ids)

    for link_id, endpoints in link_records:
        link_id = str(link_id)

        if len(endpoints) != 2:
            raise RuntimeError(
                f"{network_name}: unexpected endpoints "
                f"for {link_id}: {endpoints}"
            )

        u = str(endpoints[0])
        v = str(endpoints[1])

        graph.add_edge(
            u,
            v,
            key=link_id,
            link_id=link_id,
        )

    # --------------------------------------------------------
    # Node betweenness centrality
    # --------------------------------------------------------

    node_scores = nx.betweenness_centrality(
        graph,
        normalized=True,
        weight=None,
    )

    ranked_nodes = sorted(
        node_ids,
        key=lambda node_id: (
            -node_scores[node_id],
            node_index[node_id],
        ),
    )

    selected_node_ids_ranked = (
        ranked_nodes[:n_sensors]
    )

    selected_node_indices = sorted(
        node_index[node_id]
        for node_id
        in selected_node_ids_ranked
    )

    # --------------------------------------------------------
    # Edge betweenness centrality
    # --------------------------------------------------------

    raw_edge_scores = (
        nx.edge_betweenness_centrality(
            graph,
            normalized=True,
            weight=None,
        )
    )

    edge_scores = {}

    for edge, score in raw_edge_scores.items():

        # MultiGraph:
        # (u, v, key)
        if len(edge) != 3:
            raise RuntimeError(
                f"Unexpected edge key: {edge}"
            )

        link_id = str(edge[2])

        edge_scores[link_id] = float(score)

    missing_links = (
        set(link_ids)
        - set(edge_scores.keys())
    )

    if missing_links:
        raise RuntimeError(
            f"{network_name}: links missing from "
            f"centrality calculation: "
            f"{sorted(missing_links)}"
        )

    ranked_links = sorted(
        link_ids,
        key=lambda link_id: (
            -edge_scores[link_id],
            link_index[link_id],
        ),
    )

    selected_link_ids_ranked = (
        ranked_links[:n_sensors]
    )

    selected_link_indices = sorted(
        link_index[link_id]
        for link_id
        in selected_link_ids_ranked
    )

    # --------------------------------------------------------
    # Final sanity checks
    # --------------------------------------------------------

    if (
        len(selected_node_indices)
        != n_sensors
    ):
        raise RuntimeError(
            "Wrong number of node sensors"
        )

    if (
        len(set(selected_node_indices))
        != n_sensors
    ):
        raise RuntimeError(
            "Duplicate node sensor indices"
        )

    if (
        len(selected_link_indices)
        != n_sensors
    ):
        raise RuntimeError(
            "Wrong number of link sensors"
        )

    if (
        len(set(selected_link_indices))
        != n_sensors
    ):
        raise RuntimeError(
            "Duplicate link sensor indices"
        )

    # IDs corresponding to the final sorted EKF indices
    selected_node_ids = [
        node_ids[idx]
        for idx in selected_node_indices
    ]

    selected_link_ids = [
        link_ids[idx]
        for idx in selected_link_indices
    ]

    return {
        "network": network_name,

        "n_node_sensors": n_sensors,
        "n_link_sensors": n_sensors,

        "node_indices": selected_node_indices,
        "node_ids": selected_node_ids,

        "link_indices": selected_link_indices,
        "link_ids": selected_link_ids,

        "node_centrality": {
            node_id: float(
                node_scores[node_id]
            )
            for node_id
            in selected_node_ids_ranked
        },

        "link_centrality": {
            link_id: float(
                edge_scores[link_id]
            )
            for link_id
            in selected_link_ids_ranked
        },
    }


if __name__ == "__main__":

    results = {}

    for network_name in [
        "net1",
        "hanoi",
        "cydbp",
    ]:

        print()
        print("=" * 72)
        print(network_name.upper())
        print("=" * 72)

        placement = compute_placement(
            network_name
        )

        results[network_name] = placement

        print(
            "Node indices:",
            placement["node_indices"],
        )

        print(
            "Node IDs:",
            placement["node_ids"],
        )

        print(
            "Link indices:",
            placement["link_indices"],
        )

        print(
            "Link IDs:",
            placement["link_ids"],
        )

        print(
            "Unique node sensors:",
            len(set(
                placement["node_indices"]
            )),
        )

        print(
            "Unique link sensors:",
            len(set(
                placement["link_indices"]
            )),
        )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULT_DIR
        / "centrality_sensor_placements.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print()
    print("Saved:")
    print(output_path)
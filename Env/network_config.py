from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class NetworkConfig:
    name: str
    model_name: str
    scenario_stem: str
    surrogate_filename: str
    n_nodes: int
    n_links: int
    injection_node_id: str
    injection_pattern_id: str

    def get_scenario_stem(
        self,
        split: Optional[str] = None,
    ) -> str:
        if split is None:
            return self.scenario_stem

        if split not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(
                f"Unknown scenario split: {split}"
            )

        return f"{self.scenario_stem}_{split}"

    @property
    def observation_size(self) -> int:
        return (
            3 * self.n_nodes
            + 2 * self.n_links
        )

    # Observation:
    # pressures, flows, demands, node chlorine, link chlorine

    @property
    def pressure_slice(self) -> slice:
        return slice(
            0,
            self.n_nodes,
        )

    @property
    def flow_slice(self) -> slice:
        return slice(
            self.n_nodes,
            self.n_nodes + self.n_links,
        )

    @property
    def demand_slice(self) -> slice:
        return slice(
            self.n_nodes + self.n_links,
            2 * self.n_nodes + self.n_links,
        )

    @property
    def node_chlorine_slice(self) -> slice:
        return slice(
            2 * self.n_nodes + self.n_links,
            3 * self.n_nodes + self.n_links,
        )

    @property
    def link_chlorine_slice(self) -> slice:
        return slice(
            3 * self.n_nodes + self.n_links,
            3 * self.n_nodes + 2 * self.n_links,
        )

    @property
    def chlorine_slice(self) -> slice:
        return slice(
            self.node_chlorine_slice.start,
            self.link_chlorine_slice.stop,
        )

    # EKF state:
    # node chlorine, link chlorine, flows

    @property
    def state_node_chlorine_slice(self) -> slice:
        return slice(
            0,
            self.n_nodes,
        )

    @property
    def state_link_chlorine_slice(self) -> slice:
        return slice(
            self.n_nodes,
            self.n_nodes + self.n_links,
        )

    @property
    def state_flow_slice(self) -> slice:
        return slice(
            self.n_nodes + self.n_links,
            self.n_nodes + 2 * self.n_links,
        )


NETWORKS = {
    "hanoi": NetworkConfig(
        name="hanoi",
        model_name="Hanoi",
        scenario_stem=(
            "control_cl_injection_scenario-"
            "Net1=False_randDemand=True"
        ),
        surrogate_filename=(
            "hanoi_randDemand=True_surrogate.pt"
        ),
        n_nodes=32,
        n_links=34,
        injection_node_id="1",
        injection_pattern_id="my-chl-injection",
    ),
    "net1": NetworkConfig(
        name="net1",
        model_name="Net1",
        scenario_stem=(
            "control_cl_injection_scenario-"
            "Net1=True_randDemand=True"
        ),
        surrogate_filename=(
            "net1_randDemand=True_surrogate_multi5.pt"
        ),
        n_nodes=11,
        n_links=13,
        injection_node_id="9",
        injection_pattern_id="my-chl-injection",
    ),
    "cydbp": NetworkConfig(
        name="cydbp",
        model_name="CY-DBP",
        scenario_stem=(
            "control_cl_injection_scenario-"
            "CYDBP_randDemand=True"
        ),
        surrogate_filename=(
            "cydbp_randDemand=True_surrogate.pt"
        ),
        n_nodes=252,
        n_links=332,
        injection_node_id="T_Zone",
        injection_pattern_id="my-chl-inj-T_Zone",
    ),
}
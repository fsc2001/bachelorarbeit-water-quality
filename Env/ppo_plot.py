"""
This module plots PPO performance across different training seeds.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

FILES = {
    "Net1": "net1_ppo_final_seeded_ppo_20000_raw.csv",
    "Hanoi": "hanoi_ppo_final_seeded_ppo_20000_raw.csv",
    "CY-DBP": "cydbp_ppo_final_seeded_ppo_20000_raw.csv",
}

VARIANTS = {
    "ekf_random": "EKF Random",
    "ekf_centrality": "EKF Centrality",
    "oracle": "Oracle",
}

SPATIAL_FILES = {
    "Net1": "net1_ppo_final_seeded_ppo_20000_spatial.csv",
    "Hanoi": "hanoi_ppo_final_seeded_ppo_20000_spatial.csv",
    "CY-DBP": "cydbp_ppo_final_seeded_ppo_20000_spatial.csv",
}

def plot_spatial_performance():
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    for ax, (network_name, filename) in zip(
        axes,
        SPATIAL_FILES.items(),
    ):
        df = pd.read_csv(RESULTS_DIR / filename)

        for variant, label in VARIANTS.items():
            variant_data = df[
                df["variant"] == variant
            ]

            mean_per_node = (
                variant_data
                .groupby("node_index")["outside_range_fraction"]
                .mean()
            )

            ax.scatter(
                mean_per_node.index,
                mean_per_node.values,
                label=label,
                s=15,
            )

        ax.set_title(network_name)
        ax.set_ylabel("Outside-range fraction")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Node index")
    axes[0].legend()

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "ppo_spatial_performance_combined.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print(f"Saved figure to: {output_path}")


def main():
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    for ax, (network_name, filename) in zip(axes, FILES.items()):
        df = pd.read_csv(RESULTS_DIR / filename)

        for x, variant in enumerate(VARIANTS):
            values = df[
                df["variant"] == variant
            ]["outside_range_fraction"]

            ax.scatter(
                [x] * len(values),
                values,
                s=40,
            )

            ax.errorbar(
                x,
                values.mean(),
                yerr=values.std(),
                fmt="o",
                capsize=5,
            )

        ax.set_title(network_name)
        ax.set_ylabel("Outside-range fraction")
        ax.set_ylim(0, 1.05)

        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels(VARIANTS.values())

        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "ppo_seed_performance_combined.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print(f"Saved figure to: {output_path}")

    plot_spatial_performance()


if __name__ == "__main__":
    main()

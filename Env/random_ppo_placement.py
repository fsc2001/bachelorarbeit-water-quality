"""
This module selects representative random sensor placements for the PPO experiments.
"""

import json
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

SENSOR_COUNTS = {
    "net1": 3,
    "hanoi": 10,
    "cydbp": 75,
}


def select_placement(network_name, n_sensors):
    results_path = (
            RESULTS_DIR
            / f"{network_name}_ekf_random_sensors_validation.csv"
    )

    results = pd.read_csv(results_path)

    network_results = results[
        results["n_sensors"] == n_sensors
        ].copy()

    if network_results.empty:
        raise RuntimeError(
            f"No validation results found for {network_name} "
            f"with {n_sensors} sensors."
        )

    median_mae = network_results["all_mae"].median()

    network_results["distance_to_median"] = (
            network_results["all_mae"] - median_mae
    ).abs()

    selected = network_results.sort_values(
        ["distance_to_median", "seed"]
    ).iloc[0]

    return {
        "n_sensors": n_sensors,
        "seed": int(selected["seed"]),
        "validation_mae": float(selected["all_mae"]),
    }


def main():
    selected_placements = {}

    for network_name, n_sensors in SENSOR_COUNTS.items():
        selected = select_placement(
            network_name,
            n_sensors,
        )

        selected_placements[network_name] = selected

        print(
            f"{network_name}: seed {selected['seed']}, "
            f"MAE={selected['validation_mae']:.4f}"
        )

    output_path = RESULTS_DIR / "selected_random_ppo_placements.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            selected_placements,
            file,
            indent=2,
        )

    print(f"Saved selection to: {output_path}")


if __name__ == "__main__":
    main()

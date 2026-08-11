from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"

SENSOR_COUNTS = {
    "net1": 3,
    "hanoi": 10,
    "cydbp": 75,
}


for network_name, n_sensors in SENSOR_COUNTS.items():

    path = (
        RESULT_DIR
        / f"{network_name}_ekf_random_sensors_validation.csv"
    )

    df = pd.read_csv(path)

    # Only use the sensor count selected for PPO
    subset = df[
        df["n_sensors"] == n_sensors
    ].copy()

    if len(subset) == 0:
        raise RuntimeError(
            f"No results found for "
            f"{network_name}, sensors={n_sensors}"
        )

    mae_column = (
        "all_mae"
        if "all_mae" in subset.columns
        else "mae"
    )

    print("Available columns:")
    print(list(subset.columns))

    median_mae = subset[mae_column].median()

    subset["distance_to_median"] = (
            subset[mae_column] - median_mae
    ).abs()

    selected = subset.sort_values(
        [
            "distance_to_median",
            "seed",
        ]
    ).iloc[0]

    print()
    print("=" * 60)
    print(network_name.upper())
    print("=" * 60)

    print("Sensors:", n_sensors)
    print("Median MAE:", median_mae)
    print("Selected seed:", int(selected["seed"]))
    print("Selected MAE:", selected[mae_column])

    if "node_indices" in selected.index:
        print(
            "Node indices:",
            selected["node_indices"]
        )

    if "link_indices" in selected.index:
        print(
            "Link indices:",
            selected["link_indices"]
        )
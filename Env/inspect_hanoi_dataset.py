from pathlib import Path
import numpy as np
import inspect

from epyt_flow.simulation import ScadaData


ROOT = Path(__file__).resolve().parents[1]

scada_path = (
    ROOT
    / "NeuralSurrogateKalmanChlorineEstimation"
    / "data"
    / "hanoi_randDemand=True_training.epytflow_scada_data"
)

scada = ScadaData.load_from_file(str(scada_path))

print("Verfügbare Methoden mit 'quality':")
for name in dir(scada):
    if "quality" in name.lower():
        print(" ", name)

print("\nSensorConfig:")
print(scada.sensor_config)

checks = [
    "get_data_nodes_quality",
    "get_data_links_quality",
    "get_data_pressures",
    "get_data_demands",
    "get_data_flows",
]

for method_name in checks:
    print(f"\n--- {method_name} ---")

    if not hasattr(scada, method_name):
        print("Methode existiert nicht.")
        continue

    method = getattr(scada, method_name)

    try:
        values = np.asarray(method())

        print("shape:", values.shape)
        print(
            "min/max:",
            float(values.min()),
            float(values.max())
        )
        print("erste Zeile:")
        print(values[0])

    except Exception as error:
        print("FEHLER:", repr(error))


# Ab hier keine Einrückung mehr

print("\n--- Raw link quality ---")

raw_link_quality = np.asarray(
    scada.link_quality_data_raw
)

print("shape:", raw_link_quality.shape)
print(
    "min/max:",
    float(raw_link_quality.min()),
    float(raw_link_quality.max())
)
print("erste Zeile:")
print(raw_link_quality[0])


print("\n--- Zeitliche Variation ---")

link_quality = np.asarray(
    scada.get_data_links_quality()
)

print(
    "Maximale Änderung zur ersten Zeile:",
    float(
        np.max(
            np.abs(
                link_quality - link_quality[0]
            )
        )
    )
)

print(
    "Standardabweichung je Link, min/max:",
    float(
        np.std(
            link_quality,
            axis=0
        ).min()
    ),
    float(
        np.std(
            link_quality,
            axis=0
        ).max()
    )
)

print("\nErste gegen letzte Zeile:")
print("erste:", link_quality[0])
print("letzte:", link_quality[-1])


print("\n--- Source von get_data_links_quality ---")

print(
    inspect.getsource(
        type(scada).get_data_links_quality
    )
)
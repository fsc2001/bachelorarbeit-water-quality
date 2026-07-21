import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from epanet_plus import EPyT
from epyt_flow.simulation import EpanetConstants, ScadaData


ROOT = Path(__file__).resolve().parents[1]

REPO = (
    ROOT
    / "NeuralSurrogateKalmanChlorineEstimation"
)

DATA_DIR = REPO / "data"

SCENARIO_BASE = (
    "control_cl_injection_scenario-"
    "Net1=False_randDemand=True"
)

SCENARIO_INP = DATA_DIR / f"{SCENARIO_BASE}.inp"
SCENARIO_CONFIG = (
    DATA_DIR
    / f"{SCENARIO_BASE}.epytflow_scenario_config"
)

DATASET_NAMES = [
    "hanoi_randDemand=True_training",
    "hanoi_randDemand=True_validation",
    "hanoi_randDemand=True_test",
]


# create_data.py verwendet relative Pfade wie "data".
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from create_data import create_data_set


def verify_scenario() -> None:
    """
    Prüft vor der Datenerzeugung, dass die reparierte
    Hanoi-INP weiterhin vollständig und funktionsfähig ist.
    """
    if not SCENARIO_INP.exists():
        raise FileNotFoundError(
            f"Szenario-INP fehlt:\n{SCENARIO_INP}"
        )

    if not SCENARIO_CONFIG.exists():
        raise FileNotFoundError(
            f"Szenariokonfiguration fehlt:\n"
            f"{SCENARIO_CONFIG}"
        )

    data = SCENARIO_INP.read_bytes()
    lines = data.splitlines()

    end_lines = [
        line_number
        for line_number, line in enumerate(
            lines,
            start=1
        )
        if line.strip().upper().startswith(b"[END")
    ]

    if end_lines != [len(lines)]:
        raise RuntimeError(
            "Die Hanoi-INP ist wieder beschädigt.\n"
            f"[END]-Zeilen: {end_lines}\n"
            f"Gesamtzahl Zeilen: {len(lines)}"
        )

    api = EPyT(str(SCENARIO_INP))

    try:
        pattern_idx = api.getpatternindex(
            "my-chl-injection"
        )

        node_idx = api.get_node_idx("1")

        source_pattern = api.getnodevalue(
            node_idx,
            EpanetConstants.EN_SOURCEPAT
        )

        source_quality = api.getnodevalue(
            node_idx,
            EpanetConstants.EN_SOURCEQUAL
        )

        quality_info = api.getqualinfo()

    finally:
        api.close()

    if pattern_idx <= 0:
        raise RuntimeError(
            "Das Injektionspattern wurde nicht geladen."
        )

    if int(source_pattern) != int(pattern_idx):
        raise RuntimeError(
            "Source und Injektionspattern stimmen "
            "nicht überein."
        )

    print("Szenario erfolgreich geprüft:")
    print("  [END]-Zeile:", end_lines[0])
    print("  Quality:", quality_info)
    print("  Pattern-Index:", pattern_idx)
    print("  SourcePattern:", source_pattern)
    print("  SourceQual:", source_quality)


def backup_old_files() -> None:
    """
    Verschiebt alte Hanoi-Daten und das alte Surrogate
    in einen Backup-Ordner, statt sie endgültig zu löschen.
    """
    files_to_backup = []

    for dataset_name in DATASET_NAMES:
        files_to_backup.extend([
            DATA_DIR
            / f"{dataset_name}.epytflow_scada_data",

            DATA_DIR
            / f"{dataset_name}.npz",
        ])

    files_to_backup.append(
        DATA_DIR
        / "hanoi_randDemand=True_surrogate.pt"
    )

    existing_files = [
        path
        for path in files_to_backup
        if path.exists()
    ]

    if not existing_files:
        print("\nKeine alten Datensätze gefunden.")
        return

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_dir = (
        DATA_DIR
        / f"backup_hanoi_before_fix_{timestamp}"
    )

    backup_dir.mkdir(parents=True)

    print("\nSichere alte Dateien nach:")
    print(backup_dir)

    for path in existing_files:
        destination = backup_dir / path.name

        print("  Verschiebe:", path.name)

        shutil.move(
            str(path),
            str(destination)
        )


def inspect_dataset(dataset_name: str) -> None:
    """
    Prüft Dimensionen, Wertebereiche und NaN/Inf-Werte
    eines neu erstellten Datensatzes.
    """
    scada_path = (
        DATA_DIR
        / f"{dataset_name}.epytflow_scada_data"
    )

    actions_path = (
        DATA_DIR
        / f"{dataset_name}.npz"
    )

    if not scada_path.exists():
        raise FileNotFoundError(
            f"SCADA-Datei wurde nicht erzeugt:\n"
            f"{scada_path}"
        )

    if not actions_path.exists():
        raise FileNotFoundError(
            f"Action-Datei wurde nicht erzeugt:\n"
            f"{actions_path}"
        )

    scada_data = ScadaData.load_from_file(
        str(scada_path)
    )

    actions = np.asarray(
        np.load(
            actions_path,
            allow_pickle=False
        )["control_actions"]
    )

    nodes_quality = np.asarray(
        scada_data.get_data_nodes_quality()
    )

    links_quality = np.asarray(
        scada_data.get_data_links_quality()
    )

    flows = np.asarray(
        scada_data.get_data_flows()
    )

    arrays = {
        "Actions": actions,
        "Node chlorine": nodes_quality,
        "Link chlorine": links_quality,
        "Flows": flows,
    }

    for name, values in arrays.items():
        if values.size == 0:
            raise RuntimeError(
                f"{dataset_name}: {name} ist leer."
            )

        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"{dataset_name}: {name} enthält "
                "NaN oder unendliche Werte."
            )

    print(f"\nPrüfung {dataset_name}:")
    print("  Actions shape:", actions.shape)
    print(
        "  Actions min/max:",
        float(actions.min()),
        float(actions.max())
    )

    print(
        "  Node chlorine shape:",
        nodes_quality.shape
    )
    print(
        "  Node chlorine min/max:",
        float(nodes_quality.min()),
        float(nodes_quality.max())
    )

    print(
        "  Link chlorine shape:",
        links_quality.shape
    )
    print(
        "  Link chlorine min/max:",
        float(links_quality.min()),
        float(links_quality.max())
    )

    print("  Flows shape:", flows.shape)
    print(
        "  Flows min/max:",
        float(flows.min()),
        float(flows.max())
    )

    if np.ptp(actions) < 1.0:
        raise RuntimeError(
            f"{dataset_name}: Die Actions variieren "
            "kaum. Das wäre für das Surrogate ungeeignet."
        )

    if np.ptp(nodes_quality) < 0.1:
        raise RuntimeError(
            f"{dataset_name}: Die Chlorwerte an den "
            "Knoten variieren kaum. Die Injektion könnte "
            "erneut nicht funktionieren."
        )


if __name__ == "__main__":
    verify_scenario()
    backup_old_files()

    for dataset_name in DATASET_NAMES:
        print("\n" + "=" * 70)
        print("Erzeuge:", dataset_name)
        print("=" * 70)

        create_data_set(
            use_net1=False,
            randomized_demands=True,
            file_out=dataset_name,
            path_out="data",
        )

        inspect_dataset(dataset_name)

    print("\n" + "=" * 70)
    print("Alle Hanoi-Datensätze wurden neu erzeugt.")
    print("=" * 70)
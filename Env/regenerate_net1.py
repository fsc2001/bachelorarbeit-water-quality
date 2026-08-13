import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "NeuralSurrogateKalmanChlorineEstimation"
DATA_DIR = REPO / "data"

FILE_BASE = (
    "control_cl_injection_scenario-"
    "Net1=True_randDemand=True"
)

INP_PATH = DATA_DIR / f"{FILE_BASE}.inp"
CONFIG_PATH = DATA_DIR / f"{FILE_BASE}.epytflow_scenario_config"
REPORT_PATH = DATA_DIR / f"{FILE_BASE}.inp.rpt"
RUNTIME_PATH = DATA_DIR / f"{FILE_BASE}_runtime.inp"


# create_data.py verwendet relative Pfade wie "data".
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from epyt_flow.simulation import ScenarioSimulator
from create_data import create_leakdb_scenario


# Originalmethode sichern.
_original_save_to_epanet_file = (
    ScenarioSimulator.save_to_epanet_file
)


def _save_without_broken_report_rewrite(
    self,
    inp_file_path: str,
    msx_file_path: str = None,
    export_sensor_config: bool = True,
    undo_system_events: bool = True,
) -> None:
    """
    Ruft den normalen EPyT-Flow-Export auf, überspringt aber
    die unter Windows fehlerhafte Nachbearbeitung von [REPORT].

    Die SensorConfig wird weiterhin separat in der
    epytflow_scenario_config gespeichert.
    """
    return _original_save_to_epanet_file(
        self,
        inp_file_path=inp_file_path,
        msx_file_path=msx_file_path,
        export_sensor_config=False,
        undo_system_events=undo_system_events,
    )


def inspect_export(path: Path) -> None:
    data = path.read_bytes()
    lines = data.splitlines()

    end_lines = [
        line_number
        for line_number, line in enumerate(lines, start=1)
        if line.strip().upper().startswith(b"[END")
    ]

    print("\nDateigröße:", len(data), "Bytes")
    print("Gesamtzahl Zeilen:", len(lines))
    print("[END]-Zeilen:", end_lines)
    print(
        "Pattern-Vorkommen:",
        data.count(b"my-chl-injection"),
    )

    if end_lines != [len(lines)]:
        raise RuntimeError(
            "Export ist weiterhin beschädigt. "
            f"[END]-Zeilen: {end_lines}, "
            f"Gesamtzeilen: {len(lines)}"
        )


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    for path in [
        INP_PATH,
        CONFIG_PATH,
        REPORT_PATH,
        RUNTIME_PATH,
    ]:
        if path.exists():
            print("Lösche:", path)
            path.unlink()

    print("\nExistiert die alte INP noch?", INP_PATH.exists())

    # Temporärer Patch nur für diesen Python-Prozess.
    ScenarioSimulator.save_to_epanet_file = (
        _save_without_broken_report_rewrite
    )

    try:
        print("\nErzeuge Net1-Szenario neu ...")

        create_leakdb_scenario(
            use_net1=True,
            randomized_demands=True,
        )

    finally:
        # Originalmethode wiederherstellen.
        ScenarioSimulator.save_to_epanet_file = (
            _original_save_to_epanet_file
        )

    print("\nNeu erzeugte Datei:")
    print(INP_PATH.resolve())
    print("Existiert:", INP_PATH.exists())

    if not INP_PATH.exists():
        raise RuntimeError(
            "Es wurde keine neue INP erzeugt."
        )

    inspect_export(INP_PATH)

    print("\nExport ist strukturell korrekt.")
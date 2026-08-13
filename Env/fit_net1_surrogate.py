import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "NeuralSurrogateKalmanChlorineEstimation"
DATA_DIR = REPO / "data"

os.chdir(REPO)
sys.path.insert(0, str(REPO))

from fit_surrogates import fit_surrogate


SCADA_PATH = (
    DATA_DIR
    / "net1_randDemand=True_training.epytflow_scada_data"
)

ACTIONS_PATH = (
    DATA_DIR
    / "net1_randDemand=True_training.npz"
)

OUTPUT_PATH = (
    DATA_DIR
    / "net1_randDemand=True_surrogate.pt"
)


def main() -> None:
    if not SCADA_PATH.exists():
        raise FileNotFoundError(
            f"SCADA-Datensatz fehlt: {SCADA_PATH}"
        )

    if not ACTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Action-Datensatz fehlt: {ACTIONS_PATH}"
        )

    print("=" * 72)
    print("TRAINIERE NET1-SURROGAT")
    print("=" * 72)

    print("SCADA:")
    print(SCADA_PATH)

    print("Actions:")
    print(ACTIONS_PATH)

    print("Ausgabe:")
    print(OUTPUT_PATH)

    fit_surrogate(
        net_desc="Net1",
        scada_file_in=str(SCADA_PATH),
        control_actions_file_in=str(ACTIONS_PATH),
        file_out=str(OUTPUT_PATH),
    )

    pickle_path = Path(
        str(OUTPUT_PATH) + ".pickle"
    )

    if OUTPUT_PATH.exists():
        saved_path = OUTPUT_PATH
    elif pickle_path.exists():
        saved_path = pickle_path
    else:
        raise RuntimeError(
            "Das Surrogattraining wurde beendet, "
            "aber es wurde keine Modelldatei gefunden."
        )

    print("\nTraining abgeschlossen.")
    print("Gespeichertes Modell:")
    print(saved_path)


if __name__ == "__main__":
    main()
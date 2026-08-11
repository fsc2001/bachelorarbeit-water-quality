import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REPO = (
    ROOT
    / "NeuralSurrogateKalmanChlorineEstimation"
)

DATA_DIR = REPO / "data"

os.chdir(REPO)
sys.path.insert(0, str(REPO))

from create_data import create_data_set


RANDOMIZED_DEMANDS = True
OVERWRITE = False


def create_dataset(split: str) -> None:
    file_stem = (
        f"net1_randDemand="
        f"{RANDOMIZED_DEMANDS}_{split}"
    )

    scada_path = (
        DATA_DIR
        / f"{file_stem}.epytflow_scada_data"
    )

    actions_path = DATA_DIR / f"{file_stem}.npz"

    if (
        not OVERWRITE
        and scada_path.exists()
        and actions_path.exists()
    ):
        print(
            f"Überspringe {split}: "
            "Dateien existieren bereits."
        )
        return

    print("\n" + "=" * 72)
    print(f"ERZEUGE NET1-DATENSATZ: {split}")
    print("=" * 72)

    create_data_set(
        use_net1=True,
        randomized_demands=RANDOMIZED_DEMANDS,
        file_out=file_stem,
        path_out="data",
    )

    if not scada_path.exists():
        raise RuntimeError(
            f"SCADA-Datei fehlt: {scada_path}"
        )

    if not actions_path.exists():
        raise RuntimeError(
            f"Action-Datei fehlt: {actions_path}"
        )

    print("Erzeugt:")
    print(scada_path)
    print(actions_path)


def main() -> None:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in (
        "training",
        "validation",
        "test",
    ):
        create_dataset(split)

    print("\nAlle Net1-Datensätze sind vorhanden.")


if __name__ == "__main__":
    main()
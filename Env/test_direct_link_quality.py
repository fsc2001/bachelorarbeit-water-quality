from pathlib import Path

import numpy as np

from epanet_plus import EPyT
from epyt_flow.simulation import EpanetConstants


ROOT = Path(__file__).resolve().parents[1]

inp_path = (
    ROOT
    / "NeuralSurrogateKalmanChlorineEstimation"
    / "data"
    / "control_cl_injection_scenario-Net1=False_randDemand=True.inp"
)

api = EPyT(str(inp_path))

try:
    print("Quality info:", api.getqualinfo())

    print("\nKonstanten:")
    print("EN_QUALITY:", EpanetConstants.EN_QUALITY)
    print(
        "EN_LINKQUAL:",
        getattr(
            EpanetConstants,
            "EN_LINKQUAL",
            "NICHT VORHANDEN"
        )
    )

    for constant_name in [
        "EN_QUALITY",
        "EN_LINKQUAL",
    ]:
        if not hasattr(EpanetConstants, constant_name):
            continue

        constant = getattr(
            EpanetConstants,
            constant_name
        )

        try:
            values = np.asarray(
                api.getlinkvalues(constant),
                dtype=float
            )

            print(f"\n{constant_name}:")
            print("shape:", values.shape)
            print(
                "min/max:",
                float(values.min()),
                float(values.max())
            )
            print("erste Werte:", values[:10])

        except Exception as error:
            print(
                f"\n{constant_name}: FEHLER",
                repr(error)
            )

finally:
    api.close()
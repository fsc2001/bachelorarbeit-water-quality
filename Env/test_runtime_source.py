from pathlib import Path

from epanet_plus import EPyT
from epyt_flow.simulation import EpanetConstants


ROOT = Path(__file__).resolve().parents[1]

inp_path = (
    ROOT
    / "NeuralSurrogateKalmanChlorineEstimation"
    / "data"
    / "control_cl_injection_scenario-Net1=False_randDemand=True.inp"
)

PATTERN_ID = "my-chl-injection"
NODE_ID = "1"


api = EPyT(str(inp_path))

try:
    print("Quality-Information:", api.getqualinfo())

    # Das Pattern fehlt nach dem erneuten Laden der INP.
    try:
        pattern_idx = api.getpatternindex(PATTERN_ID)
        print("Pattern war bereits vorhanden:", pattern_idx)

    except RuntimeError:
        print("Pattern fehlt – wird im Arbeitsspeicher neu angelegt.")

        api.add_pattern(
            PATTERN_ID,
            [1.0]
        )

        pattern_idx = api.getpatternindex(PATTERN_ID)

    print("Pattern-Index:", pattern_idx)

    node_idx = api.get_node_idx(NODE_ID)

    # Diese Aufrufe erzeugen das Source-Objekt automatisch,
    # falls es noch nicht existiert.
    api.setnodevalue(
        node_idx,
        EpanetConstants.EN_SOURCETYPE,
        EpanetConstants.EN_CONCEN
    )

    api.setnodevalue(
        node_idx,
        EpanetConstants.EN_SOURCEQUAL,
        1.0
    )

    api.setnodevalue(
        node_idx,
        EpanetConstants.EN_SOURCEPAT,
        pattern_idx
    )

    print(
        "SourceType:",
        api.getnodevalue(
            node_idx,
            EpanetConstants.EN_SOURCETYPE
        )
    )

    print(
        "SourceQual:",
        api.getnodevalue(
            node_idx,
            EpanetConstants.EN_SOURCEQUAL
        )
    )

    print(
        "SourcePattern:",
        api.getnodevalue(
            node_idx,
            EpanetConstants.EN_SOURCEPAT
        )
    )

finally:
    api.close()
from pathlib import Path

from epanet_plus import EPyT
from epyt_flow.simulation import EpanetConstants


ROOT = Path(__file__).resolve().parents[1]

inp_path = (
    ROOT
    / "NeuralSurrogateKalmanChlorineEstimation"
    / "data"
    / (
        "control_cl_injection_scenario-"
        "Net1=False_randDemand=True.inp"
    )
)

print("Geprüfte Datei:")
print(inp_path.resolve())

data = inp_path.read_bytes()
lines = data.splitlines()

end_lines = [
    line_number
    for line_number, line in enumerate(lines, start=1)
    if line.strip().upper().startswith(b"[END")
]

print("[END]-Zeilen:", end_lines)
print("Gesamtzahl Zeilen:", len(lines))
print(
    "Pattern-Vorkommen:",
    data.count(b"my-chl-injection")
)

api = EPyT(str(inp_path))

try:
    pattern_idx = api.getpatternindex(
        "my-chl-injection"
    )

    node_idx = api.get_node_idx("1")

    print("Quality:", api.getqualinfo())
    print("Pattern-Index:", pattern_idx)

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
from pathlib import Path

from epanet_plus import EPyT
from epyt_flow.simulation import EpanetConstants


ROOT = Path(__file__).resolve().parents[1]

original_path = (
    ROOT
    / "NeuralSurrogateKalmanChlorineEstimation"
    / "data"
    / "control_cl_injection_scenario-Net1=False_randDemand=True.inp"
)

test_path = original_path.with_name(
    "control_cl_injection_scenario-"
    "Net1=False_randDemand=True_pattern_first.inp"
)


data = original_path.read_bytes()

old_id = b"my-chl-injection"
new_id = b"CLPAT"

# Gleiche Bytelänge beibehalten.
padded_new_id = (
    new_id
    + b" " * (len(old_id) - len(new_id))
)

data = data.replace(old_id, padded_new_id)

newline = b"\r\n" if b"\r\n" in data else b"\n"

# Direkt hinter [PATTERNS] ein zusätzliches CLPAT einfügen.
patterns_header = b"[PATTERNS]"
header_start = data.index(patterns_header)
insert_position = data.index(
    newline,
    header_start
) + len(newline)

new_pattern_line = (
    b" CLPAT                                  1.0000"
    + newline
)

data = (
    data[:insert_position]
    + new_pattern_line
    + data[insert_position:]
)

test_path.write_bytes(data)

print("Testdatei:", test_path)


for use_project in [False, True]:
    print("\nuse_project =", use_project)

    try:
        api = EPyT(
            str(test_path),
            use_project=use_project
        )

        try:
            pattern_ids = api.get_all_patterns_id()

            print("Anzahl Patterns:", len(pattern_ids))
            print("CLPAT vorhanden:", "CLPAT" in pattern_ids)
            print(
                "CLPAT-Index:",
                api.getpatternindex("CLPAT")
            )

            node_idx = api.get_node_idx("1")

            print(
                "SourcePattern:",
                api.getnodevalue(
                    node_idx,
                    EpanetConstants.EN_SOURCEPAT
                )
            )

            print(
                "SourceQual:",
                api.getnodevalue(
                    node_idx,
                    EpanetConstants.EN_SOURCEQUAL
                )
            )

        finally:
            api.close()

    except RuntimeError as error:
        print("FEHLER:", error)
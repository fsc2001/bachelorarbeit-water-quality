import inspect

import epyt_flow
from epyt_flow.simulation import EpanetConstants, ScenarioSimulator


print("EPyT-Flow geladen aus:")
print(epyt_flow.__file__)

print("\nScenarioSimulator geladen aus:")
print(inspect.getfile(ScenarioSimulator))

print("\nKonstanten:")
print("EN_QUALITY:", EpanetConstants.EN_QUALITY)
print(
    "EN_LINKQUAL:",
    getattr(EpanetConstants, "EN_LINKQUAL", "NICHT VORHANDEN")
)

print("\nZeilen mit quality_link_data / getlinkvalues:")

source = inspect.getsource(ScenarioSimulator)

for line_number, line in enumerate(
    source.splitlines(),
    start=1
):
    lower = line.lower()

    if (
        "quality_link_data" in lower
        or "getlinkvalues" in lower
        or "link_quality" in lower
    ):
        print(f"{line_number}: {line}")
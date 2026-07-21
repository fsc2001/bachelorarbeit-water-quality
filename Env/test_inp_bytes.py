from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

files = [
    (
        ROOT
        / "NeuralSurrogateKalmanChlorineEstimation"
        / "data"
        / (
            "control_cl_injection_scenario-"
            "Net1=False_randDemand=True.inp"
        )
    ),
    (
        ROOT
        / "NeuralSurrogateKalmanChlorineEstimation"
        / "data"
        / (
            "control_cl_injection_scenario-"
            "Net1=False_randDemand=True_runtime.inp"
        )
    ),
]


def line_number(data: bytes, position: int) -> int:
    return data.count(b"\n", 0, position) + 1


for path in files:
    print("\n" + "=" * 70)
    print(path)

    if not path.exists():
        print("Datei existiert nicht.")
        continue

    data = path.read_bytes()

    print("Dateigröße:", len(data), "Bytes")

    # Steuerzeichen außer Tab, LF und CR suchen.
    unusual = {}

    for position, byte_value in enumerate(data):
        if byte_value < 32 and byte_value not in (9, 10, 13):
            unusual.setdefault(byte_value, []).append(position)

    print("\nUngewöhnliche Steuerzeichen:")

    if not unusual:
        print("Keine gefunden.")
    else:
        for byte_value, positions in unusual.items():
            print(
                f"0x{byte_value:02X}:",
                f"{len(positions)} Vorkommen"
            )

            for position in positions[:5]:
                print(
                    "  Position:",
                    position,
                    "Zeile:",
                    line_number(data, position),
                    "Umgebung:",
                    repr(
                        data[
                            max(0, position - 40):
                            position + 40
                        ]
                    )
                )

    print("\nWichtige Positionen:")

    for marker in [
        b"[PATTERNS]",
        b"my-chl-injection",
        b"[SOURCES]",
        b"[OPTIONS]",
        b"[END]",
    ]:
        positions = []
        start = 0

        while True:
            position = data.find(marker, start)

            if position == -1:
                break

            positions.append(
                (
                    position,
                    line_number(data, position)
                )
            )
            start = position + 1

        print(marker, positions)

    print("\nAlle Zeilen, die mit [END beginnen:")

    for number, line in enumerate(
        data.splitlines(),
        start=1
    ):
        if line.strip().upper().startswith(b"[END"):
            print(number, repr(line))
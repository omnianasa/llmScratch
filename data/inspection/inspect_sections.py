from pathlib import Path


path = Path("../one_thousand_and_one_nights_raw.txt")

text = path.read_text(encoding="utf-8")
lines = text.splitlines()


def show_range(start, end):
    print("\n" + "=" * 80)
    print(f"LINES {start} → {end}")
    print("=" * 80)

    for i in range(start, min(end, len(lines))):
        line = lines[i].strip()

        if line:
            print(f"{i:6}: {line}")


# Important areas from the inspection
show_range(950, 1100)
show_range(1200, 1350)
show_range(1350, 1500)
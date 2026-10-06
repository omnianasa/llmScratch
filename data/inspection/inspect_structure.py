from pathlib import Path
from collections import Counter


path = Path("../one_thousand_and_one_nights_raw.txt")

text = path.read_text(encoding="utf-8")
lines = text.splitlines()


# ============================================================
# 1. MOST COMMON LINES
# ============================================================

print("=" * 70)
print("MOST COMMON NON-EMPTY LINES")
print("=" * 70)

clean_lines = [
    line.strip()
    for line in lines
    if line.strip()
]

counts = Counter(clean_lines)

for line, count in counts.most_common(30):
    if count > 2:
        print(f"{count:4}x  {line[:120]}")


# ============================================================
# 2. IMPORTANT STRUCTURAL WORDS
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANT STRUCTURAL LINES")
print("=" * 70)

patterns = [
    "PREFACE",
    "INTRODUCTION",
    "CONTENTS",
    "TABLE OF CONTENTS",
    "CHAPTER",
    "VOLUME",
    "INDEX",
    "NOTES",
    "THE END",
    "NIGHT",
    "NIGHTS",
]

found = []

for i, line in enumerate(lines):

    stripped = line.strip()

    if not stripped:
        continue

    upper = stripped.upper()

    for pattern in patterns:

        if pattern in upper:
            found.append((i, stripped))
            break


# Print only first 100 matches
for i, line in found[:100]:
    print(f"{i:6}: {line[:150]}")

print(f"\nTotal structural matches: {len(found)}")


# ============================================================
# 3. VERY SHORT LINES
# ============================================================

print("\n" + "=" * 70)
print("VERY SHORT LINES")
print("=" * 70)

short_lines = []

for i, line in enumerate(lines):

    stripped = line.strip()

    if 1 <= len(stripped) <= 15:
        short_lines.append((i, stripped))

for i, line in short_lines[:100]:
    print(f"{i:6}: {repr(line)}")

print(f"\nTotal very short lines: {len(short_lines)}")
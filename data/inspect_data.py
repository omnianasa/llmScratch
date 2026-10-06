from pathlib import Path

path = Path("one_thousand_and_one_nights_raw.txt")

text = path.read_text(encoding="utf-8")

print("Data inspection:")
print(f"Total characters: {len(text):,}")
print(f"Total lines: {len(text.splitlines()):,}")
print(f"Total words: {len(text.split()):,}")

print("\nFirst 500 characters:")
print(text[:500])

print("\nLast 500 characters:")
print(text[-500:])


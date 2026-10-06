from pathlib import Path

path = Path("processed/one_thousand_and_one_nights_clean.txt")

text = path.read_text(encoding="utf-8")

print("=" * 70)
print("CLEAN DATASET INSPECTION")
print("=" * 70)

print(f"Characters: {len(text):,}")
print(f"Words:      {len(text.split()):,}")
print(f"Paragraphs:  {len(text.split(chr(10) + chr(10))):,}")

print("\n" + "=" * 70)
print("FIRST 3000 CHARACTERS")
print("=" * 70)

print(text[:3000])

print("\n" + "=" * 70)
print("LAST 3000 CHARACTERS")
print("=" * 70)

print(text[-3000:])
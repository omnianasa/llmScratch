from pathlib import Path
import numpy as np


TOKEN_FILE = Path("../tokenization/data/tokenized/corpus.npy")
OUTPUT_DIR = Path("../tokenization/data/tokenized")

TRAIN_RATIO = 0.90

print("=" * 70)
print("PREPARING TRAIN / VALIDATION DATA")
print("=" * 70)

tokens = np.load(TOKEN_FILE)

print(f"Total tokens: {len(tokens):,}")
print(f"Dtype: {tokens.dtype}")
split_index = int(len(tokens) * TRAIN_RATIO)
train_tokens = tokens[:split_index]
val_tokens = tokens[split_index:]

train_file = OUTPUT_DIR / "train.npy"
val_file = OUTPUT_DIR / "val.npy"

np.save(train_file, train_tokens)
np.save(val_file, val_tokens)


# report the results 

print()
print(f"Train tokens: {len(train_tokens):,}")
print(f"Validation tokens: {len(val_tokens):,}")

print()
print(f"Train ratio: {len(train_tokens) / len(tokens):.2%}")
print(f"Validation ratio: {len(val_tokens) / len(tokens):.2%}")

print()
print(f"Train file: {train_file}")
print(f"Validation file: {val_file}")

print("=" * 70)
print("DONE")
print("=" * 70)
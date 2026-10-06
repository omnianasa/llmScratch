from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import GPTDataset


CONTEXT_LENGTH = 128
STRIDE = 64
BATCH_SIZE = 32


train_file = Path("../tokenization/data/tokenized/train.npy")
val_file = Path("../tokenization/data/tokenized/val.npy")

train_dataset = GPTDataset(
    train_file,
    context_length=CONTEXT_LENGTH,
    stride=STRIDE,
)

val_dataset = GPTDataset(
    val_file,
    context_length=CONTEXT_LENGTH,
    stride=STRIDE,
)

# data loaders for training and validation

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# a small test to check that the dataloader is working correctly

if __name__ == "__main__":

    print("=" * 70)
    print("DATALOADER TEST")
    print("=" * 70)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Batch size: {BATCH_SIZE}")

    # Get one batch
    x, y = next(iter(train_loader))

    print("\nFirst batch")
    print("-" * 70)

    print("X shape:", x.shape)
    print("Y shape:", y.shape)

    print("\nFirst sample X:")
    print(x[0, :10])

    print("\nFirst sample Y:")
    print(y[0, :10])

    print("\nShift check:")
    print("Y[0, 0] == X[0, 1]:", y[0, 0] == x[0, 1])

    print("\nDtypes:")
    print("X dtype:", x.dtype)
    print("Y dtype:", y.dtype)

    print("=" * 70)
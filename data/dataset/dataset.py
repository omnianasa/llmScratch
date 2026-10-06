from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    def __init__(
        self,
        token_file,
        context_length=128,
        stride=64,
    ):
        self.tokens = np.load(token_file)
        self.context_length = context_length
        self.stride = stride

    def __len__(self):
        return (
            len(self.tokens) - self.context_length
        ) // self.stride

    def __getitem__(self, index):
        start = index * self.stride
        end = start + self.context_length

        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]

        return x.astype(np.int64), y.astype(np.int64)


if __name__ == "__main__":

    train_file = Path("../tokenization/data/tokenized/train.npy")

    dataset = GPTDataset(
        train_file,
        context_length=128,
        stride=64,
    )

    print("=" * 70)
    print("SLIDING WINDOW DATASET TEST")
    print("=" * 70)

    print(f"Total train tokens: {len(dataset.tokens):,}")
    print(f"Context length:     {dataset.context_length}")
    print(f"Stride:             {dataset.stride}")
    print(f"Number of samples:  {len(dataset):,}")

    print()

    # Test first sample
    x, y = dataset[0]

    print("Sample 0")
    print("-" * 70)

    print("X shape:", x.shape)
    print("Y shape:", y.shape)

    print()
    print("First 10 X tokens:")
    print(x[:10])

    print()
    print("First 10 Y tokens:")
    print(y[:10])

    print()
    print("Check shifting:")

    print("X:", x[:10])
    print("Y:", y[:10])

    print()
    print("Y[0] == X[1]:", y[0] == x[1])

    # Test second sample
    x2, y2 = dataset[1]

    print()
    print("Sample 1")
    print("-" * 70)

    print("First 10 X tokens:")
    print(x2[:10])

    print("First 10 Y tokens:")
    print(y2[:10])

    print()
    print("X[64:] starts after first window?")
    print("Sample 0 start:", dataset[0][0][0])
    print("Sample 1 start:", dataset[1][0][0])

    print("=" * 70)
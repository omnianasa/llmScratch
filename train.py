import torch

from torch.utils.data import DataLoader

from data.dataset.dataset import GPTDataset

from model.gpt import GPTModel
from model.loss import GPTLoss

from trainer import Trainer


VOCAB_SIZE = 8000
CONTEXT_LENGTH = 128

EMBEDDING_DIM = 256
NUM_HEADS = 4
HIDDEN_DIM = 1024
NUM_LAYERS = 4

DROPOUT = 0.1

BATCH_SIZE = 32

LEARNING_RATE = 3e-4

EPOCHS = 1

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


train_dataset = GPTDataset(
    "data/tokenization/data/tokenized/train.npy",
    context_length=CONTEXT_LENGTH,
    stride=64
)

val_dataset = GPTDataset(
    "data/tokenization/data/tokenized/val.npy",
    context_length=CONTEXT_LENGTH,
    stride=64
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


model = GPTModel(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

model = model.to(device)

loss_fn = GPTLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device
)


# training loop

trainer.fit(epochs=EPOCHS)
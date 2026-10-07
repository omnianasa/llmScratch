import torch


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):

        self.model.train()
        total_loss = 0.0
        for x, targets in self.train_loader:

            x = x.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.loss_fn(logits,targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():

            for x, targets in self.val_loader:
                x = x.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(x)
                loss = self.loss_fn(
                    logits,
                    targets
                )

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def fit(self, epochs):

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"| Train Loss: {train_loss:.4f} "
                f"| Val Loss: {val_loss:.4f}"
            )
import torch.nn as nn


class GPTLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):

        logits = logits.transpose(1, 2)
        return self.loss(logits,targets)
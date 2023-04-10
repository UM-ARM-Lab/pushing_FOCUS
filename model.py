"""
This adapts the source model by loading the previously trained checkpoint,
then training more with a smaller learning rate on the target dataset.
"""
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam


def phi(j):
    return 10 * j


class DynamicsNetwork(pl.LightningModule):
    def __init__(self, method: Optional[str] = None, input_dim=80, output_dim=60):
        super().__init__()
        self.gamma = 0.02
        self.method = method
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim))

    def forward(self, inputs):
        outputs = self.mlp(inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = nn.functional.mse_loss(outputs, targets, reduce=False)
        match self.method:
            case 'FOCUS':
                weight = 1 - torch.sigmoid(phi(self.global_step) * (loss - self.gamma))
                loss = loss * weight
            case 'initial_low_error':
                loss = loss * self.initial_weight
            case 'all_data':
                pass
            case _:
                raise NotImplementedError(f"Unknown {self.method=}")

        loss = loss.mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = nn.functional.mse_loss(outputs, targets)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4)
        return optimizer

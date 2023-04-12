"""
This adapts the source model by loading the previously trained checkpoint,
then training more with a smaller learning rate on the target dataset.
"""
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam

import wandb


class DynamicsNetwork(pl.LightningModule):
    def __init__(self, method: Optional[str] = None, input_dim=80, output_dim=60, lr=1e-3, train_dataloaders=None,
                 val_dataloaders=None, gamma=0.001, global_k=10):
        super().__init__()
        assert global_k > 0
        self.train_dataloaders = train_dataloaders
        self.val_dataloaders = val_dataloaders
        self.lr = lr
        self.gamma = gamma
        self.global_k = global_k
        self.method = method
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim))

    def train_dataloader(self):
        return self.train_dataloaders

    def val_dataloader(self):
        return self.val_dataloaders

    def forward(self, inputs):
        outputs = self.mlp(inputs)
        return outputs

    def compute_errors(self, outputs, targets):
        error = nn.functional.mse_loss(outputs, targets, reduction='none')
        log_dict = {"error": error.detach().cpu().numpy().mean(-1)}
        match self.method:
            case 'FOCUS2':
                weight = 1 - torch.sigmoid(self.global_k * self.global_step * (error - torch.quantile(error, 0.25)))
                weighted_error = error * weight
                log_dict["weights"] = weight.detach().cpu().numpy().mean(-1)
            case 'FOCUS':
                weight = 1 - torch.sigmoid(self.global_k * self.global_step * (error - self.gamma))
                weighted_error = error * weight
                log_dict["weights"] = weight.detach().cpu().numpy().mean(-1)
            case 'initial_low_error':
                weighted_error = error * self.initial_weight
            case 'all_data':
                weighted_error = error
            case _:
                raise NotImplementedError(f"Unknown {self.method=}")

        loss = weighted_error.mean()
        log_dict["loss"] = loss
        return log_dict

    def training_step(self, batch, batch_idx):
        inputs, targets, uuids = batch
        outputs = self.forward(inputs)
        log_dict = self.compute_errors(outputs, targets)
        log_dict["uuids"] = uuids

        train_log_dict = {f"train_{k}": v for k, v in log_dict.items()}
        train_log_dict["global_step"] = self.global_step

        if self.global_step % 5 == 0:
            wandb.log(train_log_dict)

        return log_dict['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets, uuids = batch

        outputs = self.forward(inputs)

        log_dict = self.compute_errors(outputs, targets)
        log_dict["uuids"] = uuids

        val_dataset_name = self.val_dataloaders[dataloader_idx].dataset.name
        val_log_dict = {f"{val_dataset_name}_{k}": v for k, v in log_dict.items()}
        val_log_dict["global_step"] = self.global_step

        if self.global_step % 5 == 0:
            wandb.log(val_log_dict)

        return log_dict['loss']

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

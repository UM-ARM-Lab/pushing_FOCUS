"""
This adapts the source model by loading the previously trained checkpoint,
then training more with a smaller learning rate on the target dataset.
"""
from typing import Optional

import pytorch_lightning as pl
import torch
import wandb
from torch import nn
from torch.optim import Adam

from dataset import P, H


class DynamicsNetwork(pl.LightningModule):
    def __init__(self, method: Optional[str] = None, context_dim=H * 9, action_dim=P * 3, output_dim=P * 3, lr=1e-3,
                 train_dataloaders=None, val_dataloaders=None, gamma=0.001, global_k=10):
        super().__init__()
        assert global_k > 0
        self.train_dataloaders = train_dataloaders
        self.val_dataloaders = val_dataloaders
        self.lr = lr
        self.gamma = gamma
        self.global_k = global_k
        self.method = method
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.actions_mlp = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.ReLU(),
        )
        self.predictor_mlp = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def train_dataloader(self):
        return self.train_dataloaders

    def val_dataloader(self):
        return self.val_dataloaders

    def forward(self, context, actions):
        """
        Args:
            context: a tensor of shape (batch_size, H*9)
            actions: a tensor of shape (batch_size, P*3)
        """
        context_z = self.context_mlp(context)
        actions_z = self.actions_mlp(actions)
        deltas = self.predictor_mlp(torch.cat((context_z, actions_z), dim=-1))
        # integrate deltas
        # starting from the last object position in the context
        # this is a well known trick in dynamics predictions problems,
        # and is similar to how ResNets work.
        deltas = deltas.reshape(-1, P, 3)
        deltas = torch.cumsum(deltas, dim=1)
        context = context.reshape(-1, 3, 9)
        context_object_positions = context[:, :, 3:6]
        final_object_positions = context_object_positions[:, -1:, :]
        final_object_positions = torch.tile(final_object_positions, (1, P, 1))
        outputs = final_object_positions + deltas
        outputs = outputs.reshape(-1, P * 3)
        return outputs

    def compute_errors(self, outputs, targets, global_step=-1):
        error = (outputs - targets).square().sum(-1)
        log_dict = {"error": error.detach().cpu().numpy()}
        weight = self.compute_weight(error, global_step, log_dict)
        weighted_error = error * weight

        loss = weighted_error.mean()
        log_dict["loss"] = loss
        return log_dict

    def compute_weight(self, error, global_step, log_dict):
        with torch.no_grad():  # don't accidentally do metalearning...
            match self.method:
                case 'FOCUS2':
                    weight = 1 - torch.sigmoid(self.global_k * global_step * (error - torch.quantile(error, 0.25)))
                    log_dict["weights"] = weight.detach().cpu().numpy().mean(-1)
                case 'FOCUS':
                    weight = 1 - torch.sigmoid(self.global_k * global_step * (error - self.gamma))
                    log_dict["weights"] = weight.detach().cpu().numpy().mean(-1)
                case 'CL':  # standard curriculum learning
                    weight = 1 - torch.sigmoid(self.global_k * error - self.gamma)
                    log_dict["weights"] = weight.detach().cpu().numpy().mean(-1)
                case 'all_data':
                    weight = torch.ones_like(error)
                case 'all_data_no_mde':
                    weight = torch.ones_like(error)
                case _:
                    raise NotImplementedError(f"Unknown {self.method=}")
        return weight

    def training_step(self, batch, batch_idx=None):
        context, actions, targets, _, uuids = batch
        outputs = self.forward(context, actions)
        log_dict = self.compute_errors(outputs, targets, self.global_step)
        log_dict["uuids"] = uuids

        train_log_dict = {f"train_{k}": v for k, v in log_dict.items()}
        train_log_dict["global_step"] = self.global_step

        if self.global_step % 10 == 0:
            wandb.log(train_log_dict)

        return log_dict['loss']

    def validation_step(self, batch, batch_idx=None, dataloader_idx=0):
        context, actions, targets, _, uuids = batch

        outputs = self.forward(context, actions)

        log_dict = self.compute_errors(outputs, targets, self.global_step)
        log_dict["uuids"] = uuids

        val_dataset_name = self.val_dataloaders[dataloader_idx].dataset.name
        val_log_dict = {f"{val_dataset_name}_{k}": v for k, v in log_dict.items()}
        val_log_dict["global_step"] = self.global_step

        if self.global_step % 10 == 0:
            wandb.log(val_log_dict)

        return log_dict['loss']

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer


class MDE(pl.LightningModule):
    def __init__(self, context_dim=9 * 3, action_dim=P * 3, predictions_dim=P * 3, output_dim=P):
        super().__init__()
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.ReLU(),
        )
        self.predictions_mlp = nn.Sequential(
            nn.Linear(predictions_dim, 256),
            nn.ReLU(),
        )
        self.deviation_mlp = nn.Sequential(
            nn.Linear(256 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, context, actions, predictions):
        """
        Args:
            context: a tensor of shape (batch_size, H*9)
            actions: a tensor of shape (batch_size, P*3)
            predictions: a tensor of shape (batch_size, P*3)
        """
        context_z = self.context_mlp(context)
        actions_z = self.action_mlp(actions)
        predictions_z = self.predictions_mlp(predictions)
        deviations = self.deviation_mlp(torch.cat((context_z, actions_z, predictions_z), dim=-1))
        return deviations

    def compute_errors(self, outputs, targets):
        error = (outputs - targets).square().sum(-1)
        log_dict = {"error": error.detach().cpu().numpy()}
        loss = error.mean()
        log_dict["loss"] = loss
        return log_dict

    def training_step(self, batch, batch_idx=None):
        context, actions, predictions, target_deviations, uuids = batch
        outputs = self.forward(context, actions, predictions)
        log_dict = self.compute_errors(outputs, target_deviations)
        log_dict["uuids"] = uuids

        return log_dict['loss']

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

"""
This adapts the source model by loading the previously trained checkpoint,
then training more with a smaller learning rate on the target dataset.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split

import wandb


class DynamicsDataset(Dataset):
    def __init__(self):
        artifact = wandb.Api().artifact('armlab/pushing_focus/target-dataset:latest', type='dataset')
        dataset_file = Path(artifact.download()) / 'dataset.pkl'
        with dataset_file.open('rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        trajectory = self.dataset[idx]
        before_robot_pos = torch.tensor(np.array([b['before']['robot_pos'] for b in trajectory]))
        before_object_pos = torch.tensor(np.array([b['before']['object_pos'] for b in trajectory]))
        action = torch.tensor(np.array([b['action'] for b in trajectory]))
        after_robot_pos = torch.tensor(np.array([b['after']['robot_pos'] for b in trajectory]))
        after_object_pos = torch.tensor(np.array([b['after']['object_pos'] for b in trajectory]))
        inputs = torch.cat([before_robot_pos, before_object_pos, action], -1).reshape(-1).float()
        targets = torch.cat([after_robot_pos, after_object_pos], -1).reshape(-1).float()
        return inputs, targets


def phi(j):
    return 10 * j


class DynamicsNetwork(pl.LightningModule):
    def __init__(self, method: str, input_dim=80, output_dim=60):
        super().__init__()
        self.gamma = 0.02
        self.method = method
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim))

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.mlp(inputs)
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
        outputs = self.mlp(inputs)
        loss = nn.functional.mse_loss(outputs, targets)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4)
        return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['FOCUS', 'all_data', 'initial_low_error'])

    args = parser.parse_args()

    # Load the dataset
    dataset = DynamicsDataset()

    # Create random subsets of training and validation
    train_set, val_set = random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])
    train_loader = DataLoader(train_set, batch_size=16, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, num_workers=0)

    # Train the network
    model = DynamicsNetwork(args.method)

    # train the model
    wandb_logger = WandbLogger(project="pushing_focus", log_model="all", checkpoint_name="model.ckpt")
    wandb.config({
        'method': args.method,
        'gamma': model.gamma,
        'is_adapt': True,
    }
    )
    wandb_logger.watch(model)
    trainer = pl.Trainer(max_epochs=100, logger=wandb_logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()

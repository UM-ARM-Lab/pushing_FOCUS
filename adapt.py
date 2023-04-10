"""
This adapts the source model by loading the previously trained checkpoint,
then training more with a smaller learning rate on the target dataset.
"""
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

import wandb
from dataset import DynamicsDataset
from model import DynamicsNetwork


def phi(j):
    return 10 * j


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['FOCUS', 'all_data', 'initial_low_error'])

    args = parser.parse_args()

    # Load the dataset
    dataset = DynamicsDataset('target-dataset')

    # Create random subsets of training and validation
    train_set, val_set = random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])
    train_loader = DataLoader(train_set, batch_size=16, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, num_workers=0)

    # Train the network
    model = DynamicsNetwork(args.method)

    # train the model
    wandb_logger = WandbLogger(project="pushing_focus", log_model="all", checkpoint_name="model.ckpt")
    wandb.config.update({
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

"""
This script trains a simple 2-layer MLP on the source dynamics dataset
"""

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

import wandb
from dataset import DynamicsDataset
from model import DynamicsNetwork


def main():
    # Load the dataset
    dataset = DynamicsDataset('source-dataset')

    # Create random subsets of training and validation
    train_set, val_set = random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])
    train_loader = DataLoader(train_set, batch_size=16)
    val_loader = DataLoader(val_set, batch_size=16)
    val_loader.dataset.name = 'val'

    # Train the network
    model = DynamicsNetwork(method='all_data', train_dataloaders=train_loader, val_dataloaders=[val_loader])

    # train the model
    wandb.init(entity='armlab', project='pushing_focus')
    wandb_logger = WandbLogger(project="pushing_focus", log_model="all", checkpoint_name="source_model")
    wandb.config.update({
        'is_adapt': False,
    })
    trainer = pl.Trainer(max_epochs=100, logger=wandb_logger, check_val_every_n_epoch=5, log_every_n_steps=5)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()

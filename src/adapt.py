"""
This adapts the source model by loading the previously trained checkpoint,
then training more with a smaller learning rate on the target dataset.
"""
import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from dataset import DynamicsDataset
from model import DynamicsNetwork


def main():
    np.set_printoptions(precision=3, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['FOCUS2', 'FOCUS', 'all_data', 'initial_low_error'])

    args = parser.parse_args()

    dataset = DynamicsDataset('target-dataset')
    similar_dataset = DynamicsDataset('similar-dataset')
    dissimilar_dataset = DynamicsDataset('dissimilar-dataset')

    train_loader = DataLoader(dataset, batch_size=16)

    val_dataloaders = [
        DataLoader(similar_dataset, batch_size=16),
        DataLoader(dissimilar_dataset, batch_size=16),
    ]

    # Load from source model
    wandb.init(entity='armlab')
    api = wandb.Api()
    artifact = api.artifact('armlab/pushing_focus/source_model:latest')
    model_path = Path(artifact.download())
    model = DynamicsNetwork.load_from_checkpoint(model_path / 'model.ckpt',
                                                 lr=1e-4,
                                                 method=args.method,
                                                 train_dataloaders=train_loader,
                                                 val_dataloaders=val_dataloaders,
                                                 gamma=0.0001,
                                                 global_k=10)

    # fine-tune the model
    wandb_logger = WandbLogger(project="pushing_focus", log_model="all", checkpoint_name=f"adapted_{args.method}_model")
    wandb.config.update({
        'method': args.method,
        'gamma': model.gamma,
        'is_adapt': True,
    })
    trainer = pl.Trainer(max_epochs=100, logger=wandb_logger, log_every_n_steps=2, enable_checkpointing=True, check_val_every_n_epoch=5)
    trainer.fit(model=model)


if __name__ == '__main__':
    main()

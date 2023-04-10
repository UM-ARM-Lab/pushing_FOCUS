import pickle
from pathlib import Path

import numpy as np
import torch

import wandb
from torch.utils.data import Dataset


class DynamicsDataset(Dataset):
    def __init__(self, dataset_name: str):
        self.artifact = wandb.Api().artifact(f'armlab/pushing_focus/{dataset_name}:latest', type='dataset')
        self.dataset_file = Path(self.artifact.download()) / 'dataset.pkl'
        with self.dataset_file.open('rb') as f:
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


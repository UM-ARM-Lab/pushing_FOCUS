import pickle
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import Dataset

H = 3


class DynamicsDataset(Dataset):
    def __init__(self, dataset_name: str, version: str = 'latest'):
        self.name = dataset_name
        self.artifact = wandb.Api().artifact(f'armlab/pushing_focus/{dataset_name}:{version}', type='dataset')
        self.dataset_file = Path(self.artifact.download()) / 'dataset.pkl'
        with self.dataset_file.open('rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a tuple of (context, actions, target_obj_positions, uuid).
        context: (H*8) flattened
            Robot position, and object position, and action for H time steps before prediction starts.
        actions: (T*2) flattened
            Action for T time steps after prediction starts.
        target_obj_positions: (T*3) flattened
            Object position for T time steps after prediction starts.
        uuid: string
            Unique identifier for the trajectory.
        """
        trajectory = self.dataset[idx]

        context = []
        for t in range(H):
            context_transition_t = trajectory[t]
            context_before_t = context_transition_t['before']
            context_action_t = context_transition_t['action']
            context_robot_pos_t = context_before_t['robot_pos']
            context_object_pos_t = context_before_t['object_pos']
            context.append(np.concatenate([context_robot_pos_t, context_object_pos_t, context_action_t]))
        context = torch.flatten(torch.tensor(np.array(context)))

        actions = []
        for t in range(H - 1, len(trajectory)):
            transition_t = trajectory[t]
            action_t = transition_t['action']
            actions.append(action_t)
        actions = torch.flatten(torch.tensor(np.array(actions)))

        target_obj_positions = []
        target_robot_positions = []
        for t in range(H - 1, len(trajectory)):
            transition_t = trajectory[t]
            target_obj_t = transition_t['after']['object_pos']
            target_obj_positions.append(target_obj_t)
            target_robot_t = transition_t['after']['robot_pos']
            target_robot_positions.append(target_robot_t)
        target_obj_positions = np.array(target_obj_positions)
        target_obj_positions = torch.flatten(torch.tensor(target_obj_positions))
        target_robot_positions = torch.flatten(torch.tensor(np.array(target_robot_positions)))

        uuid = trajectory[0]['uuid']
        return context.float(), actions.float(), target_obj_positions.float(), target_robot_positions.float(), uuid


def save(dataset, env_name, seed):
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    run = wandb.init(project='pushing_focus', entity='armlab', config={
        'env_name': env_name,
        'seed': seed,
        'n_transitions': len(dataset),
    })

    artifact = wandb.Artifact(f'{env_name}-dataset', type='dataset')
    artifact.add_file('dataset.pkl')
    run.log_artifact(artifact)

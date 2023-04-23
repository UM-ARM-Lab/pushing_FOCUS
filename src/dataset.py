import pickle
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt, cm
from torch.utils.data import Dataset

from collect_data import T

H = 3
# the prediction horizon
P = T - H + 1


def get_context(trajectory):
    context = []
    for t in range(H):
        context_transition_t = trajectory[t]
        context_before_t = context_transition_t['before']
        context_action_t = context_transition_t['action']
        context_robot_pos_t = context_before_t['robot_pos']
        context_object_pos_t = context_before_t['object_pos']

        context.append(np.concatenate([context_robot_pos_t, context_object_pos_t, context_action_t]))
    context = torch.flatten(torch.tensor(np.array(context)))
    return context


def get_actions(trajectory):
    actions = []
    for t in range(H - 1, len(trajectory)):
        transition_t = trajectory[t]
        action_t = transition_t['action']
        actions.append(action_t)
    actions = torch.flatten(torch.tensor(np.array(actions)))
    return actions


def get_predictions(trajectory):
    predictions = []
    for t in range(H - 1, len(trajectory)):
        transition_t = trajectory[t]
        pred_object_pos_t = transition_t['pred_object_pos']
        predictions.append(pred_object_pos_t)
    predictions = torch.flatten(torch.tensor(np.array(predictions)))
    return predictions


def get_targets(trajectory):
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
    return target_obj_positions, target_robot_positions


class DynamicsDataset(Dataset):
    def __init__(self,
                 dataset_name: str,
                 version: str = 'latest',
                 trajectories: Optional[List[List[Dict]]] = None):
        self.name = dataset_name
        if trajectories is not None:
            self.trajectories = trajectories
        else:
            self.artifact = wandb.Api().artifact(f'armlab/pushing_focus/{dataset_name}:{version}', type='dataset')
            self.dataset_file = Path(self.artifact.download()) / 'dataset.pkl'
            with self.dataset_file.open('rb') as f:
                self.trajectories = pickle.load(f)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        """
        Returns a tuple of (context, actions, target_obj_positions, uuid).
        context: (H*9) flattened
            Robot position, and object position, and action for H time steps before prediction starts.
        actions: (T*3) flattened
            Action for T time steps after prediction starts.
        target_obj_positions: (T*3) flattened
            Object position for T time steps after prediction starts.
        uuid: string
            Unique identifier for the trajectory.
        """
        trajectory = self.trajectories[idx]

        context = get_context(trajectory)
        actions = get_actions(trajectory)
        target_obj_positions, target_robot_positions = get_targets(trajectory)

        uuid = trajectory[0]['uuid']
        return context.float(), actions.float(), target_obj_positions.float(), target_robot_positions.float(), uuid


class MDEDataset(Dataset):
    def __init__(self, dataset_name: str, trajectories: Optional[List[List[Dict]]] = None):
        self.name = dataset_name
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        """
        Takes in trajectories of length 10 and just ignores the last 7.

        Returns a tuple of (context, actions, deviations, uuid).
        context: (H*9) flattened
            Robot position, and object position, and action for H time steps before prediction starts.
        actions: (T*3)
            next action
        target_deviation: (T)
            Object position for the next time step
        uuid: string
            Unique identifier for the trajectory.
        """
        trajectory = self.trajectories[idx]

        context = get_context(trajectory)
        actions = get_actions(trajectory)
        predictions = get_predictions(trajectory)
        target_deviations = torch.tensor([t['deviation'] for t in trajectory[H - 1:]], dtype=torch.float)
        uuid = trajectory[0]['uuid']
        return context.float(), actions.float(), predictions.float(), target_deviations, uuid

    def viz_dataset(self):
        """
        Plot the x/y positions of the final object position,
        with the final deviation as the color of the point.
        """
        final_deviations = []
        all_deviations = []
        for traj in self.trajectories:
            final_deviations.append(traj[-1]['deviation'])
            for transition in traj[2:]:
                all_deviations.append(transition['deviation'])
        final_deviations = np.array(final_deviations)

        viz_final_deviations(self.trajectories, final_deviations, 'all data')

        # plt.figure()
        # plt.hist(all_deviations)
        # plt.xlabel("deviation")
        # plt.ylabel("count")
        # plt.title("all deviations")

        plt.show()


def viz_final_deviations(trajectories, final_deviations, title):
    plt.figure()
    for traj, final_deviation in zip(trajectories, final_deviations):
        # context = get_context(traj).detach().numpy()
        # context = context.reshape(H, 9)
        # context_object_positions = context[-1, 3:6]
        predictions = get_predictions(traj).detach().numpy()
        predictions = predictions.reshape(P, 3)
        color = cm.jet(final_deviation * 3)
        plt.plot(predictions[:, 0], predictions[:, 1], color=color)
        plt.scatter(predictions[0, 0], predictions[0, 1], color=color)
        plt.scatter(predictions[-1, 0], predictions[-1, 1], color=color)

    plt.axis("equal")
    plt.xlim(0, 0.5)
    plt.ylim(-0.25, 0.25)
    plt.title(f"{title}: final deviations")
    plt.xlabel("final object x")
    plt.ylabel("final object y")
    plt.show()


def save(dataset, env_name, seed):
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    run = wandb.init(project='pushing_focus', entity='armlab', config={
        'env_name': env_name,
        'seed': seed,
        'n_transitions': len(dataset),
        'H': H,
    })

    artifact = wandb.Artifact(f'{env_name}-dataset', type='dataset')
    artifact.add_file('dataset.pkl')
    run.log_artifact(artifact)

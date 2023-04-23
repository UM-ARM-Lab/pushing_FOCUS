"""
The main file demonstrating the full online-learning method proposed in the following paper:
https://arxiv.org/abs/2209.14261
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from more_itertools import windowed
from torch.utils.data import DataLoader

from dataset import DynamicsDataset, MDEDataset, get_actions, get_context, H
from model import DynamicsNetwork, MDE
from mppi_runner import LearnedMPPIRunner


class FOCUS:

    def __init__(self):
        # get the source dynamics model
        api = wandb.Api()
        artifact = api.artifact('armlab/pushing_focus/source_model:latest')
        model_path = Path(artifact.download())
        source_dynamics = DynamicsNetwork.load_from_checkpoint(model_path / 'model.ckpt', method='FOCUS')

        # Since we don't have any data from the target environment yet, initialize with the source dynamics
        self.dynamics = source_dynamics
        # There's no such thing as an MDE for the source environment, so we'll just set it to None
        self.mde = None

        self.runner = LearnedMPPIRunner('target.xml', horizon=8, dynamics=self.dynamics)
        self.goal_rng = np.random.RandomState(0)

        self.dynamics_trajectories = []
        self.dynamics_dataset = DynamicsDataset('online_dynamics', trajectories=self.dynamics_trajectories)
        self.mde_trajectories = []
        self.mde_dataset = MDEDataset('online_mde', trajectories=self.mde_trajectories)
        self.itr = 0

    def run(self):
        for self.itr in range(20):
            self.run_episode(self.itr)

    def viz_mde(self):
        # plot the estimated deviation for each x,y point in a grid
        # from [0, -0.5] to [1, 0.5]
        s = 20
        xmin = 0
        xmax = 0.75
        ymin = -0.5
        ymax = 0.5
        x = np.linspace(xmin, xmax, s)
        y = np.linspace(ymin, ymax, s)
        xy = torch.tensor(np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2))
        xy = torch.tile(xy[:, None], [1, 3, 1])
        context = get_context(self.dynamics_trajectories[0])
        context = torch.tile(context.reshape([1, -1, 9]), [len(xy), 1, 1])
        context[:, :, 0:2] += xy
        context[:, :, 4:6] += xy
        context = context.reshape([len(xy), 27])
        # offset the robot and object positions by the XY grid
        action = torch.tensor([0.08, 0, 0])  # a simple forward push
        actions = torch.reshape(torch.tile(action[None, None], [s**2, 8, 1]), [s**2, -1])
        estimated_deviations = self.mde(context.float(), actions.float())
        mean_est_deviation = estimated_deviations.mean(-1)
        print(min(mean_est_deviation), max(mean_est_deviation))

        import matplotlib.pyplot as plt
        data = mean_est_deviation.reshape(s, s).detach().numpy()
        plt.imshow(data)
        plt.xticks(np.linspace(0, s, 5), np.linspace(xmin, xmax, 5))
        plt.yticks(np.linspace(0, s, 5), np.linspace(ymin, ymax, 5))
        plt.title(self.itr)
        plt.show()

    def run_episode(self, episode_idx: int):
        # planning & execution
        goal = self.goal_rng.uniform([0.5, -0.1, 0.05], [0.7, 0.1, 0.05])
        # TODO: not sure I need to re-set these each time, probably not?
        self.runner.dynamics = self.dynamics
        self.runner.mde = self.mde
        success, full_trajectory = self.runner.run(goal)

        # update dynamics model dataset
        # split the full_trajectory, which may be very long, into
        # smaller (possibly overlapping trajectories) of length 10
        # for training the dynamics
        trajectories = list(windowed(full_trajectory, 10))
        for traj in trajectories:
            self.dynamics_trajectories.append(list(traj))

        # fine-tune dynamics model (using the proposed adaptation method!)
        # dynamics_loader = DataLoader(self.dynamics_dataset, batch_size=16)
        # wandb.init(project='pushing_focus_online', name=f'fine_tune_dynamics_{episode_idx   }')
        # trainer = pl.Trainer(max_epochs=10, logger=None, log_every_n_steps=1)
        # trainer.fit(self.dynamics, dynamics_loader, val_dataloaders=[])
        # wandb.finish()

        # update MDE dataset
        # use a deepcopy, so we don't accidentally modify the dynamics dataset trajectories
        for mde_traj in deepcopy(trajectories):
            context = get_context(mde_traj).float()[None]
            actions = get_actions(mde_traj).float()[None]
            pred_object_positions = self.dynamics(context, actions)[0]
            pred_object_positions = pred_object_positions.reshape(-1, 3)
            pred_object_positions = pred_object_positions.detach().numpy()  # [8, 3]
            object_positions = np.array([t['before']['object_pos'] for t in mde_traj[H - 1:]])  # [8, 3]
            deviations = np.linalg.norm(pred_object_positions - object_positions, axis=-1)
            print(deviations)
            for t, transition in enumerate(mde_traj[H - 1:]):
                transition['deviation'] = deviations[t]
            self.mde_trajectories.append(list(mde_traj))

        # fine-tune MDE (standard)
        mde_loader = DataLoader(self.mde_dataset, batch_size=16)
        trainer = pl.Trainer(max_epochs=5, logger=None, log_every_n_steps=1)
        if self.mde is None:
            self.mde = MDE()
        trainer.fit(self.mde, mde_loader)
        self.mde_dataset.viz_dataset()
        self.viz_mde()
        print("done")


def main():
    torch.manual_seed(0)
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    f = FOCUS()
    f.run()


if __name__ == '__main__':
    main()

"""
The main file demonstrating the full online-learning method proposed in the following paper:
https://arxiv.org/abs/2209.14261
"""
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from matplotlib import pyplot as plt
from more_itertools import windowed
from torch.utils.data import DataLoader

from collect_data import T
from dataset import DynamicsDataset, MDEDataset, get_actions, get_context, H, viz_final_deviations
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

        self.runner = LearnedMPPIRunner('target.xml', dynamics=self.dynamics)
        self.goal_rng = np.random.RandomState(0)

        self.dynamics_trajectories = []
        self.dynamics_dataset = DynamicsDataset('online_dynamics', trajectories=self.dynamics_trajectories)
        self.mde_trajectories = []
        self.mde_dataset = MDEDataset('online_mde', trajectories=self.mde_trajectories)
        self.itr = 0

    def run(self):
        for self.itr in range(500):
            self.run_episode(self.itr)

    def viz_mde(self, new_mde_trajectories, title: str):
        if self.mde is None:
            return

        context = torch.stack([get_context(t) for t in new_mde_trajectories]).float()
        actions = torch.stack([get_actions(t) for t in new_mde_trajectories]).float()
        predictions = self.dynamics(context, actions)

        estimated_deviations = self.mde(context, actions, predictions)
        final_estimated_deviation = estimated_deviations[:, -1]
        final_estimated_deviation = final_estimated_deviation.detach().numpy()
        # print(f'{min(final_estimated_deviation)=} {max(final_estimated_deviation)=}')

        viz_final_deviations(new_mde_trajectories, final_estimated_deviation, f'{title}')
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
        # dynamics_loader = DataLoader(self.dynamics_dataset, batch_size=64)
        # wandb.init(project='pushing_focus_online', name=f'fine_tune_dynamics_{episode_idx   }')
        # trainer = pl.Trainer(max_epochs=10, logger=None, log_every_n_steps=1)
        # trainer.fit(self.dynamics, dynamics_loader, val_dataloaders=[])
        # wandb.finish()

        # update MDE dataset
        # use a deepcopy, so we don't accidentally modify the dynamics dataset trajectories
        new_mde_trajectories = []
        for mde_traj in deepcopy(trajectories):
            context = get_context(mde_traj).float()[None]
            actions = get_actions(mde_traj).float()[None]
            pred_object_positions_flat = self.dynamics(context, actions)[0]
            pred_object_positions = pred_object_positions_flat.reshape(-1, 3)
            pred_object_positions = pred_object_positions.detach().numpy()  # [T, 3]
            object_positions = np.array([t['before']['object_pos'] for t in mde_traj[H - 1:]])  # [T, 3]
            deviations = np.linalg.norm(pred_object_positions - object_positions, axis=-1)
            # print(f'{deviations[-1]=}')
            for t, transition in enumerate(mde_traj[H - 1:]):
                transition['deviation'] = deviations[t]
                transition['pred_object_pos'] = pred_object_positions[t]
            new_mde_trajectories.append(list(mde_traj))
        self.mde_trajectories.extend(new_mde_trajectories)

        # fine-tune MDE (standard)
        # self.viz_mde(new_mde_trajectories, 'before')
        mde_loader = DataLoader(self.mde_dataset, batch_size=999)  # FIXME: full batch vs minibatch
        trainer = pl.Trainer(max_epochs=25, logger=None, log_every_n_steps=1)
        if self.mde is None:
            self.mde = MDE()
        trainer.fit(self.mde, mde_loader)
        # self.viz_mde(new_mde_trajectories, 'after')
        # self.mde_dataset.viz_dataset()
        print("done")


def main():
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    torch.manual_seed(0)
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    f = FOCUS()
    f.run()


if __name__ == '__main__':
    main()

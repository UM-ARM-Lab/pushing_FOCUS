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
from colorama import Fore
from matplotlib import pyplot as plt
from more_itertools import windowed
from torch.utils.data import DataLoader

from dataset import DynamicsDataset, MDEDataset, get_actions, get_context, H, viz_final_deviations, T
from model import DynamicsNetwork, MDE
from mppi_runner import LearnedMPPIRunner

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class FOCUS:

    def __init__(self, seed, method='FOCUS'):
        self.seed = seed

        # get the source dynamics model
        api = wandb.Api()
        artifact = api.artifact('armlab/pushing_focus/source_model:latest')
        model_path = Path(artifact.download())
        source_dynamics = DynamicsNetwork.load_from_checkpoint(model_path / 'model.ckpt', method=method)

        # Since we don't have any data from the target environment yet, initialize with the source dynamics
        self.dynamics = source_dynamics
        self.dynamics.lr = 1e-4

        self.runner = LearnedMPPIRunner('target.xml', dynamics=self.dynamics)
        self.goal_rng = np.random.RandomState(seed)

        self.dynamics_trajectories = []
        self.dynamics_dataset = DynamicsDataset('online_dynamics', trajectories=self.dynamics_trajectories)
        self.mde_trajectories = []
        self.mde_dataset = MDEDataset('online_mde', trajectories=self.mde_trajectories)
        self.n_episodes = 25

        run = wandb.init(project='pushing_focus', entity='armlab',
                         job_type='full_online_adaptation',
                         config={
                             'seed': seed,
                             'method': method,
                             'n_episodes': self.n_episodes,
                             'source_dynamics_version': artifact.version,
                         })

    def run(self):
        for i in range(self.n_episodes):
            self.run_episode(i)

    def eval(self):
        n_success = 0
        n_total = 20
        for i in range(n_total):
            goal = self.sample_goal()
            success, full_trajectory = self.runner.run(goal)
            if success:
                n_success += 1

        logger.info(Fore.GREEN + f'{n_success}/{n_total}={n_success / n_total:.1%} success rate' + Fore.RESET)

        wandb.log({
            'n_success': n_success,
            'n_total': n_total,
            'success_rate': n_success / n_total,
            'seed': self.seed,
        })

    def viz_mde(self, new_mde_trajectories, title: str):
        if self.runner.mde is None:
            return

        context = torch.stack([get_context(t) for t in new_mde_trajectories]).float()
        actions = torch.stack([get_actions(t) for t in new_mde_trajectories]).float()
        predictions = self.dynamics(context, actions)

        estimated_deviations = self.runner.mde(context, actions, predictions)
        final_estimated_deviation = estimated_deviations[:, -1]
        final_estimated_deviation = final_estimated_deviation.detach().numpy()
        # print(f'{min(final_estimated_deviation)=} {max(final_estimated_deviation)=}')

        viz_final_deviations(new_mde_trajectories, final_estimated_deviation, f'{title}')
        plt.show()

    def run_episode(self, episode_idx: int):
        logger.info(Fore.CYAN + f'episode {episode_idx}' + Fore.RESET)

        # planning & execution
        goal = self.sample_goal()
        success, full_trajectory = self.runner.run(goal)

        wandb.log({
            'episode_idx': episode_idx,
            'success': success,
        })

        # update dynamics model dataset
        # split the full_trajectory, which may be very long, into
        # smaller (possibly overlapping trajectories) of length T
        # for training the dynamics
        trajectories = list(windowed(full_trajectory, T))
        for traj in trajectories:
            self.dynamics_trajectories.append(list(traj))

        # fine-tune dynamics model (using the proposed adaptation method!)
        if episode_idx > 0:
            dynamics_loader = DataLoader(self.dynamics_dataset, batch_size=999)
            trainer = pl.Trainer(max_epochs=10, logger=None, log_every_n_steps=1)
            trainer.fit(self.dynamics, dynamics_loader, val_dataloaders=[])

        # update MDE dataset
        # use a deepcopy, so we don't accidentally modify the dynamics dataset trajectories
        new_mde_trajectories = []
        for mde_traj in deepcopy(trajectories):
            context = get_context(mde_traj).float()[None]
            actions = get_actions(mde_traj).float()[None]
            pred_object_positions_flat = self.dynamics(context, actions)[0]
            pred_object_positions = pred_object_positions_flat.reshape(-1, 3)
            pred_object_positions = pred_object_positions.detach().numpy()  # [P, 3]
            object_positions = np.array([t['before']['object_pos'] for t in mde_traj[H - 1:]])  # [P, 3]
            deviations = np.linalg.norm(pred_object_positions - object_positions, axis=-1)
            # print(f'{deviations[-1]=}')
            for t, transition in enumerate(mde_traj[H - 1:]):
                transition['deviation'] = deviations[t]
                transition['pred_object_pos'] = pred_object_positions[t]
            new_mde_trajectories.append(list(mde_traj))
        self.mde_trajectories.extend(new_mde_trajectories)

        # fine-tune MDE (standard)
        if episode_idx > 0 and 'no_mde' not in self.dynamics.method:
            # self.viz_mde(new_mde_trajectories, 'before')
            mde_loader = DataLoader(self.mde_dataset, batch_size=999)  # FIXME: full batch vs minibatch
            trainer = pl.Trainer(max_epochs=25, logger=None, log_every_n_steps=1)
            if self.runner.mde is None:
                self.runner.mde = MDE()
            trainer.fit(self.runner.mde, mde_loader)
            # self.viz_mde(new_mde_trajectories, 'after')
            # self.mde_dataset.viz_dataset()

    def sample_goal(self):
        goal = self.goal_rng.uniform([0.6, -0.1, 0.05], [0.9, 0.1, 0.05])
        return goal


def main():
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    seed = 2
    torch.manual_seed(seed)
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    f = FOCUS(seed, method='FOCUS')
    f.run()
    f.eval()

    wandb.finish()


if __name__ == '__main__':
    main()

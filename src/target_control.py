from pathlib import Path

import numpy as np
import torch
import wandb

from model import DynamicsNetwork
from mppi_runner import MujocoMPPIRunner, LearnedMPPIRunner


def main():
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    # get the source dynamics model
    api = wandb.Api()
    artifact = api.artifact('armlab/pushing_focus/source_model:latest')
    model_path = Path(artifact.download())
    dynamics = DynamicsNetwork.load_from_checkpoint(model_path / 'model.ckpt')
    dynamics.eval()

    # runner = MujocoMPPIRunner('target.xml',  dynamics=None)
    runner = LearnedMPPIRunner('target.xml', dynamics=dynamics)  # must be 8 to match how model was trained

    n_total = 0
    n_success = 0
    for i in range(3):
        for seed in range(3):
            goal = np.random.uniform([0.5, -0.1, 0.05], [0.7, 0.1, 0.05])
            success, _ = runner.run(goal)
            n_total += 1
            if success:
                n_success += 1
    print(f'Success rate: {n_success / n_total:.1%}')


if __name__ == '__main__':
    main()

import numpy as np
import torch

from mppi_runner import MujocoMPPIRunner, LearnedMPPIRunner


def main():
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    # runner = MujocoMPPIRunner(horizon=8)
    runner = LearnedMPPIRunner(horizon=8)  # must be 8 to match how model was trained

    for i in range(3):
        for seed in range(3):
            goal = np.random.uniform([0.3, -0.4, 0.05], [0.7, 0.4, 0.05])
            runner.reload_env()
            runner.run(goal)


if __name__ == '__main__':
    main()

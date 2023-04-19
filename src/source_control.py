import numpy as np
import torch

from mppi_runner import MPPIRunner


def main():
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    runner = MPPIRunner()

    for i in range(3):
        for seed in range(3):
            goal = np.random.uniform([0.3, -0.4, 0.05], [0.7, 0.4, 0.05])
            runner.reload_env()
            runner.run(goal)


if __name__ == '__main__':
    main()

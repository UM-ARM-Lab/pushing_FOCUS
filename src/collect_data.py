import time
from typing import Callable
from uuid import uuid4

import numpy as np
import rerun as rr
import wandb
from tqdm import tqdm

import rrr
from dataset import save
from env import Env

H = 3
T = 10
T_STEP = 1


def collect_data(nominal_generator: Callable, env_name: str, model_xml_filename: str, n_trajs: int, seed: int):
    """
    Collects a dataset of trajectories from the environment.

    Args:
        nominal_generator: A function that takes a random number generator and returns a nominal (x, y) velocity.
        env_name: The name of the environment.
        model_xml_filename: The name of the MuJoCo XML file.
        n_trajs: The number of trajectories to collect.
        seed: The random seed.
    """
    rrr.init()

    dataset = []

    rng = np.random.RandomState(seed)

    bar = tqdm(total=n_trajs)
    while len(dataset) < n_trajs:

        # re-initialize the simulation
        env = Env(model_xml_filename)
        env.step(None)
        before = env.get_state()

        # sample a nominal velocity
        nominal_x_vel, nominal_y_vel = nominal_generator(rng)

        trajectory = []

        # since we use a window of states as input, we will often be running
        # the network from states where nothing has moved yet. Therefore,
        # our training data should also have this, so we step a few times to collect
        # some initial data with no motion.
        for _ in range(H):
            action = np.array([0, 0])

            env.step(action)
            after = env.get_state()

            viz_state(before)

            append_transition(after, before, action, trajectory)

            before = after

        for _ in range(12):
            action = np.array([nominal_x_vel, nominal_y_vel]) + rng.normal(0, 0.005, size=(2,))

            env.step(action)
            after = env.get_state()

            viz_state(before)

            append_transition(after, before, action, trajectory)

            before = after

        # slice the trajectory into T-length chunks overlapping by T_STEP
        for start_t in range(0, len(trajectory) - T, T_STEP):
            end_t = start_t + T
            traj_sliced = trajectory[start_t:end_t]
            dataset.append(traj_sliced)
            bar.update(1)
    bar.close()

    save(dataset, env_name, seed)

    wandb.finish()


def append_transition(after, before, action, trajectory):
    transition = {
        'before': before,
        'action': action,
        'after': after,
        'uuid': str(uuid4()),
    }
    trajectory.append(transition)


def viz_state(before):
    rr.log_scalar('state/obj_x', before['object_pos'][0], label='obj x')
    rr.log_scalar('state/obj_y', before['object_pos'][1], label='obj y')
    rr.log_scalar('state/robot_x', before['robot_pos'][0], label='robot x')
    rr.log_scalar('state/robot_y', before['robot_pos'][1], label='robot y')

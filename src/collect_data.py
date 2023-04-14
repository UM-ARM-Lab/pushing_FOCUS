import time
from typing import Callable
from uuid import uuid4

import numpy as np
import rerun as rr

import rrr
import wandb
from dataset import save
from env import Env


def collect_data(nominal_generator: Callable, env_name: str, model_xml_filename: str, n_transitions: int, seed: int):
    """
    Collects a dataset of transitions from the environment.

    Args:
        nominal_generator: A function that takes a random number generator and returns a nominal (x, y) velocity.
        env_name: The name of the environment.
        model_xml_filename: The name of the MuJoCo XML file.
        n_transitions: The number of transitions to collect.
        seed: The random seed.
    """
    rrr.init()

    dataset = []

    rng = np.random.RandomState(seed)

    # run a bunch of smaller push episodes
    while len(dataset) < n_transitions:
        episode_t0 = time.time()

        # re-initialize the simulation
        env = Env(model_xml_filename)
        env.step(None)
        before = env.get_state()

        # sample a nominal velocity
        nominal_x_vel, nominal_y_vel = nominal_generator(rng)

        episode = []

        for _ in range(10):
            action = np.array([nominal_x_vel, nominal_y_vel]) + rng.normal(0, 0.005, size=(2,))

            env.step(action)
            after = env.get_state()

            rr.log_scalar('state/obj_x', before['object_pos'][0], label='obj x')
            rr.log_scalar('state/obj_y', before['object_pos'][1], label='obj y')
            rr.log_scalar('state/robot_x', before['robot_pos'][0], label='robot x')
            rr.log_scalar('state/robot_y', before['robot_pos'][1], label='robot y')

            transition = {
                'before': before,
                'action': action,
                'after': after,
                'uuid': str(uuid4()),
            }

            episode.append(transition)

            before = after

        dataset.append(episode)

        print(f'episode {len(dataset)} took {time.time() - episode_t0:.2f}s')

    save(dataset, env_name, seed)

    wandb.finish()

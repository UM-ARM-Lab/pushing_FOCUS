import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import wandb

import rrr
from env import Env
from model import DynamicsNetwork
from mujoco_mppi import MujocoMPPI, normalized

T = 5



def state_to_vec(state):
    return np.array([state['object_pos'][0],
                     state['object_pos'][1],
                     state['object_pos'][2]])


def goal_satisfied(state, x_goal, threshold=0.1):
    """
    Check if the goal has been satisfied.

    Args:
        state: the current state of the environment
        x_goal: the goal position of the object
        threshold: the distance threshold for the goal to be satisfied

    Returns:
        True if the goal has been satisfied, False otherwise
    """
    distance = np.linalg.norm(state_to_vec(state) - x_goal)
    return distance < threshold


def main():
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    # get the source dynamics model
    api = wandb.Api()
    artifact = api.artifact('armlab/pushing_focus/source_model:latest')
    model_path = Path(artifact.download())
    model = DynamicsNetwork.load_from_checkpoint(model_path / 'model.ckpt')

    # initialize rerun
    rrr.init()

    # initialize the target simulation environment
    env = Env('source.xml')

    # gather a few states to form the initial trajectory for the model
    # this must be greater than or equal to the trajectory size of the model
    trajectory = []
    action = np.zeros(2)
    for t in range(5):
        env.step(None)
        state = env.get_state()
        rrr.viz_state(state)
        transition = {
            'before': state,
            'action': action,
        }
        trajectory.append(transition)

    x_goal = np.array([1.5, 0., 0.05])

    goal_threshold = 0.05
    rr.log_point('object/goal', x_goal, color=(255, 0, 255), radius=goal_threshold)

    pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)

    mppi = MujocoMPPI(pool, env.model, num_samples=50, noise_sigma=0.02, horizon=T, lambda_=10, seed=0, gamma=0.9)
    mppi.U = np.array([[0.01, 0.0]] * T)

    def get_results(_, data):
        state = env.get_state(data=data)
        return state['object_pos'], state['robot_pos']

    def cost(results, action_sequences):
        object_sequences, robot_sequences = results
        for object_sequence, robot_sequence in list(zip(object_sequences, robot_sequences))[::5]:
            rr.log_line_strip(f'object/pred', object_sequence, color=(255, 0, 0), stroke_width=0.002)
            rr.log_line_strip(f'robot/pred', robot_sequence, color=(0, 255, 0), stroke_width=0.002)
        goal_cost = np.linalg.norm(object_sequences - x_goal, axis=-1)[:, 1:]
        obj_robot_dist_cost = np.linalg.norm(object_sequences - robot_sequences, axis=-1)[:, 1:] - 0.1
        return goal_cost + obj_robot_dist_cost * 5

    # warmstart mppi
    for t in range(20):
        action = mppi._command(env.data, get_results, cost)

    for t in range(75):
        state = env.get_state()

        env.step(action)

        if goal_satisfied(state, x_goal, goal_threshold):
            print(f'Goal reached in {t} steps')
            break

        mppi.roll()
        for t in range(10):
            action = mppi._command(env.data, get_results, cost)


if __name__ == '__main__':
    main()

from pathlib import Path
from typing import List, Dict

import numpy as np
import rerun as rr
import torch
import wandb

import dataset
import rrr
from env import Env
from model import DynamicsNetwork


def state_to_vec(state):
    return np.array([state['object_pos'][0],
                     state['object_pos'][1],
                     state['object_pos'][2]])


def predict(model, trajectory: List[Dict], actions: torch.tensor):
    """
    Predict the next state of the object given the current state and a sequence of actions

    Args:
        model: the dynamics model
        trajectory: the current trajectory
        actions: [b, T, 2] a sequence of actions to apply to the object

    Returns:
        [b, T, 3] the predicted next state of the object
    """
    context = dataset.get_context(trajectory).float()
    context = torch.tile(context, [actions.shape[0], 1])
    actions = torch.tensor(actions).float()
    actions = torch.reshape(actions, [actions.shape[0], -1])
    outputs = model(context, actions)
    outputs = outputs.detach().numpy().reshape(actions.shape[0], -1, 3)

    return outputs


def predictive_sampling(model, x_goal: np.ndarray, trajectory: List[Dict]):
    """
    Predict the next state of the object given the current state and a sequence of actions.

    Args:
        model: the dynamics model
        x_goal: the goal position of the object
        trajectory: the current trajectory

    Returns:
        [2] the best action to take
    """
    # sample a bunch of action sequences and pick the one that gets closest to the goal
    n_samples = 100
    T = 8

    rng = np.random.RandomState(0)
    actions = rng.uniform(-0.1, 0.1, [n_samples, 1, 2])
    noise = rng.uniform(-0.01, 0.01, [n_samples, T, 2])
    actions = np.tile(actions, [1, T, 1]) + noise

    outputs = predict(model, trajectory, actions)

    best_action = get_best_action(actions, outputs, x_goal)

    return best_action


def mujoco_predict(model, data, actions: np.array):
    pass


def mujoco_predictive_sampling(model, data, x_goal: np.ndarray):
    # sample a bunch of action sequences and pick the one that gets closest to the goal
    n_samples = 100
    T = 8

    rng = np.random.RandomState(0)
    actions = rng.uniform(-0.1, 0.1, [n_samples, 1, 2])
    noise = rng.uniform(-0.01, 0.01, [n_samples, T, 2])
    actions = np.tile(actions, [1, T, 1]) + noise

    outputs = mujoco_predict(model, data, actions)

    best_action = get_best_action(actions, outputs, x_goal)

    return best_action


def get_best_action(actions, outputs, x_goal):
    # visualize all the predictions
    # for i, pred_obj_positions in enumerate(outputs):
    #     rr.log_line_strip(f'object/pred', pred_obj_positions, color=(0, 0, 1.), stroke_width=0.01)
    # compute the distance to the goal for each sample
    final_states = outputs[:, -1, :]  # [n_samples, 3]
    distances = np.linalg.norm(final_states - x_goal, axis=1)  # [n_samples]
    action_cost = np.linalg.norm(actions, axis=1).sum(-1)
    costs = distances + action_cost
    best_idx = np.argmin(costs)
    best_action = actions[best_idx, 0, :]
    best_pred_obj_positions = outputs[best_idx]
    rr.log_line_strip(f'object/pred', best_pred_obj_positions, color=(0, 255, 0), stroke_width=0.01)
    return best_action


def goal_satisfied(state, x_goal):
    """
    Check if the goal has been satisfied.

    Args:
        state: the current state of the environment
        x_goal: the goal position of the object

    Returns:
        True if the goal has been satisfied, False otherwise
    """
    distance = np.linalg.norm(state_to_vec(state) - x_goal)
    print(distance)
    return distance < 0.1


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

    rr.log_point('object/goal', x_goal, color=(255, 0, 255), radius=0.1)

    for t in range(30):
        state = env.get_state()
        rrr.viz_state(state)

        transition = {
            'before': state,
            'action': action,
        }
        trajectory.append(transition)

        # FIXME: use the H stored in the dataset since it might change
        # action = predictive_sampling(model, x_goal, trajectory[-dataset.H:])
        action = mujoco_predictive_sampling(env.model, env.data, x_goal)

        env.step(action)

        # remove the first element, so we don't keep states we don't need
        trajectory.pop(0)

        if goal_satisfied(state, x_goal):
            print(f'Goal reached in {t} steps')
            break


if __name__ == '__main__':
    main()

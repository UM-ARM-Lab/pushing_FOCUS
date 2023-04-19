import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import wandb

import rrr
from dataset import get_context
from env import Env
from model import DynamicsNetwork
from mppi import MPPI, normalized
from rollout import parallel_rollout


def goal_satisfied(state, goal, threshold=0.1):
    """
    Check if the goal has been satisfied.

    Args:
        state: the current state of the environment
        goal: the goal position of the object
        threshold: the distance threshold for the goal to be satisfied

    Returns:
        True if the goal has been satisfied, False otherwise
    """
    object_pos = np.array([state['object_pos'][0],
                           state['object_pos'][1],
                           state['object_pos'][2]])
    distance = np.linalg.norm(object_pos - goal)
    return distance < threshold


class MPPIRunner:

    def __init__(self, horizon):
        self.horizon = horizon
        self.env = Env('source.xml')

        # get the source dynamics model
        api = wandb.Api()
        artifact = api.artifact('armlab/pushing_focus/source_model:latest')
        model_path = Path(artifact.download())
        self.nn = DynamicsNetwork.load_from_checkpoint(model_path / 'model.ckpt')
        self.nn.eval()

        self.pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)

        noise_sigma = np.array([0.04, 0.04, 0.1])
        self.mppi = MPPI(self.pool, self.env.model, num_samples=35, noise_sigma=noise_sigma, horizon=self.horizon,
                         lambda_=8,
                         seed=0)

        # initialize rerun
        rrr.init()

        self.context_buffer = []
        self.rr_time_offset = 0
        self.goal_threshold = 0.05
        self.global_planning_step = 0

    def reload_env(self):
        self.rr_time_offset += self.env.data.time

        # re-initialize the target simulation environment
        self.env = Env('source.xml')

    def run(self, goal):
        rr.set_time_sequence('planning', self.global_planning_step)
        rr.log_point('goal', goal, color=(255, 0, 255), radius=self.goal_threshold)

        # build up context
        action = np.array([0, 0, 0])
        for _ in range(3):
            state = self.env.get_state()
            self.context_buffer.append({'before': state, 'action': action})
            self.env.step(action)

        # warmstart mppi
        for _ in range(5):
            self.global_planning_step += 1
            action = self.mppi.command(self.dynamics_and_cost_func, goal=goal)

        for t in range(50):
            state = self.env.get_state()
            self.context_buffer.append({'before': state, 'action': action})

            rr.log_arrow('action/xy',
                         state['robot_pos'],
                         np.array([action[0], action[1], 0]),
                         color=(255, 0, 255),
                         width_scale=0.01)
            rrr.log_rotational_velocity('action/z', state['robot_pos'], action[2], color=(255, 0, 255),
                                        stroke_width=0.01)

            self.env.step(action, time_offset=self.rr_time_offset)

            if goal_satisfied(state, goal, self.goal_threshold):
                print(f'Goal reached in {t} steps')
                break

            self.mppi.roll()
            for t in range(1):
                self.global_planning_step += 1
                action = self.mppi.command(self.dynamics_and_cost_func, goal=goal)

    def cost(self, object_sequences, goal):
        """
        Compute the cost of the object sequences.

        Args:
            object_sequences: [num_samples, T, 3]
        """
        goal_cost = np.linalg.norm(object_sequences - goal, axis=-1)

        cost_viz = np.sum(goal_cost, axis=-1)
        cost_viz = normalized(cost_viz)

        for i in np.argsort(cost_viz):
            object_sequence = object_sequences[i]
            c = cost_viz[i]
            rr.set_time_sequence('planning', self.global_planning_step)
            self.global_planning_step += 1

            obj_color = tuple((np.array([255, 0, 0]) * (1 - c)).astype(np.int32))
            robot_color = tuple((np.array([0, 255, 0]) * (1 - c)).astype(np.int32))
            rr.log_line_strip(f'object/pred', object_sequence, color=obj_color, stroke_width=0.002)
            # rr.log_line_strip(f'robot/pred', robot_sequence, color=robot_color, stroke_width=0.002)
        return goal_cost

    def dynamics_and_cost_func(self, perturbed_action, goal):
        raise NotImplementedError


class MujocoMPPIRunner(MPPIRunner):

    def get_results(self, _, data):
        state = self.env.get_state(data=data)
        return state['object_pos'], state['robot_pos']

    def cost_from_results(self, results, goal):
        object_sequences, robot_sequences = results
        object_sequences = object_sequences[:, 1:]
        return self.cost(object_sequences, goal)

    def dynamics_and_cost_func(self, perturbed_action, goal):
        results = parallel_rollout(self.pool, self.env.model, self.env.data, perturbed_action, self.get_results)
        costs = self.cost_from_results(results, goal)
        return costs


class LearnedMPPIRunner(MPPIRunner):

    def dynamics_and_cost_func(self, perturbed_action, goal):
        """
        Run the learned dynamics model and compute the cost.
        Also generate visualization for the predicted trajectories.

        Args:
            perturbed_action: [num_samples, T, 3]
        """
        # context [num_samples, 3, 9] flattened to [num_samples, 27]
        # actions [num_samples, 8, 3] flattened to [num_samples, 24]
        if len(self.context_buffer) > 3:
            self.context_buffer.pop(0)

        num_samples = perturbed_action.shape[0]
        context = get_context(self.context_buffer).float()
        context = torch.tile(context, [num_samples, 1])
        perturbed_action_flat = torch.tensor(perturbed_action).float().reshape(num_samples, -1)
        outputs = self.nn(context, perturbed_action_flat)  # [num_samples, 8, 3] flattened to [num_samples, 24]
        object_sequences = outputs.reshape(num_samples, 8, 3).cpu().detach().numpy()
        costs = self.cost(object_sequences, goal)
        return costs

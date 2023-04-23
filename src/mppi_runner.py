import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rerun as rr
import torch

import rrr
from collect_data import append_transition
from dataset import get_context, P
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

    def __init__(self, xml_model_filename: str, dynamics: DynamicsNetwork):
        self.xml_model_filename = xml_model_filename
        self.env = Env(xml_model_filename)
        self.dynamics = dynamics
        self.mde = None

        self.pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)

        noise_sigma = np.array([0.04, 0.04, 0.1])
        self.mppi = MPPI(self.pool, self.env.model, num_samples=35, noise_sigma=noise_sigma, horizon=P,
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
        self.env = Env(self.xml_model_filename)

    def run(self, goal):
        trajectory = []
        self.reload_env()

        rr.set_time_sequence('planning', self.global_planning_step)
        rr.log_point('goal', goal, color=(255, 0, 255), radius=self.goal_threshold)

        # build up context
        action = np.array([0, 0, 0])
        before = self.env.get_state()
        for _ in range(3):
            self.env.step(action)
            after = self.env.get_state()
            append_transition(after, before, action, trajectory)
            append_transition(after, before, action, self.context_buffer)
            before = after

        # warmstart mppi
        for _ in range(5):
            self.global_planning_step += 1
            action = self.mppi.command(self.dynamics_and_cost_func, goal=goal)

        for t in range(50):
            after = self.env.get_state()
            append_transition(after, before, action, trajectory)
            append_transition(after, before, action, self.context_buffer)

            rr.log_arrow('action/xy',
                         before['robot_pos'],
                         np.array([action[0], action[1], 0]),
                         color=(255, 0, 255),
                         width_scale=0.01)
            rrr.log_rotational_velocity('action/z', before['robot_pos'], action[2], color=(255, 0, 255),
                                        stroke_width=0.01)

            self.env.step(action, time_offset=self.rr_time_offset)

            if goal_satisfied(before, goal, self.goal_threshold):
                print(f'Goal reached in {t} steps')
                return True, trajectory

            if before['object_pos'][2] < 0.0:
                print('Object fell')
                return False, trajectory

            self.mppi.roll()
            for t in range(1):
                self.global_planning_step += 1
                action = self.mppi.command(self.dynamics_and_cost_func, goal=goal)

            before = after

        print('Goal not reached')
        return False, trajectory

    def cost(self, object_sequences, goal):
        """
        Compute the cost of the object sequences.

        Args:
            object_sequences: [num_samples, P, 3]
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
            perturbed_action: [num_samples, P, 3]
        """
        # context [num_samples, H, 9] flattened to [num_samples, 27]
        # actions [num_samples, P, 3] flattened to [num_samples, 24]
        if len(self.context_buffer) > 3:
            self.context_buffer.pop(0)

        num_samples = perturbed_action.shape[0]
        context = get_context(self.context_buffer).float()
        context = torch.tile(context, [num_samples, 1])
        perturbed_action_flat = torch.tensor(perturbed_action).float().reshape(num_samples, -1)
        predictions = self.dynamics(context, perturbed_action_flat)  # [num_samples, P, 3] flattened to [num_samples, 24]
        object_sequences = predictions.reshape(num_samples, P, 3).cpu().detach().numpy()
        costs = self.cost(object_sequences, goal)
        if self.mde is not None:
            mde_costs = self.mde_costs(context, perturbed_action_flat, predictions)
            binary_mde_costs = np.maximum(mde_costs - 0.02, 0)
        else:
            binary_mde_costs = 0

        return costs + binary_mde_costs * 100

    def mde_costs(self, context, actions, predictions):
        deviations = self.mde(context, actions, predictions)
        deviations = deviations.detach().numpy()
        return deviations

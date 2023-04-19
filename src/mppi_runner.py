import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rerun as rr
import wandb

import rrr
from env import Env
from model import DynamicsNetwork
from mujoco_mppi import MujocoMPPI, normalized
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


T = 3


class MPPIRunner:

    def __init__(self):
        self.env = Env('source.xml')

        # get the source dynamics model
        api = wandb.Api()
        artifact = api.artifact('armlab/pushing_focus/source_model:latest')
        model_path = Path(artifact.download())
        self.model = DynamicsNetwork.load_from_checkpoint(model_path / 'model.ckpt')

        # initialize rerun
        rrr.init()
        self.rr_time_offset = 0

    def reload_env(self):
        self.rr_time_offset += self.env.data.time

        # re-initialize the target simulation environment
        self.env = Env('source.xml')

    def run(self, goal):
        goal_threshold = 0.05
        global_planning_step = 0
        rr.set_time_sequence('planning', global_planning_step)
        rr.log_point('goal', goal, color=(255, 0, 255), radius=goal_threshold)

        pool = ThreadPoolExecutor(multiprocessing.cpu_count() - 1)

        noise_sigma = np.array([0.04, 0.04, 0.1])
        mppi = MujocoMPPI(pool, self.env.model, num_samples=25, noise_sigma=noise_sigma, horizon=T, lambda_=8, seed=0)

        def _get_results(_, data):
            state = self.env.get_state(data=data)
            return state['object_pos'], state['robot_pos']

        def _cost(results, action_sequences):
            nonlocal global_planning_step
            object_sequences, robot_sequences = results

            goal_cost = np.linalg.norm(object_sequences - goal, axis=-1)[:, 1:]
            # encourage keeping the circle inside the fingers by penalizing
            # distance between the robot and the object
            robot_object_distance = np.linalg.norm(object_sequences - robot_sequences, axis=-1)[:, 1:]

            cost_viz = np.sum(goal_cost, axis=-1)
            cost_viz = normalized(cost_viz)

            for i in np.argsort(cost_viz):
                object_sequence = object_sequences[i]
                robot_sequence = robot_sequences[i]
                c = cost_viz[i]
                rr.set_time_sequence('planning', global_planning_step)
                global_planning_step += 1

                obj_color = tuple((np.array([255, 0, 0]) * (1 - c)).astype(np.int32))
                robot_color = tuple((np.array([0, 255, 0]) * (1 - c)).astype(np.int32))
                rr.log_line_strip(f'object/pred', object_sequence, color=obj_color, stroke_width=0.002)
                rr.log_line_strip(f'robot/pred', robot_sequence, color=robot_color, stroke_width=0.002)
            return goal_cost + robot_object_distance * 0.1

        def _dynamics_and_cost_func(perturbed_action):
            results = parallel_rollout(pool, self.env.model, self.env.data, perturbed_action, _get_results)
            costs = _cost(results, perturbed_action)
            return costs

        # warmstart mppi
        for t in range(5):
            global_planning_step += 1
            action = mppi.command(_dynamics_and_cost_func)

        for t in range(50):
            state = self.env.get_state()

            rr.log_arrow('action/xy',
                         state['robot_pos'],
                         np.array([action[0], action[1], 0]),
                         color=(255, 0, 255),
                         width_scale=0.01)
            rrr.log_rotational_velocity('action/z', state['robot_pos'], action[2], color=(255, 0, 255),
                                        stroke_width=0.01)

            self.env.step(action, time_offset=self.rr_time_offset)

            if goal_satisfied(state, goal, goal_threshold):
                print(f'Goal reached in {t} steps')
                break

            mppi.roll()
            for t in range(1):
                global_planning_step += 1
                action = mppi.command(_dynamics_and_cost_func)

import argparse
import pickle
import time
from copy import copy
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

import wandb
import rerun as rr
import rrr

SIM_STEPS_PER_CONTROL_STEP = 50
SIM_STEPS_PER_VIZ = 10


class Env:

    def __init__(self, model_xml_filename):
        self.model = mujoco.MjModel.from_xml_path(model_xml_filename)
        self.data = mujoco.MjData(self.model)

    def step(self, action: Optional[np.ndarray]):
        # 2d plots
        rr.log_scalar("curves/robot_x_vel", self.data.qvel[0], label="robot x vel")
        rr.log_scalar("curves/robot_y_vel", self.data.qvel[1], label="robot y vel")

        if action is not None:
            np.copyto(self.data.ctrl, action)
        for sim_step_i in range(SIM_STEPS_PER_CONTROL_STEP):
            mujoco.mj_step(self.model, self.data)
            if sim_step_i % SIM_STEPS_PER_VIZ == 0:
                self.viz()

    def viz(self):
        # 3d geometry
        for body_idx in range(self.model.nbody):
            body = self.model.body(body_idx)
            for geom_idx in np.arange(body.geomadr, (body.geomadr + body.geomnum)):
                geom = self.model.geom(geom_idx)
                self.viz_geom(body.name, geom, geom_idx)

    def viz_geom(self, body_name, geom, geom_idx):
        match geom.type:
            case mujoco.mjtGeom.mjGEOM_PLANE:
                rrr.mesh_plane(body_name, geom, self.data.geom(geom_idx))
            case mujoco.mjtGeom.mjGEOM_CYLINDER:
                rrr.mesh_cylinder(body_name, geom, self.data.geom(geom_idx))
            case mujoco.mjtGeom.mjGEOM_SPHERE:
                rrr.mesh_sphere(body_name, geom, self.data.geom(geom_idx))
            case mujoco.mjtGeom.mjGEOM_BOX:
                rrr.mesh_box(body_name, geom, self.data.geom(geom_idx))
            case _:
                raise NotImplementedError(f'{geom.type=} not implemented')

    def get_state(self):
        return {
            'robot_pos': copy(self.data.body('robot').xpos),
            'object_pos': copy(self.data.body('object').xpos),
        }


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=250)

    parser = argparse.ArgumentParser()
    parser.add_argument('model_xml_filename', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-transitions', type=int, default=1000)

    args = parser.parse_args()

    rrr.init()

    env_name = Path(args.model_xml_filename).stem
    dataset = []

    rng = np.random.RandomState(args.seed)

    # run a bunch of smaller push episodes
    while len(dataset) < args.n_transitions:
        episode_t0 = time.time()

        # re-initialize the simulation
        env = Env(args.model_xml_filename)
        env.step(None)
        before = env.get_state()

        # sample a nominal velocity
        nominal_x_vel = rng.uniform(0.02, 0.1)
        nominal_y_vel = rng.normal(0, 0.02)

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
            }

            episode.append(transition)

            before = after

        dataset.append(episode)

        print(f'episode {len(dataset)} took {time.time() - episode_t0:.2f}s')

    with open('dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    run = wandb.init(project='pushing_focus', entity='armlab', config={
        'env_name': env_name,
        'seed': args.seed,
        'n_transitions': args.n_transitions,
    })

    artifact = wandb.Artifact(f'{env_name}-dataset', type='dataset')
    artifact.add_file('dataset.pkl')
    run.log_artifact(artifact)

    wandb.finish()


if __name__ == '__main__':
    main()

import argparse
import pickle
import time
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import rerun as rr
import wandb
from mujoco._structs import _MjModelGeomViews, _MjDataGeomViews
from scipy.spatial.transform import Rotation
from trimesh.creation import box, cylinder

SIM_STEPS_PER_CONTROL_STEP = 50


def name(*names):
    """ joins names with slashes but ignores empty names """
    return '/'.join([name for name in names if name])


def get_trasnform(data: _MjDataGeomViews):
    xmat = data.xmat.reshape(3, 3)
    transform = np.eye(4)
    transform[0:3, 0:3] = xmat
    transform[0:3, 3] = data.xpos
    return transform


def rr_mesh_plane(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    transform = get_trasnform(data)
    mesh = box(np.array([model.size[0], model.size[1], 0.001]), transform)
    rr.log_mesh(entity_path=name(body_name, model.name),
                positions=mesh.vertices,
                indices=mesh.faces,
                albedo_factor=model.rgba)


def rr_mesh_cylinder(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    transform = get_trasnform(data)
    mesh = cylinder(radius=model.size[0], height=2 * model.size[1], transform=transform, sections=16)

    rr.log_line_segments(entity_path=name(body_name, model.name, 'outline'),
                         positions=mesh.vertices[mesh.edges_unique.flatten()],
                         color=(0, 0, 0))
    rr.log_mesh(entity_path=name(body_name, model.name),
                positions=mesh.vertices,
                indices=mesh.faces,
                albedo_factor=model.rgba)


def rr_mesh_sphere(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    rr.log_point(entity_path=name(body_name, model.name),
                 position=data.xpos,
                 radius=model.size[0],
                 color=tuple(model.rgba))


def rr_mesh_box(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    transform = get_trasnform(data)
    q = Rotation.from_matrix(transform[:3, :3]).as_quat()
    mesh = box(model.size * 2, transform)

    rr.log_obb(entity_path=name(body_name, model.name, 'outline'),
               half_size=model.size * 2,
               position=data.xpos,
               rotation_q=q,
               color=(0, 0, 0))
    rr.log_mesh(entity_path=name(body_name, model.name),
                positions=mesh.vertices,
                indices=mesh.faces,
                albedo_factor=model.rgba)


def rr_init():
    rr.init('simulation')
    rr.connect()
    rr.log_view_coordinates("world", up="+Z", timeless=True)
    rr.log_arrow('world_x', [0, 0, 0], [1, 0, 0], color=(255, 0, 0), width_scale=0.02)
    rr.log_arrow('world_y', [0, 0, 0], [0, 1, 0], color=(0, 255, 0), width_scale=0.02)
    rr.log_arrow('world_z', [0, 0, 0], [0, 0, 1], color=(0, 0, 255), width_scale=0.02)


class Env:

    def __init__(self, model_xml_filename):
        self.model = mujoco.MjModel.from_xml_path(model_xml_filename)
        self.data = mujoco.MjData(self.model)

    def step(self, action: Optional[np.ndarray]):
        if action is not None:
            np.copyto(self.data.ctrl, action)
        for _ in range(SIM_STEPS_PER_CONTROL_STEP):
            mujoco.mj_step(self.model, self.data)
            self.viz()

    def viz(self):
        # 2d plots
        rr.log_scalar("curves/robot_x_vel", self.data.qvel[0], label="robot x vel")
        rr.log_scalar("curves/robot_y_vel", self.data.qvel[1], label="robot y vel")

        # 3d geometry
        for body_idx in range(self.model.nbody):
            body = self.model.body(body_idx)
            for geom_idx in np.arange(body.geomadr, (body.geomadr + body.geomnum)):
                geom = self.model.geom(geom_idx)
                self.viz_geom(body.name, geom, geom_idx)

    def viz_geom(self, body_name, geom, geom_idx):
        match geom.type:
            case mujoco.mjtGeom.mjGEOM_PLANE:
                rr_mesh_plane(body_name, geom, self.data.geom(geom_idx))
            case mujoco.mjtGeom.mjGEOM_CYLINDER:
                rr_mesh_cylinder(body_name, geom, self.data.geom(geom_idx))
            case mujoco.mjtGeom.mjGEOM_SPHERE:
                rr_mesh_sphere(body_name, geom, self.data.geom(geom_idx))
            case mujoco.mjtGeom.mjGEOM_BOX:
                rr_mesh_box(body_name, geom, self.data.geom(geom_idx))
            case _:
                raise NotImplementedError(f'{geom.type=} not implemented')

    def get_state(self):
        return {
            'robot_pos': self.data.body('robot').xpos,
            'object_pos': self.data.body('object').xpos,
        }


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=250)

    parser = argparse.ArgumentParser()
    parser.add_argument('model_xml_filename', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-transitions', type=int, default=1000)

    args = parser.parse_args()

    rr_init()

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

        for _ in range(10):
            action = np.array([nominal_x_vel, nominal_y_vel]) + rng.normal(0, 0.005, size=(2,))

            env.step(action)
            after = env.get_state()

            transition = {
                'before': before,
                'action': action,
                'after': after,
            }

            rr.log_scalar('state/obj_x', before['object_pos'][0], label='obj x')
            rr.log_scalar('state/obj_y', before['object_pos'][1], label='obj y')
            rr.log_scalar('state/robot_x', before['robot_pos'][0], label='robot x')
            rr.log_scalar('state/robot_y', before['robot_pos'][1], label='robot y')

            dataset.append(transition)

            before = after

        print(f'episode took {time.time() - episode_t0:.2f}s')

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

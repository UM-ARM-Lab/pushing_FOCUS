import argparse
from typing import Optional

import mujoco
import numpy as np
import rerun as rr
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

    def __init__(self, env_type):
        self.model = mujoco.MjModel.from_xml_path('source.xml')
        self.data = mujoco.MjData(self.model)
        self.env_type = env_type

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
    parser.add_argument('env_type', type=str, choices=['source', 'target'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-transitions', type=int, default=20)

    args = parser.parse_args()

    rr_init()

    env = Env(args.env_type)

    env.step(None)

    dataset = []
    before = env.get_state()

    for transition_idx in range(args.n_transitions):
        action = np.array([0.05, 0])

        env.step(action)
        after = env.get_state()

        transition = {
            'before': before,
            'action': action,
            'after': after,
        }

        rr.log_scalar('state/obj_x', before['object_pos'][0], label='obj  x')
        rr.log_scalar('state/obj_y', before['object_pos'][1], label='obj  y')
        rr.log_scalar('state/robot_x', before['robot_pos'][0], label='robot  x')
        rr.log_scalar('state/robot_y', before['robot_pos'][1], label='robot  y')

        dataset.append(transition)

        before = after


if __name__ == '__main__':
    main()

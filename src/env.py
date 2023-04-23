from copy import copy
from typing import Optional

import mujoco
import numpy as np
import rerun as rr

import rrr
from rollout import N_SUB_STEPS

SIM_STEPS_PER_VIZ = 25


def nonempty_name(name):
    if name is None or name == '':
        return '_'
    return name


class Env:

    def __init__(self, model_xml_filename: Optional[str], model: Optional = None, data: Optional = None):
        if model is not None:
            if data is None:
                raise ValueError('data must be provided if model is provided')
            self.model = model
            self.data = data
        else:
            self.model = mujoco.MjModel.from_xml_path(model_xml_filename)
            self.data = mujoco.MjData(self.model)

        mujoco.mj_forward(self.model, self.data)

    def step(self, action: Optional[np.ndarray], log=True, time_offset=0):
        # 2d plots
        if log:
            rr.set_time_seconds('sim_time', self.data.time + time_offset)
            rr.log_scalar("curves/robot_x_vel", self.data.qvel[0], label="robot x vel")
            rr.log_scalar("curves/robot_y_vel", self.data.qvel[1], label="robot y vel")
            rr.log_scalar("curves/robot_z_vel", self.data.qvel[2], label="robot z vel")

        if action is not None:
            np.copyto(self.data.ctrl, action)
        for sim_step_i in range(N_SUB_STEPS):
            mujoco.mj_step(self.model, self.data)
            if sim_step_i % SIM_STEPS_PER_VIZ == 0:
                if log:
                    rr.set_time_seconds('sim_time', self.data.time + time_offset)
                    self.viz()

    def viz(self):
        # 3d geometry
        for body_idx in range(self.model.nbody):
            body = self.model.body(body_idx)
            for geom_idx in np.arange(body.geomadr, (body.geomadr + body.geomnum)):
                geom = self.model.geom(geom_idx)
                self.viz_geom(body.name, geom, geom_idx)
        # contacts
        for contact_idx in range(self.data.ncon):
            geom1 = self.data.contact.geom1[contact_idx]
            geom2 = self.data.contact.geom2[contact_idx]
            geom1_name = nonempty_name(self.model.geom(geom1).name)
            geom2_name = nonempty_name(self.model.geom(geom2).name)
            pos = self.data.contact.pos[contact_idx]
            depth = self.data.contact.dist[contact_idx]
            normal = self.data.contact.frame[contact_idx, :3] * min(max(depth, 0.02), 0.2)
            rr.log_arrow(f'contact/{geom1_name}/{geom2_name}', pos, pos+normal, color=(255, 0, 0, 125), width_scale=0.003)

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

    def get_state(self, data: Optional[mujoco.MjData] = None):
        if data is None:
            data = self.data
        return {
            'robot_pos': copy(data.body('robot').xpos),
            'object_pos': copy(data.body('object').xpos),
        }

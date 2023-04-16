from copy import copy
from typing import Optional

import mujoco
import numpy as np
import rerun as rr

import rrr

SIM_STEPS_PER_CONTROL_STEP = 50
SIM_STEPS_PER_VIZ = 10


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

    def step(self, action: Optional[np.ndarray], log=True):
        # 2d plots
        if log:
            rr.log_scalar("curves/robot_x_vel", self.data.qvel[0], label="robot x vel")
            rr.log_scalar("curves/robot_y_vel", self.data.qvel[1], label="robot y vel")

        if action is not None:
            np.copyto(self.data.ctrl, action)
        for sim_step_i in range(SIM_STEPS_PER_CONTROL_STEP):
            mujoco.mj_step(self.model, self.data)
            if sim_step_i % SIM_STEPS_PER_VIZ == 0:
                if log:
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

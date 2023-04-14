import numpy as np
import rerun as rr
from mujoco._structs import _MjDataGeomViews, _MjModelGeomViews
from scipy.spatial.transform import Rotation
from trimesh.creation import box, cylinder


def init():
    rr.init('simulation')
    rr.connect()
    rr.log_view_coordinates("world", up="+Z", timeless=True)
    rr.log_arrow('world_x', [0, 0, 0], [1, 0, 0], color=(255, 0, 0), width_scale=0.02)
    rr.log_arrow('world_y', [0, 0, 0], [0, 1, 0], color=(0, 255, 0), width_scale=0.02)
    rr.log_arrow('world_z', [0, 0, 0], [0, 0, 1], color=(0, 0, 255), width_scale=0.02)



def name(*names):
    """ joins names with slashes but ignores empty names """
    return '/'.join([name for name in names if name])


def get_transform(data: _MjDataGeomViews):
    xmat = data.xmat.reshape(3, 3)
    transform = np.eye(4)
    transform[0:3, 0:3] = xmat
    transform[0:3, 3] = data.xpos
    return transform


def mesh_plane(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    transform = get_transform(data)
    mesh = box(np.array([model.size[0], model.size[1], 0.001]), transform)
    rr.log_mesh(entity_path=name(body_name, model.name),
                positions=mesh.vertices,
                indices=mesh.faces,
                albedo_factor=model.rgba)


def mesh_cylinder(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    transform = get_transform(data)
    mesh = cylinder(radius=model.size[0], height=2 * model.size[1], transform=transform, sections=16)

    segment_indices = mesh.edges_unique.flatten()
    rr.log_line_segments(entity_path=name(body_name, model.name, 'outline'),
                         positions=mesh.vertices[segment_indices],
                         color=(0, 0, 0))
    rr.log_mesh(entity_path=name(body_name, model.name),
                positions=mesh.vertices,
                indices=mesh.faces,
                albedo_factor=model.rgba)


def mesh_sphere(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    rr.log_point(entity_path=name(body_name, model.name),
                 position=data.xpos,
                 radius=model.size[0],
                 color=tuple(model.rgba))


def mesh_box(body_name, model: _MjModelGeomViews, data: _MjDataGeomViews):
    transform = get_transform(data)
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

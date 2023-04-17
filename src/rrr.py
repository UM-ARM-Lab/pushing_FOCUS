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


def log_rotational_velocity(entity_name,
                            position,
                            rotational_velocity,
                            color,
                            stroke_width,
                            max_vel=1.5,
                            z=0.12,
                            radius=0.1):
    """
    Draw an arc with an arrow tip centered a position, and with a radius and length proportional to the rotational velocity.
    """
    vel_rel = rotational_velocity / max_vel
    angles = np.linspace(0, 2 * np.pi * vel_rel, 16)
    arc_xs = position[0] + np.cos(angles) * radius
    arc_ys = position[1] + np.sin(angles) * radius
    arc_positions = np.stack([arc_xs, arc_ys, np.ones_like(arc_xs) * z], axis=1)
    # main body of the arrow
    rr.log_line_strip(entity_name + '/arc', arc_positions, color=color, stroke_width=stroke_width)
    # arrow tips
    tip_positions = [
        position + (arc_positions[-6] - position) * 0.8,
        position + (arc_positions[-6] - position) * 1.2,
    ]
    tip_positions[0][2] = z
    tip_positions[1][2] = z
    rr.log_line_segments(entity_name + '/tip',
                         [arc_positions[-1], tip_positions[0], arc_positions[-1], tip_positions[1]], color=color,
                         stroke_width=stroke_width)


def viz_state(before):
    rr.log_scalar('state/obj_x', before['object_pos'][0], label='obj x')
    rr.log_scalar('state/obj_y', before['object_pos'][1], label='obj y')
    rr.log_scalar('state/robot_x', before['robot_pos'][0], label='robot x')
    rr.log_scalar('state/robot_y', before['robot_pos'][1], label='robot y')

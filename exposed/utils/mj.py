"""
Utilities for fishing data of the mujoco simulator
"""
import exposed.utils.m3d as m3d

# TODO: Maybe use xmat for orientatino instead of xquat. (or not?)

def get_pos_vec3(physics, object, frame=None):
    """Grabs position, relative to frame if requested"""
    binding = physics.bind(object)
    pos_vec3 = m3d.Vector3(binding.xpos)
    if not frame is None:
        # If frame, apply rigid transform
        frame_rmat = get_orn_rmat(physics, frame)
        frame_vec3 = get_pos_vec3(physics, frame)
        pos_vec3 = frame_rmat.inverse @ pos_vec3 - frame_vec3
    return pos_vec3

def get_dir_vec3(physics, object_a, object_b, frame=None, normalize=False):
    pos_vec3_a = get_pos_vec3(physics, object_a)
    pos_vec3_b = get_pos_vec3(physics, object_b)
    dir_vec3 = pos_vec3_b - pos_vec3_a
    if not frame is None:
        frame_rmat = get_orn_rmat(physics, frame)
        dir_vec3 = frame_rmat.inverse @ dir_vec3
    if normalize:
        dir_vec3 = dir_vec3.normalized
    return dir_vec3

def get_orn_rmat(physics, object, frame=None):
    """Grabs orientation as a rotation matrix, possibly relative to frame"""
    binding = physics.bind(object)
    orn_rmat = m3d.RotationMatrix(binding.xmat.reshape(3, 3))
    if not frame is None:
        frame_rmat = get_orn_rmat(physics, frame)
        orn_rmat = frame_rmat.inverse @ orn_rmat
    return orn_rmat

def get_orn_quat(physics, object, frame=None):
    binding = physics.bind(object)
    if object.tag == 'site':
        orn_quat = m3d.RotationMatrix(binding.xmat.reshape(3, 3)).quat
    else:
        orn_quat = m3d.Quaternion(binding.xquat)
    if not frame is None:
        frame_quat = get_orn_quat(physics, frame)
        quat = frame_quat.transpose() * orn_quat
    return orn_quat.upper_hemisphere.normalized

def get_lin_vel_vec3(physics, object, frame=None):
    binding = physics.bind(object)
    lin_vel_vec3 = m3d.Vector3(binding.cvel[3:])
    if not frame is None:
        frame_rmat = get_orn_rmat(physics, frame)
        vec3 = frame_rmat.inverse @ lin_vel_vec3
    return lin_vel_vec3

def get_ang_vel_vec3(physics, object, frame=None):
    binding = physics.bind(object)
    ang_vel_vec3 = m3d.Vector3(binding.cvel[:3])
    if not frame is None:
        frame_rmat = get_orn_rmat(physics, frame)
        vec3 = frame_rmat.inverse @ ang_vel_vec3
    return ang_vel_vec3

def get_distance(physics, object_a, object_b):
    return get_dir_vec3(physics, object_a, object_b).length

def get_angle(physics, object_a, object_b):
    quat_a = get_orn_quat(physics, object_a)
    quat_b = get_orn_quat(physics, object_b)
    return (quat_a * quat_b.conjugate()).upper_hemisphere.normalized.angle

def get_lin_speed(physics, object):
    return get_lin_vel_vec3(physics, object).length

def get_ang_speed(physics, object):
    return get_ang_vel_vec3(physics, object).length

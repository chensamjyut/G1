"""Four-DoF vocabulary and the ONLY task/native coordinate adapter.

Task X is right, Y forward. Native G1 X is forward, Y left, Z up.
STOP keeps legacy index 4. Existing two-axis ACTIONS remains unchanged.
"""
from enum import IntEnum
import math
import numpy as np
from .action_space import ACTIONS

class Action(IntEnum):
    PAN_LEFT = 0
    PAN_RIGHT = 1
    TILT_UP = 2
    TILT_DOWN = 3
    STOP = 4
    MOVE_X_POS = 5
    MOVE_X_NEG = 6
    MOVE_Y_POS = 7
    MOVE_Y_NEG = 8

ACTION_NAMES = ACTIONS + ("MoveXPos", "MoveXNeg", "MoveYPos", "MoveYNeg")
TRANSLATIONS = {Action.MOVE_X_POS: (1, 0), Action.MOVE_X_NEG: (-1, 0),
                Action.MOVE_Y_POS: (0, 1), Action.MOVE_Y_NEG: (0, -1)}
INVERSES = {Action.PAN_LEFT: Action.PAN_RIGHT, Action.PAN_RIGHT: Action.PAN_LEFT,
            Action.TILT_UP: Action.TILT_DOWN, Action.TILT_DOWN: Action.TILT_UP,
            Action.MOVE_X_POS: Action.MOVE_X_NEG, Action.MOVE_X_NEG: Action.MOVE_X_POS,
            Action.MOVE_Y_POS: Action.MOVE_Y_NEG, Action.MOVE_Y_NEG: Action.MOVE_Y_POS}

def parse_action(value):
    if isinstance(value, str):
        return Action(ACTION_NAMES.index(value))
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer, Action)):
        raise ValueError(f"invalid discrete action: {value!r}")
    return Action(value)

def task_velocity_to_native(vx_lateral, vy_forward):
    values = np.asarray([vx_lateral, vy_forward], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite velocity")
    return np.array([values[1], -values[0]])

def native_to_task(x_forward, y_left):
    return np.array([-y_left, x_forward], dtype=float)

def world_to_task(vector, yaw):
    x, y = np.asarray(vector)[:2]
    c, s = math.cos(yaw), math.sin(yaw)
    return native_to_task(c*x+s*y, -s*x+c*y)

def yaw_from_quaternion(q):
    w, x, y, z = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))

def limited_velocity(request, previous, dt, cfg):
    request = np.asarray(request, dtype=float)
    if request.shape != (2,) or not np.isfinite(request).all():
        raise ValueError("velocity must contain two finite values")
    request = np.clip(request, -cfg['max_speed_m_s'], cfg['max_speed_m_s'])
    norm = np.linalg.norm(request)
    if norm > cfg['max_speed_m_s']:
        request *= cfg['max_speed_m_s']/norm
    request[np.abs(request) < cfg['dead_zone_m_s']] = 0
    delta = request - np.asarray(previous)
    norm = np.linalg.norm(delta)
    limit = cfg['acceleration_m_s2']*dt
    if norm > limit:
        delta *= limit/norm
    return np.asarray(previous)+delta

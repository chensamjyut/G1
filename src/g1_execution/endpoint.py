import math,numpy as np
from .servo import FootReferencedExpertExecutor

def endpoint(action, s, cfg):
    xy = np.array(s['position'][:2])
    yaw = s['yaw']
    pitch = s['tilt_deg']
    axes = {'MoveXPos': (1, 0), 'MoveXNeg': (-1, 0), 'MoveYPos': (0, 1), 'MoveYNeg': (0, -1)}
    if action in axes:
        (dx, dy) = axes[action]
        xy += cfg['translation_step_m'] * np.array([math.sin(yaw) * dx + math.cos(yaw) * dy, -math.cos(yaw) * dx + math.sin(yaw) * dy])
    if action.startswith('BodyYaw'):
        yaw += math.radians(cfg['yaw_step_deg']) * (1 if action == 'BodyYawLeft' else -1)
    if action.startswith('GimbalPitch'):
        pitch += cfg['pitch_step_deg'] * (-1 if action == 'GimbalPitchUp' else 1)
    return {'xy': xy.tolist(), 'yaw': yaw, 'pitch': pitch}

class PolicyExecutor(FootReferencedExpertExecutor):

    def verify_target_view(self, expected):
        return {'required': False, 'passed': True, 'reason': 'relative primitive, no goal feedback'}

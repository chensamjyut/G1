from pathlib import Path
from types import SimpleNamespace
import math,re,numpy as np,yaml
from .body_xy import world_to_task, yaw_from_quaternion,limited_velocity,task_velocity_to_native

def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg['_config_path'] = str(path)
    cfg['sim_project'] = str((path.parent / cfg['sim_project']).resolve())
    cfg['robot'] = yaml.safe_load((path.parent / cfg['robot_config']).read_text())
    return cfg

def _rpy_from_quaternion(q):
    (w, x, y, z) = [float(v) for v in q]
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (roll, pitch, yaw)

class BodyXYRuntime:

    def read_state(self):
        from pxr import UsdGeom, Gf
        print('[round5] read_state_robot_start', flush=True)
        robot = self.env.scene['robot']
        d = robot.data
        values = d.root_state_w[0].detach().cpu().numpy().astype(float)
        if not np.isfinite(values).all():
            raise RuntimeError('non-finite robot state')
        yaw = yaw_from_quaternion(values[3:7])
        gravity = d.projected_gravity_b[0].detach().cpu().numpy()
        fall = bool(values[2] < self.cfg['safety']['min_height_m'] or -gravity[2] < self.cfg['safety']['min_upright_cos'])
        print('[round5] read_state_contact_start', flush=True)
        sensor = self.env.scene['contact_forces']
        forces = sensor.data.net_forces_w[0].detach().cpu().numpy()
        names = sensor.body_names
        mask = [not re.fullmatch(self.cfg['safety']['contact_ignore_regex'], n) for n in names]
        collision = bool(np.any(np.linalg.norm(forces[mask], axis=-1) > self.cfg['safety']['collision_force_n']))
        print('[round5] read_state_camera_start', flush=True)
        prim = self.stage.GetPrimAtPath(self.cfg['robot']['camera_prim_path'])
        if not prim or not prim.IsValid():
            raise RuntimeError('required_gimbal_camera_prim_missing')
        if self.controller is None:
            raise RuntimeError('required_gimbal_controller_missing')
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim) if prim and prim.IsValid() else None
        forward = matrix.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized() if matrix else Gf.Vec3d(0, 0, 0)
        camera_position = np.asarray(list(matrix.ExtractTranslation()), dtype=float) if matrix else values[:3].copy()
        try:
            sensor_pos = self.env.scene['inspection_camera'].data.pos_w[0].detach().cpu().numpy()
            if np.isfinite(sensor_pos).all():
                camera_position = sensor_pos.astype(float)
        except (AttributeError, KeyError, IndexError):
            pass
        print('[round5] read_state_joint_start', flush=True)
        joints = self.controller.read_state() if self.controller is not None else SimpleNamespace(pan_deg=0.0, tilt_deg=0.0)
        return dict(position=values[:3].tolist(), quaternion=values[3:7].tolist(), yaw=yaw, velocity_body=world_to_task(values[7:10], yaw).tolist(), yaw_rate=float(values[12]), pan_deg=joints.pan_deg, tilt_deg=joints.tilt_deg, camera_position=camera_position.tolist(), camera_forward=list(forward), collision=collision, fall_detected=fall, stable=not (fall or collision))

    def _velocity_tick(self, request, yaw_rate=0):
        tick_cfg = dict(self.cfg['control'])
        if np.linalg.norm(request) < np.linalg.norm(self.velocity):
            tick_cfg['acceleration_m_s2'] = tick_cfg.get('deceleration_m_s2', tick_cfg['acceleration_m_s2'])
        self.velocity = limited_velocity(request, self.velocity, self.dt, tick_cfg)
        native = task_velocity_to_native(*self.velocity)
        self.raw_yaw_rate_command = float(yaw_rate)
        correction = 0.0
        self.heading_error = 0.0
        hh = self.cfg['control'].get('heading_hold', {})
        enabled = bool(hh.get('enabled', False)) and self.heading_hold_active
        elapsed = float(self.env.sim.current_time) - self.primitive_start_time
        enabled = enabled and elapsed >= float(hh.get('correction_start_delay_s', 0))
        if self.phase == 'command' and elapsed >= self.primitive_duration - float(hh.get('correction_end_before_brake_s', 0)):
            enabled = False
        if enabled and (self.phase != 'brake' or bool(hh.get('brake_yaw_control_enabled', True))):
            q = self.env.scene['robot'].data.root_state_w[0, 3:7].detach().cpu().tolist()
            current_yaw = yaw_from_quaternion(q)
            ref = float(self.heading_reference_yaw if self.heading_reference_yaw is not None else current_yaw)
            self.heading_error = (ref - current_yaw + math.pi) % (2 * math.pi) - math.pi
            dead = float(hh.get('yaw_error_deadzone_rad', 0.01))
            if abs(self.heading_error) > dead:
                current_rate = float(self.env.scene['robot'].data.root_ang_vel_b[0, 2].detach().cpu())
                kp = float(hh.get('kp_yaw', 0.0))
                kd = float(hh.get('kd_yaw', 0.0))
                limit = float(hh.get('max_yaw_rate_correction', 0.0))
                correction = max(-limit, min(limit, kp * self.heading_error - kd * current_rate))
        self.heading_correction = correction
        self.command = [float(native[0]), float(native[1]), float(yaw_rate + correction), self.cfg['control']['height_m']]
        self.step()

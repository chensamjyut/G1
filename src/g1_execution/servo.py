import contextlib,math,numpy as np
from .state import Executor
from .angles import angle_delta
from gimbal_tvr.body_xy import world_to_task

class PhysicalExpertExecutor(Executor):

    def realized_gimbal(self):
        from pxr import UsdGeom, Gf
        tilt = UsdGeom.Xformable(self.rt.stage.GetPrimAtPath(self.rt.controller.tilt_path)).GetLocalTransformation().TransformDir(Gf.Vec3d(1, 0, 0))
        pan = UsdGeom.Xformable(self.rt.stage.GetPrimAtPath(self.rt.controller.pan_path)).GetLocalTransformation().TransformDir(Gf.Vec3d(1, 0, 0))
        return (math.degrees(math.atan2(-tilt[2], tilt[0])), math.degrees(math.atan2(pan[1], pan[0])))

    def state(self):
        state = super().state()
        (pitch, pan) = self.realized_gimbal()
        state['commanded_tilt_deg'] = state['tilt_deg']
        state['tilt_deg'] = pitch
        state['pan_deg'] = pan
        state['gimbal_state_source'] = 'realized local Xform orientation; kinematic simulated actuator, not physical joint encoder'
        return state

    def act(self, action, expected):
        from g1_active_view.actions import ACTIONS
        if action not in ACTIONS or not self.locked:
            raise ValueError('invalid action or uninitialized executor')
        rt = self.rt
        before = self.state()
        start = float(rt.env.sim.current_time)
        rows = []
        collision = before['collision']
        fall = before['fall_detected']
        obstacle = False
        completed = False
        reason = None
        if action == 'Stop':
            rt.command = [0.0, 0.0, 0.0, rt.cfg['control']['height_m']]
            rt.velocity[:] = 0
            obstacle = not self.ground.free(*before['position'][:2])['legal']
            completed = np.linalg.norm(before['velocity_body']) < 0.05 and abs(before['yaw_rate']) < 0.04
        elif action.startswith('GimbalPitch'):
            rt.controller.increment('TiltUp' if action == 'GimbalPitchUp' else 'TiltDown', self.cfg['pitch_step_deg'])
            required = int(rt.controller.config['settle_frames'])
            threshold = float(rt.controller.config['settle_velocity_threshold'])
            stable = 0
            previous = self.realized_gimbal()[0]
            for tick in range(120):
                with contextlib.redirect_stdout(self.null):
                    rt._velocity_tick((0, 0), 0)
                current = self.realized_gimbal()[0]
                state = self.state()
                speed = abs(current - previous) / rt.dt
                previous = current
                obstacle |= not self.ground.free(*state['position'][:2])['legal']
                stable = stable + 1 if abs(current - expected['pitch']) <= self.cfg['pitch_tolerance_deg'] and speed <= threshold / rt.dt else 0
                rows.append({'time': float(rt.env.sim.current_time), 'phase': 'gimbal_settle', 'request_task_velocity': [0.0, 0.0], 'yaw_rate_command': 0.0, 'pitch': current, 'pitch_speed_deg_s': speed, 'state': state})
                collision |= state['collision']
                fall |= state['fall_detected']
                if stable >= required:
                    completed = True
                    break
        else:
            integral = np.zeros(2)
            iyaw = 0.0
            stable = 0
            phase = 'servo'
            settled = 0
            for tick in range(math.ceil(45.0 / rt.dt)):
                state = self.state()
                error = np.array(expected['xy']) - state['position'][:2]
                yawerr = angle_delta(expected['yaw'], state['yaw'])
                velocity = np.array(state['velocity_body'])
                err_task = world_to_task(error, state['yaw'])
                if phase == 'servo':
                    integral = np.clip(integral + error * rt.dt, -0.4, 0.4)
                    iyaw = float(np.clip(iyaw + yawerr * rt.dt, -0.7, 0.7))
                    request = world_to_task(error * 2.2 + integral * 1.8, state['yaw']) - 0.65 * velocity
                    if np.linalg.norm(request) > 0.5:
                        request *= 0.5 / np.linalg.norm(request)
                    yawcmd = float(np.clip(4 * yawerr + 1.5 * iyaw - 0.7 * state['yaw_rate'], -1.2, 1.2))
                    near = np.linalg.norm(error) < 0.006 and abs(yawerr) < math.radians(0.15) and (np.linalg.norm(velocity) < 0.018) and (abs(state['yaw_rate']) < 0.015)
                    stable = stable + 1 if near else 0
                    if stable >= 10:
                        phase = 'settle'
                        settled = 0
                if phase == 'settle':
                    request = np.zeros(2)
                    yawcmd = 0.0
                    settled += 1
                rt.cfg['control']['height_m'] = self.cfg.get('locomotion_height_m', 0.8) if phase == 'servo' else 0.8
                with contextlib.redirect_stdout(self.null):
                    rt._velocity_tick(request, yawcmd)
                post = self.state()
                collision |= post['collision']
                fall |= post['fall_detected']
                obstacle |= not self.ground.free(*post['position'][:2])['legal']
                rows.append({'time': float(rt.env.sim.current_time), 'phase': phase, 'height_command_m': rt.cfg['control']['height_m'], 'request_task_velocity': request.tolist(), 'yaw_rate_command': yawcmd, 'position': post['position'], 'yaw': post['yaw'], 'velocity': post['velocity_body']})
                if collision or fall or obstacle:
                    reason = 'unsafe'
                    break
                if np.linalg.norm(np.array(post['position'][:2]) - before['position'][:2]) > self.cfg.get('max_primitive_travel_m', 0.45):
                    reason = 'primitive_travel_limit'
                    break
                if phase == 'settle' and settled >= math.ceil(1.2 / rt.dt):
                    xyerr = np.linalg.norm(np.array(post['position'][:2]) - expected['xy'])
                    ye = abs(angle_delta(post['yaw'], expected['yaw']))
                    if xyerr < 0.015 and ye < math.radians(0.5):
                        completed = True
                        break
                    phase = 'servo'
                    stable = 0
            rt.cfg['control']['height_m'] = 0.8
            if not completed:
                for _ in range(math.ceil(0.8 / rt.dt)):
                    with contextlib.redirect_stdout(self.null):
                        rt._velocity_tick((0, 0), 0)
        after = self.state()
        xe = float(np.linalg.norm(np.array(after['position'][:2]) - expected['xy']))
        ye = abs(math.degrees(angle_delta(after['yaw'], expected['yaw'])))
        pe = abs(after['tilt_deg'] - expected['pitch'])
        ok = bool(completed and (not (collision or fall or obstacle)) and (xe <= self.cfg['primitive_xy_tolerance_m']) and (ye <= self.cfg['primitive_yaw_tolerance_deg']) and (pe <= self.cfg['pitch_tolerance_deg']))
        return {'action': action, 'success': ok, 'action_completed': bool(completed), 'collision': collision, 'fall': fall, 'scene_envelope_collision': obstacle, 'before': before, 'after': after, 'expected_node': expected, 'xy_error_m': xe, 'yaw_error_deg': ye, 'pitch_error_deg': pe, 'controls': rows, 'duration_s': float(rt.env.sim.current_time) - start, 'root_writes': 0, 'failure_reason': reason or (None if ok else 'servo_not_converged'), 'execution_mode': 'coupled_position_velocity_servo', 'gimbal_readback_source': 'realized local Xform orientation; kinematic simulated actuator, not physical encoder'}

class FootReferencedExpertExecutor(PhysicalExpertExecutor):
    """Drive support placement, so temporary torso lean is not goal progress."""

    def support(self):
        from gimbal_tvr.body_xy import yaw_from_quaternion
        robot = self.robot
        ids = [i for (i, n) in enumerate(robot.body_names) if n in ['left_ankle_roll_link', 'right_ankle_roll_link']]
        if len(ids) != 2:
            raise RuntimeError('missing measured support links')
        pos = robot.data.body_pos_w[0, ids].detach().cpu().numpy()
        quat = robot.data.body_quat_w[0, ids].detach().cpu().numpy()
        ys = [yaw_from_quaternion(q) for q in quat]
        return {'xy': pos[:, :2].mean(axis=0), 'yaw': math.atan2(sum((math.sin(y) for y in ys)), sum((math.cos(y) for y in ys))), 'feet_xyz': pos.tolist(), 'feet_yaw': ys}

    def act(self, action, expected):
        if action == 'Stop' or action.startswith('GimbalPitch'):
            return super().act(action, expected)
        rt = self.rt
        before = self.state()
        support0 = self.support()
        angle = angle_delta(expected['yaw'], before['yaw'])
        (c, s) = (math.cos(angle), math.sin(angle))
        rotation = np.array([[c, -s], [s, c]])
        goal = np.array(expected['xy']) + rotation @ (support0['xy'] - before['position'][:2])
        goal_yaw = support0['yaw'] + angle
        integral = np.zeros(2)
        iyaw = 0.0
        rows = []
        collision = fall = obstacle = False
        completed = False
        reason = None
        start = float(rt.env.sim.current_time)
        phase = 'lower'
        settled = 0
        attempts = 0
        lowered = 0
        goal_stable = 0
        previous_rest_error = None
        previous_rest_yaw_error = None
        lower_ticks = math.ceil(self.cfg.get('support_height_ramp_s', 1.2) / rt.dt)
        brake_ticks = math.ceil(self.cfg.get('support_brake_hold_s', 0.8) / rt.dt)
        ramp_ticks = math.ceil(self.cfg.get('support_height_ramp_s', 1.2) / rt.dt)
        settle_ticks = brake_ticks + ramp_ticks + math.ceil(self.cfg.get('support_rest_s', 3.0) / rt.dt)
        for tick in range(math.ceil(self.cfg.get('support_max_seconds', 65.0) / rt.dt)):
            state = self.state()
            support = self.support()
            error = goal - support['xy']
            yawerr = angle_delta(goal_yaw, support['yaw'])
            if phase == 'servo':
                integral = np.clip(integral + error * rt.dt, -0.35, 0.35)
                iyaw = float(np.clip(iyaw + yawerr * rt.dt, -0.8, 0.8))
                request = world_to_task(error * 2.5 + integral * 0.8, state['yaw']) - 0.25 * np.array(state['velocity_body'])
                speed_limit = min(self.cfg.get('support_max_speed_m_s', 0.35), max(self.cfg.get('support_approach_min_cap_m_s', 0.15), self.cfg.get('support_approach_gain_s_inv', 1.5) * float(np.linalg.norm(error))))
                if np.linalg.norm(request) > speed_limit:
                    request *= speed_limit / np.linalg.norm(request)
                yawcmd = float(np.clip(yawerr * 4.0 + iyaw * 1.2 - 0.4 * state['yaw_rate'], -1.2, 1.2))
                if np.linalg.norm(error) < min(self.cfg['primitive_xy_tolerance_m'] * 0.98, self.cfg.get('support_brake_xy_m', 0.015) if attempts == 0 else max(0.004, min(self.cfg.get('support_brake_xy_m', 0.015), 0.25 * last_rest_xy_error))) and abs(yawerr) < math.radians(min(self.cfg['primitive_yaw_tolerance_deg'] * 0.8, 0.5 if attempts == 0 else max(0.15, min(0.5, 0.25 * last_rest_yaw_error_deg)))):
                    phase = 'settle'
                    settled = 0
                    brake_support = support
            if phase == 'lower':
                request = np.zeros(2)
                yawcmd = 0.0
                lowered += 1
            if phase == 'settle':
                request = np.zeros(2)
                yawcmd = 0.0
                settled += 1
            walking_height = self.cfg.get('locomotion_height_m', 0.8)
            height_fraction = 1.0 - min(1.0, lowered / max(1, lower_ticks)) if phase == 'lower' else 0.0 if phase == 'servo' else float(np.clip((settled - brake_ticks) / max(1, ramp_ticks), 0.0, 1.0))
            rt.cfg['control']['height_m'] = walking_height + height_fraction * (0.8 - walking_height)
            with contextlib.redirect_stdout(self.null):
                rt._velocity_tick(request, yawcmd)
            post = self.state()
            post_support = self.support()
            collision |= post['collision']
            fall |= post['fall_detected']
            obstacle |= not self.ground.free(*post['position'][:2])['legal']
            rows.append({'time': float(rt.env.sim.current_time), 'phase': phase, 'height_command_m': rt.cfg['control']['height_m'], 'request_task_velocity': request.tolist(), 'yaw_rate_command': yawcmd, 'position': post['position'], 'yaw': post['yaw'], 'support_xy': post_support['xy'].tolist(), 'support_yaw': post_support['yaw'], 'support_error_m': float(np.linalg.norm(goal - post_support['xy'])), 'velocity_body': post['velocity_body'], 'yaw_rate': post['yaw_rate']})
            if tick % 250 == 0:
                print('SUPPORT_SERVO', action, tick, round(rows[-1]['support_error_m'], 4), round(math.degrees(yawerr), 2), flush=True)
            if collision or fall or obstacle:
                reason = 'unsafe'
                break
            if np.linalg.norm(np.array(post['position'][:2]) - before['position'][:2]) > self.cfg.get('max_primitive_travel_m', 0.45):
                reason = 'primitive_travel_limit'
                break
            if phase == 'lower' and lowered >= lower_ticks:
                phase = 'servo'
            if phase == 'settle' and settled >= brake_ticks + ramp_ticks:
                xe = np.linalg.norm(np.array(post['position'][:2]) - expected['xy'])
                ye = abs(angle_delta(post['yaw'], expected['yaw']))
                near = xe <= min(self.cfg['primitive_xy_tolerance_m'], self.cfg.get('servo_goal_xy_m', 0.015)) and ye <= math.radians(min(self.cfg['primitive_yaw_tolerance_deg'], self.cfg.get('servo_goal_yaw_deg', 0.5))) and (np.linalg.norm(post['velocity_body']) < 0.025) and (abs(post['yaw_rate']) < 0.04)
                goal_stable = goal_stable + 1 if near else 0
                rows[-1]['goal_stable_frames'] = goal_stable
                if goal_stable >= 10:
                    viewcheck = self.verify_target_view(expected)
                    rows[-1]['target_view_check'] = viewcheck
                    if viewcheck['passed']:
                        completed = True
                        break
                    goal_stable = 0
            if phase == 'settle' and settled >= settle_ticks:
                rows[-1]['settling_support_drift_xy'] = (post_support['xy'] - brake_support['xy']).tolist()
                rows[-1]['settling_support_drift_yaw_deg'] = math.degrees(angle_delta(post_support['yaw'], brake_support['yaw']))
                rest_error = np.array(expected['xy']) - post['position'][:2]
                rest_yaw_error = angle_delta(expected['yaw'], post['yaw'])
                gain = np.full(2, self.cfg.get('support_correction_gain', 0.65))
                yaw_gain = self.cfg.get('support_yaw_correction_gain', 0.75)
                if previous_rest_error is not None:
                    gain = np.where(rest_error * previous_rest_error < 0, gain * 0.5, gain)
                if previous_rest_yaw_error is not None and rest_yaw_error * previous_rest_yaw_error < 0:
                    yaw_gain *= 0.5
                rows[-1]['rest_correction_gain_xy'] = gain.tolist()
                rows[-1]['rest_correction_gain_yaw'] = yaw_gain
                previous_rest_error = rest_error.copy()
                previous_rest_yaw_error = rest_yaw_error
                last_rest_xy_error = float(xe)
                last_rest_yaw_error_deg = math.degrees(ye)
                goal = brake_support['xy'] + gain * rest_error
                goal_yaw = brake_support['yaw'] + yaw_gain * rest_yaw_error
                integral[:] = 0
                iyaw = 0.0
                phase = 'lower'
                lowered = 0
                goal_stable = 0
                attempts += 1
                if attempts >= self.cfg.get('support_max_corrections', 6):
                    reason = 'relaxed_support_offset_not_converged'
                    break
        rt.cfg['control']['height_m'] = 0.8
        if not completed:
            for _ in range(math.ceil(self.cfg.get('support_rest_s', 3.0) / rt.dt)):
                with contextlib.redirect_stdout(self.null):
                    rt._velocity_tick((0, 0), 0)
                stopped = self.state()
                collision |= stopped['collision']
                fall |= stopped['fall_detected']
                obstacle |= not self.ground.free(*stopped['position'][:2])['legal']
                rows.append({'time': float(rt.env.sim.current_time), 'phase': 'final_zero_command_settle', 'request_task_velocity': [0.0, 0.0], 'yaw_rate_command': 0.0, 'position': stopped['position'], 'yaw': stopped['yaw']})
        after = self.state()
        xe = float(np.linalg.norm(np.array(after['position'][:2]) - expected['xy']))
        ye = abs(math.degrees(angle_delta(after['yaw'], expected['yaw'])))
        pe = abs(after['tilt_deg'] - expected['pitch'])
        if not completed and (not (collision or fall or obstacle)) and (reason != 'primitive_travel_limit') and (xe <= self.cfg['primitive_xy_tolerance_m']) and (ye <= self.cfg['primitive_yaw_tolerance_deg']) and (pe <= self.cfg['pitch_tolerance_deg']) and (np.linalg.norm(after['velocity_body']) < 0.025) and (abs(after['yaw_rate']) < 0.04):
            check = self.verify_target_view(expected)
            completed = check['passed']
            rows.append({'time': float(rt.env.sim.current_time), 'phase': 'final_zero_command_settle_check', 'request_task_velocity': [0.0, 0.0], 'yaw_rate_command': 0.0, 'target_view_check': check})
            if completed:
                reason = None
        ok = bool(completed and (not (collision or fall or obstacle)) and (xe <= self.cfg['primitive_xy_tolerance_m']) and (ye <= self.cfg['primitive_yaw_tolerance_deg']) and (pe <= self.cfg['pitch_tolerance_deg']))
        return {'action': action, 'success': ok, 'action_completed': completed, 'collision': collision, 'fall': fall, 'scene_envelope_collision': obstacle, 'before': before, 'after': after, 'expected_node': expected, 'xy_error_m': xe, 'yaw_error_deg': ye, 'pitch_error_deg': pe, 'controls': rows, 'duration_s': float(rt.env.sim.current_time) - start, 'root_writes': 0, 'failure_reason': reason or (None if ok else 'support_servo_not_converged'), 'execution_mode': 'measured_support_placement_servo', 'support_start': {**support0, 'xy': support0['xy'].tolist()}, 'support_goal_xy': goal.tolist()}

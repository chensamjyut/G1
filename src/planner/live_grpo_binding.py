"""Bind existing GRPO action endpoints to the already running G1.

No BodyXYRuntime constructor, Executor.initialize, scene creation or root write.
Use the live controller and its shared DDS slot. Gimbal endpoints retain the
original 5-degree semantics, with an explicit 10 deg/s simulation-time ramp
instead of an instantaneous USD jump. This is not a physical joint driver.
"""
from pathlib import Path
import contextlib
import json
import math
import os
import sys
import numpy as np


def _wrapped_angle_delta(a, b):
    """Small absolute yaw difference in radians, including the +/-pi seam."""
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


# These are readback tolerances for a kinematic simulation pose, not motor
# calibration values. A stationary G1 can have sub-millimetre/sub-milliradian
# numerical drift while the zero command is being ticked.
BODY_POSE_DRIFT_TOLERANCE_M = 1e-3
BODY_YAW_DRIFT_TOLERANCE_RAD = 1e-4


def bind_executor(env, stage, controller, provider, navigation, policy_root, output, app):
    root = Path(policy_root)
    
    from gimbal_tvr.body_xy_runtime import BodyXYRuntime, load_config
    from gimbal_tvr.joint_controller import GimbalController
    from tools.gimbal_controller import set_gimbal_angles
    from g1_execution.endpoint import PolicyExecutor
    from g1_execution.shared_relative_executor import SharedRelativeExecutor
    from g1_execution.ground import Ground
    from .action_semantics import GIMBAL_YAW_ACTIONS, deployment_semantic
    from .gimbal_motion import move_angles

    class LiveGimbalController(GimbalController):
        """Retain training limits/semantics; write via the existing G1 helper."""
        def set_targets(self, pan_deg, tilt_deg, action_name='DirectTarget'):
            current = self.read_state()
            target_pan = self._clip(pan_deg, self.pan_limits)
            target_pitch = self._clip(tilt_deg, self.tilt_limits)
            if max(abs(target_pan-current.pan_deg), abs(target_pitch-current.tilt_deg)) > self.max_step_deg+1e-8:
                raise RuntimeError('unsafe_gimbal_command_exceeds_original_per_command_limit')
            def write(pan, pitch):
                count = set_gimbal_angles(self.stage, pan, pitch)
                if count != 1:
                    raise RuntimeError(f'expected_one_existing_gimbal_updated_got_{count}')
            def tick():
                # Same standstill tick used by the original pitch-settle loop.
                # Viewpoint gimbal actions must never inject body yaw here.
                rt._velocity_tick((0.,0.),0.)
                actual = ex.state()
                if actual['collision'] or actual['fall_detected']:
                    raise RuntimeError('unsafe_during_gimbal_motion')
            trace = move_angles((current.pan_deg,current.tilt_deg), (target_pan,target_pitch),
                                dt=rt.dt,write=write,tick=tick,max_delta_deg=self.max_step_deg)
            after = self.read_state()
            if abs(after.pan_deg-target_pan)>1e-6 or abs(after.tilt_deg-target_pitch)>1e-6:
                raise RuntimeError('existing_gimbal_command_readback_mismatch')
            self.last_live_command = {
                'entry': 'tools.gimbal_controller.set_gimbal_angles',
                'action_name': action_name,
                'pan_prim': self.pan_path, 'pan_attribute': 'xformOp:rotateZ:command',
                'pitch_prim': self.tilt_path, 'pitch_attribute': 'xformOp:rotateY:command',
                # Kept for consumers of the previous pitch-only audit schema.
                'attribute': 'xformOp:rotateY:command',
                'pitch_before': current.tilt_deg, 'pitch_command': target_pitch,
                'pitch_after_attribute': after.tilt_deg, 'updated_gimbals': 1,
                'yaw_before': current.pan_deg, 'yaw_command': target_pan,
                'yaw_after_attribute': after.pan_deg,
                'motion_profile': 'linear_10_deg_per_sim_second',
                'intermediate_angles': trace,
                'readback_type': 'existing kinematic Xform command; not hardware encoder'}
            return after

    class LiveRuntime(BodyXYRuntime):
        def __init__(self):
            # Explicit binding only; the parent constructor launches a second Sim.
            self.env, self.stage, self.provider, self.output = env, stage, provider, Path(output)
            self.cfg = load_config(root / 'configs/g1_gimbal_4dof_body_xy.yaml')
            self.cfg['control']['dead_zone_m_s'] = 0.0  # same dedicated runtime()
            self.dt = float(env.sim.get_physics_dt()) * 4
            self.velocity = np.zeros(2)
            self.command = [0., 0., 0., .8]
            self.phase = 'servo'
            self.primitive_start_time = float(env.sim.current_time)
            self.primitive_duration = 0.
            self.heading_hold_active = False
            self.controller = LiveGimbalController(stage, self.cfg['robot'], self.step)
            self.sim_step_count = 0

        def _velocity_tick(self, request, yaw_rate=0):
            # User contract: viewpoint NEVER commands body yaw, including XY correction.
            self.heading_hold_active = False
            self.suppressed_body_yaw_request = float(yaw_rate)
            return super()._velocity_tick(request, yaw_rate=0.0)

        def step(self, frames=1):
            for _ in range(frames):
                if not app.is_running():
                    raise RuntimeError('simulation_closed')
                before = float(env.sim.current_time)
                # Update sensing/ROS feedback without overwriting GRPO's command.
                navigation.step(write_commands=False)
                navigation.run_command_dds.write_run_command(str(self.command))
                controller.step()
                self.sim_step_count += 4
                if provider._error_trace_printed or float(env.sim.current_time) <= before:
                    raise RuntimeError('live_locomotion_did_not_advance_physics')

    rt = LiveRuntime()
    # Calling the legacy ctor would replace render() and attach extra annotators.
    # Supply only fields read by the original state()/act() implementation.
    ex = PolicyExecutor.__new__(PolicyExecutor)
    ex.rt, ex.robot = rt, env.scene['robot']
    ex.cfg = json.loads((root / 'configs/executor.json').read_text())
    ex.null = open(os.devnull, 'w')
    ex.ground = Ground(rt, Path(output))
    ex.control_rows, ex.locked = [], True
    class PlannerSharedRelativeExecutor(SharedRelativeExecutor):
        """Planner-only action boundary; training action IDs stay unchanged.

        The original expert executor's BodyYaw actions are body-placement
        primitives.  That is correct for its historical expert data, but it
        is not the live inspection contract: navigation owns the G1 body yaw
        and the viewpoint policy's horizontal actions own the gimbal pan.
        """

        def _execute_gimbal_yaw(self, action):
            if action not in GIMBAL_YAW_ACTIONS:
                raise ValueError(f"not_a_gimbal_yaw_action:{action}")
            rt = self.executor.rt
            before = self.executor.state()
            pan_before = float(before["pan_deg"])
            body_yaw_before = float(before["yaw"])
            body_xy_before = list(before["position"][:2])
            # Use the existing configured five-degree gimbal command.  The
            # policy token is retained; no training action index is changed.
            step_deg = float(rt.controller.config.get("action_step_deg", 5.0))
            controller_action = "PanLeft" if action == "BodyYawLeft" else "PanRight"
            started = float(rt.env.sim.current_time)
            try:
                rt.controller.increment(controller_action, step_deg)
            except Exception as exc:
                after = self.executor.state()
                return {
                    "action": action,
                    "success": False,
                    "action_completed": False,
                    "collision": bool(after.get("collision")),
                    "fall": bool(after.get("fall_detected")),
                    "scene_envelope_collision": False,
                    "before": before,
                    "after": after,
                    "expected_node": {
                        "xy": body_xy_before,
                        "yaw": body_yaw_before,
                        "pitch": float(before["tilt_deg"]),
                    },
                    "xy_error_m": 0.0,
                    "yaw_error_deg": 0.0,
                    "pitch_error_deg": 0.0,
                    "controls": [],
                    "duration_s": float(rt.env.sim.current_time) - started,
                    "root_writes": 0,
                    "failure_reason": f"gimbal_yaw_command_failed:{type(exc).__name__}:{exc}",
                    "execution_mode": "live_gimbal_yaw_xform",
                    "deployment_semantic": deployment_semantic(action),
                    "policy_action": action,
                    "body_yaw_commanded": False,
                    "body_yaw_changed": False,
                    "gimbal_yaw_commanded": True,
                    "gimbal_yaw_changed": False,
                    "gimbal_pitch_commanded": False,
                }
            after = self.executor.state()
            body_yaw_delta = _wrapped_angle_delta(after["yaw"], body_yaw_before)
            body_xy_delta = float(np.linalg.norm(np.asarray(after["position"][:2]) - np.asarray(body_xy_before)))
            pan_after = float(after["pan_deg"])
            pan_changed = abs(pan_after - pan_before) > 1e-6
            body_unchanged = (body_yaw_delta <= BODY_YAW_DRIFT_TOLERANCE_RAD
                              and body_xy_delta <= BODY_POSE_DRIFT_TOLERANCE_M)
            safety_ok = not bool(after.get("collision")) and not bool(after.get("fall_detected"))
            success = bool(pan_changed and body_unchanged and safety_ok)
            reason = None
            if not pan_changed:
                reason = "gimbal_yaw_readback_unchanged"
            elif not body_unchanged:
                reason = "body_pose_changed_during_gimbal_yaw"
            elif not safety_ok:
                reason = "unsafe_during_gimbal_yaw"
            command = getattr(rt.controller, "last_live_command", {}).copy()
            command.update({
                "policy_action": action,
                "deployment_semantic": deployment_semantic(action),
                "controller_action": controller_action,
                "pan_before": pan_before,
                "pan_after": pan_after,
                "pitch_before": float(before["tilt_deg"]),
                "pitch_after": float(after["tilt_deg"]),
                "body_yaw_before": body_yaw_before,
                "body_yaw_after": float(after["yaw"]),
                "body_yaw_delta_rad": body_yaw_delta,
                "body_xy_delta_m": body_xy_delta,
                "gimbal_yaw_changed": pan_changed,
                "gimbal_pitch_changed": abs(float(after["tilt_deg"]) - float(before["tilt_deg"])) > 1e-6,
                "body_yaw_changed": body_yaw_delta > BODY_YAW_DRIFT_TOLERANCE_RAD,
                "body_pose_drift_tolerance_m": BODY_POSE_DRIFT_TOLERANCE_M,
                "body_yaw_drift_tolerance_rad": BODY_YAW_DRIFT_TOLERANCE_RAD,
            })
            return {
                "action": action,
                "success": success,
                "action_completed": success,
                "collision": bool(after.get("collision")),
                "fall": bool(after.get("fall_detected")),
                "scene_envelope_collision": False,
                "before": before,
                "after": after,
                "expected_node": {
                    "xy": body_xy_before,
                    "yaw": body_yaw_before,
                    "pitch": float(before["tilt_deg"]),
                },
                "xy_error_m": body_xy_delta,
                "yaw_error_deg": float(np.degrees(body_yaw_delta)),
                "pitch_error_deg": abs(float(after["tilt_deg"]) - float(before["tilt_deg"])),
                "controls": command.get("intermediate_angles", []),
                "duration_s": float(rt.env.sim.current_time) - started,
                "root_writes": 0,
                "failure_reason": reason,
                "execution_mode": "live_gimbal_yaw_xform",
                "gimbal_readback_source": "realized local Xform orientation; kinematic simulated actuator, not physical encoder",
                "deployment_semantic": deployment_semantic(action),
                "policy_action": action,
                "body_yaw_commanded": False,
                "body_yaw_changed": body_yaw_delta > BODY_YAW_DRIFT_TOLERANCE_RAD,
                "gimbal_yaw_commanded": True,
                "gimbal_yaw_changed": pan_changed,
                "gimbal_pitch_commanded": False,
                "gimbal": command,
            }

        def execute_action(self, action, *, caller):
            if action in GIMBAL_YAW_ACTIONS:
                if caller not in ("expert", "policy"):
                    raise ValueError("invalid_action_caller")
                return self._execute_gimbal_yaw(action)
            result = super().execute_action(action, caller=caller)
            result.setdefault("policy_action", action)
            result.setdefault("deployment_semantic", deployment_semantic(action))
            result.setdefault("body_yaw_commanded", False)
            result.setdefault("body_yaw_changed", _wrapped_angle_delta(result["after"]["yaw"], result["before"]["yaw"]) > 1e-6)
            result.setdefault("gimbal_yaw_commanded", False)
            result.setdefault("gimbal_yaw_changed", False)
            result.setdefault("gimbal_pitch_commanded", action in ("GimbalPitchUp", "GimbalPitchDown"))
            result.setdefault("gimbal_pitch_changed", abs(float(result["after"].get("tilt_deg", 0.0)) - float(result["before"].get("tilt_deg", 0.0))) > 1e-6)
            return result

        def act(self, action, expected):
            # Preserve the old method signature for callers that still use it,
            # while ensuring planner yaw tokens never reach body yaw servoing.
            if action in GIMBAL_YAW_ACTIONS:
                return self.execute_action(action, caller="policy")
            return super().act(action, expected)

    shared = PlannerSharedRelativeExecutor(ex)
    shared.state()  # actual sensors, contacts and authored command attrs required
    return shared

"""Safe controller for the authored kinematic pan/tilt Xforms.

The inspected payload is not a pair of PhysicsRevoluteJoint prims and has no
Articulation DOFs.  Its command/readback state is authored directly in degree
valued rotate ops.  This controller deliberately refuses to operate if those
real attributes cannot be read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .action_space import PAN_LEFT, PAN_RIGHT, STOP, TILT_DOWN, TILT_UP, validate_action


@dataclass(frozen=True)
class JointState:
    pan_deg: float
    tilt_deg: float


class GimbalController:
    def __init__(self, stage, config: dict, step_callback):
        self.stage = stage
        self.config = config
        self.step_callback = step_callback
        self.pan_path = config["pan_joint_prim_path"]
        self.tilt_path = config["tilt_joint_prim_path"]
        self.pan_limits = (float(config["pan_min_deg"]), float(config["pan_max_deg"]))
        self.tilt_limits = (float(config["tilt_min_deg"]), float(config["tilt_max_deg"]))
        self.max_step_deg = min(5.0, float(config.get("max_command_step_deg", 5.0)))
        self._pan_attr = self._required_attr(self.pan_path, "xformOp:rotateZ:command")
        self._tilt_attr = self._required_attr(self.tilt_path, "xformOp:rotateY:command")
        self.read_state()  # hard safety gate before any command

    def _required_attr(self, path: str, name: str):
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"required gimbal prim does not exist: {path}")
        attr = prim.GetAttribute(name)
        if not attr or not attr.IsValid() or attr.Get() is None:
            raise RuntimeError(f"required readable command attribute is absent: {path}.{name}")
        return attr

    def read_state(self) -> JointState:
        pan, tilt = self._pan_attr.Get(), self._tilt_attr.Get()
        if pan is None or tilt is None or not math.isfinite(float(pan)) or not math.isfinite(float(tilt)):
            raise RuntimeError(f"invalid gimbal readback: pan={pan!r}, tilt={tilt!r}")
        return JointState(float(pan), float(tilt))

    @staticmethod
    def _clip(value: float, limits: tuple[float, float]) -> float:
        return max(limits[0], min(limits[1], float(value)))

    def set_targets(self, pan_deg: float, tilt_deg: float, action_name: str = "DirectTarget") -> JointState:
        current = self.read_state()
        target_pan = self._clip(pan_deg, self.pan_limits)
        target_tilt = self._clip(tilt_deg, self.tilt_limits)
        pan_delta, tilt_delta = target_pan - current.pan_deg, target_tilt - current.tilt_deg
        if abs(pan_delta) > self.max_step_deg + 1e-8 or abs(tilt_delta) > self.max_step_deg + 1e-8:
            raise RuntimeError(
                f"unsafe command rejected: delta=({pan_delta:.3f}, {tilt_delta:.3f}) deg "
                f"exceeds per-command limit {self.max_step_deg:.3f} deg"
            )
        if self.config.get("verbose_commands", True):
            print(
                f"[gimbal] action={action_name} current=({current.pan_deg:.3f},{current.tilt_deg:.3f})deg "
                f"request_delta=({pan_delta:.3f},{tilt_delta:.3f})deg "
                f"clipped_target=({target_pan:.3f},{target_tilt:.3f})deg "
                f"limits=pan{self.pan_limits},tilt{self.tilt_limits}"
            )
        self._pan_attr.Set(target_pan)
        self._tilt_attr.Set(target_tilt)
        return JointState(target_pan, target_tilt)

    def increment(self, action: str, step_deg: float | None = None) -> JointState:
        validate_action(action)
        current = self.read_state()
        if action == STOP:
            return self.set_targets(current.pan_deg, current.tilt_deg, STOP)
        step = float(self.config.get("action_step_deg", 5.0) if step_deg is None else step_deg)
        if not 0.0 < step <= self.max_step_deg:
            raise ValueError(f"step must be in (0, {self.max_step_deg}] deg, got {step}")
        pan_positive_left = self.config.get("pan_positive_visual_direction") == "left"
        tilt_positive_up = self.config.get("tilt_positive_visual_direction") == "up"
        dp = dt = 0.0
        if action == PAN_LEFT:
            dp = step if pan_positive_left else -step
        elif action == PAN_RIGHT:
            dp = -step if pan_positive_left else step
        elif action == TILT_UP:
            dt = step if tilt_positive_up else -step
        elif action == TILT_DOWN:
            dt = -step if tilt_positive_up else step
        return self.set_targets(current.pan_deg + dp, current.tilt_deg + dt, action)

    def wait_until_stable(self, max_frames: int = 120) -> JointState:
        threshold = float(self.config.get("settle_velocity_threshold", 0.05))
        required = int(self.config.get("settle_frames", 3))
        stable = 0
        previous = self.read_state()
        for _ in range(max_frames):
            self.step_callback(1)
            current = self.read_state()
            speed = max(abs(current.pan_deg - previous.pan_deg), abs(current.tilt_deg - previous.tilt_deg))
            if speed <= threshold:
                stable += 1
                if stable >= required:
                    return current
            else:
                stable = 0
            previous = current
        raise TimeoutError(f"gimbal did not settle within {max_frames} frames")

    def restore(self, initial: JointState) -> JointState:
        current = self.read_state()
        # Restore in bounded increments, never with one large command.
        while abs(current.pan_deg - initial.pan_deg) > 1e-6 or abs(current.tilt_deg - initial.tilt_deg) > 1e-6:
            pan = current.pan_deg + max(-self.max_step_deg, min(self.max_step_deg, initial.pan_deg - current.pan_deg))
            tilt = current.tilt_deg + max(-self.max_step_deg, min(self.max_step_deg, initial.tilt_deg - current.tilt_deg))
            self.set_targets(pan, tilt, "Restore")
            current = self.wait_until_stable()
        return current

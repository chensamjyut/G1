"""Helpers for the kinematic g1_comp inspection gimbal."""

from __future__ import annotations


# The XL330 pan stage used by g1_comp can rotate through a full turn in the
# simulation.  A wide pan range is required when the whole-body locomotion
# policy cannot safely perform a large in-place turn in front of a shelf.
YAW_LIMIT_DEG = (-180.0, 180.0)
PITCH_LIMIT_DEG = (-20.0, 90.0)


def _clamp(value: float, limits: tuple[float, float]) -> float:
    return max(limits[0], min(limits[1], float(value)))


def set_gimbal_angles(stage, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> int:
    """Set every inspection gimbal on a USD stage and return the count."""
    yaw_deg = _clamp(yaw_deg, YAW_LIMIT_DEG)
    pitch_deg = _clamp(pitch_deg, PITCH_LIMIT_DEG)
    updated = 0

    for prim in stage.Traverse():
        if prim.GetName() != "gimbal_yaw_joint":
            continue
        pitch_prim = prim.GetChild("gimbal_pitch_joint")
        yaw_attr = prim.GetAttribute("xformOp:rotateZ:command")
        pitch_attr = pitch_prim.GetAttribute("xformOp:rotateY:command")
        if yaw_attr and pitch_attr:
            yaw_attr.Set(yaw_deg)
            pitch_attr.Set(pitch_deg)
            updated += 1

    return updated

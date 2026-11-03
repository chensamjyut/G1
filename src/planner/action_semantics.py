"""Deployment semantics for the unchanged nine-action policy vocabulary.

The checkpoint vocabulary is historical: ``BodyYawLeft`` and
``BodyYawRight`` are the two horizontal viewpoint actions.  In the live
inspection deployment they address the authored inspection gimbal pan axis;
the G1 body yaw remains owned by navigation.  Keeping this mapping in a
small, dependency-free module makes the boundary explicit and testable.
"""

from __future__ import annotations


POLICY_TO_DEPLOYMENT = {
    "MoveXPos": "BodyMoveXPos",
    "MoveXNeg": "BodyMoveXNeg",
    "MoveYPos": "BodyMoveYPos",
    "MoveYNeg": "BodyMoveYNeg",
    "BodyYawLeft": "GimbalYawLeft",
    "BodyYawRight": "GimbalYawRight",
    "GimbalPitchUp": "GimbalPitchUp",
    "GimbalPitchDown": "GimbalPitchDown",
    "Stop": "Stop",
}

GIMBAL_YAW_ACTIONS = frozenset(("BodyYawLeft", "BodyYawRight"))
GIMBAL_PITCH_ACTIONS = frozenset(("GimbalPitchUp", "GimbalPitchDown"))


def deployment_semantic(policy_action: str) -> str:
    """Return the physical deployment meaning of an unchanged policy token."""

    try:
        return POLICY_TO_DEPLOYMENT[policy_action]
    except KeyError as exc:
        raise ValueError(f"unknown_policy_action:{policy_action!r}") from exc


def is_gimbal_action(policy_action: str) -> bool:
    return policy_action in GIMBAL_YAW_ACTIONS or policy_action in GIMBAL_PITCH_ACTIONS


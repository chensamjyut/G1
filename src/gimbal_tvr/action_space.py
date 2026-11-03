"""Discrete action vocabulary shared with the future visual-action agent."""

PAN_LEFT = "PanLeft"
PAN_RIGHT = "PanRight"
TILT_UP = "TiltUp"
TILT_DOWN = "TiltDown"
STOP = "Stop"
ACTIONS = (PAN_LEFT, PAN_RIGHT, TILT_UP, TILT_DOWN, STOP)


def validate_action(action: str) -> str:
    valid = ACTIONS + ("MoveXPos", "MoveXNeg", "MoveYPos", "MoveYNeg")
    if action not in valid:
        raise ValueError(f"invalid action {action!r}; expected one of {valid}")
    return action

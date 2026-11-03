"""The nine labels stored in the retained VA-SFT checkpoint."""
ACTIONS = (
    "MoveXPos", "MoveXNeg", "MoveYPos", "MoveYNeg",
    "BodyYawLeft", "BodyYawRight", "GimbalPitchUp", "GimbalPitchDown", "Stop",
)
STOP = "Stop"
PAD_ID = len(ACTIONS)
HISTORY_LENGTH = 8

def encode_history(names):
    names = list(names)[-HISTORY_LENGTH:]
    ids = []
    for name in names:
        if name not in ACTIONS:
            raise ValueError(f"unknown action in history: {name!r}")
        ids.append(ACTIONS.index(name))
    return ids + [PAD_ID] * (HISTORY_LENGTH - len(ids))

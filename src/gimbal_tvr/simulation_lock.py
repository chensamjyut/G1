"""OS-backed exclusion for the single authorized Isaac environment."""
import fcntl
import os
from pathlib import Path


def acquire_simulation_lock(path=None):
    path = Path(path) if path is not None else Path('/tmp') / f'gimbal_tvr_isaac_{os.getuid()}.lock'
    handle = path.open('a+')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip()
        handle.close()
        raise RuntimeError(f'SIMULATION_ALREADY_RUNNING: {owner}') from None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle  # Keep open for the full app lifetime; OS also releases on exit.

"""Read the existing Isaac Lab inspection_camera; no sensor creation or stepping."""
from __future__ import annotations

import math


class FreshFramePending(RuntimeError):
    pass


class CameraAdapter:
    sensor_name = "inspection_camera"

    def __init__(self, env):
        self.env = env
        self.sensor = env.scene[self.sensor_name]
        self._previous_id = None

    def get_latest_rgb(self, after_frame_id=None):
        # Camera.data updates its annotator buffer lazily. Camera.frame is the
        # real Isaac Lab buffer counter, not a counter invented by this adapter.
        rgb = self.sensor.data.output["rgb"][0]
        frame_id = int(self.sensor.frame[0].item())
        timestamp = float(self.sensor._timestamp_last_update[0].item())
        paths = list(self.sensor._view.prim_paths)
        if len(paths) != 1:
            raise RuntimeError("expected_exactly_one_inspection_camera")
        if self._previous_id is not None and frame_id < self._previous_id:
            raise RuntimeError("camera_episode_reset_requires_adapter_reset")
        if after_frame_id is not None and frame_id <= after_frame_id:
            raise FreshFramePending("no_post_action_camera_frame_yet")
        array = rgb.detach().cpu().numpy().copy()
        if array.ndim != 3 or array.shape[2] != 3 or str(array.dtype) != "uint8":
            raise RuntimeError(f"unexpected_rgb_contract:{array.shape}:{array.dtype}")
        if not math.isfinite(timestamp) or frame_id < 1:
            raise RuntimeError("invalid_sensor_frame_metadata")
        self._previous_id = frame_id
        return dict(rgb=array, frame_id=frame_id, timestamp=timestamp,
                    timestamp_source="IsaacLab.SensorBase._timestamp_last_update",
                    frame_id_source="IsaacLab.Camera.frame", shape=list(array.shape),
                    dtype=str(array.dtype), sensor_name=self.sensor_name, prim_path=paths[0])

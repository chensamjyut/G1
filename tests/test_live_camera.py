"""Contract guards only; these tests do not certify a simulator or a fresh sensor."""
import unittest
from types import SimpleNamespace

import numpy as np

from planner.live_camera import CameraAdapter, FreshFramePending


class Array:
    def __init__(self, value): self.value = np.asarray(value)
    def __getitem__(self, index): return Array(self.value[index])
    def item(self): return self.value.item()
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


class CameraGuards(unittest.TestCase):
    def setUp(self):
        self.sensor = SimpleNamespace(
            data=SimpleNamespace(output={'rgb': Array(np.zeros((1, 480, 640, 3), dtype=np.uint8))}),
            frame=Array([4]), _timestamp_last_update=Array([0.2]),
            _view=SimpleNamespace(prim_paths=['/World/envs/env_0/Robot/inspection_camera']))
        self.adapter = CameraAdapter(SimpleNamespace(scene={'inspection_camera': self.sensor}))

    def test_old_frame_is_rejected_without_incrementing_counter(self):
        with self.assertRaises(FreshFramePending): self.adapter.get_latest_rgb(4)
        self.assertEqual(self.sensor.frame[0].item(), 4)

    def test_new_buffer_is_copied_and_source_metadata_retained(self):
        frame = self.adapter.get_latest_rgb(3)
        self.assertEqual(frame['frame_id'], 4)
        self.assertEqual(frame['timestamp'], .2)
        frame['rgb'][0, 0, 0] = 255
        self.assertEqual(self.sensor.data.output['rgb'].value[0, 0, 0, 0], 0)

    def test_episode_counter_reset_requires_explicit_reinitialization(self):
        self.adapter.get_latest_rgb()
        self.sensor.frame = Array([1])
        with self.assertRaisesRegex(RuntimeError, 'episode_reset'):
            self.adapter.get_latest_rgb()

    def test_wrong_image_contract_is_rejected(self):
        self.sensor.data.output['rgb'] = Array(np.zeros((1, 10, 10), dtype=np.uint8))
        with self.assertRaisesRegex(RuntimeError, 'unexpected_rgb_contract'):
            self.adapter.get_latest_rgb()

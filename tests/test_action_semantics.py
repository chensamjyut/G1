import unittest

from planner.action_semantics import (
    GIMBAL_PITCH_ACTIONS,
    GIMBAL_YAW_ACTIONS,
    deployment_semantic,
)


class ActionSemantics(unittest.TestCase):
    def test_policy_yaw_tokens_are_gimbal_yaw(self):
        self.assertEqual(deployment_semantic("BodyYawLeft"), "GimbalYawLeft")
        self.assertEqual(deployment_semantic("BodyYawRight"), "GimbalYawRight")
        self.assertEqual(GIMBAL_YAW_ACTIONS, {"BodyYawLeft", "BodyYawRight"})

    def test_pitch_tokens_remain_gimbal_pitch(self):
        self.assertEqual(deployment_semantic("GimbalPitchUp"), "GimbalPitchUp")
        self.assertEqual(deployment_semantic("GimbalPitchDown"), "GimbalPitchDown")
        self.assertEqual(GIMBAL_PITCH_ACTIONS, {"GimbalPitchUp", "GimbalPitchDown"})

    def test_body_motion_and_stop_are_not_gimbal_yaw(self):
        self.assertEqual(deployment_semantic("MoveXPos"), "BodyMoveXPos")
        self.assertEqual(deployment_semantic("MoveYNeg"), "BodyMoveYNeg")
        self.assertEqual(deployment_semantic("Stop"), "Stop")

    def test_unknown_action_is_rejected(self):
        with self.assertRaises(ValueError):
            deployment_semantic("RotateLeft")


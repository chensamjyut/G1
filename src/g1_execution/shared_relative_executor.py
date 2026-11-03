"""Versioned compatibility layer; historical absolute-node experts stay immutable.

Both new expert collection and policy evaluation call execute_action(). A role
is an audit tag only: it cannot supply a goal, planner node, or endpoint override.
The unchanged physical controller reads state for low-level relative servoing.
"""
import math
import copy
import json
from pathlib import Path
import numpy as np
from .endpoint import endpoint
from g1_active_view.actions import ACTIONS


class SharedRelativeExecutor:
    def __init__(self, executor):
        # PolicyExecutor retains the legacy physical servo but disables the
        # privileged target-view correction for BOTH new callers.
        from .endpoint import PolicyExecutor
        if type(executor) is not PolicyExecutor:
            raise TypeError('Requires audited PolicyExecutor with no goal-view correction')
        self.executor = executor
        self.cfg = copy.deepcopy(executor.cfg)
        profile = json.loads((Path(executor.rt.cfg['_config_path']).parent/'execution_profile_packaged.json').read_text())['internal_execution_profile']
        for key, value in profile.items():
            if key not in ('primitive_xy_tolerance_m','primitive_yaw_tolerance_deg','servo_goal_xy_m','servo_goal_yaw_deg'):
                raise ValueError('unaudited execution override: '+key)
            if value > self.cfg[key]:
                raise ValueError('execution tolerance relaxation forbidden')
            self.cfg[key] = value
        gains = json.loads((Path(executor.rt.cfg['_config_path']).parent/'execution_profile_packaged.json').read_text()).get('internal_controller_overrides', {})
        for key,value in gains.items():
            if key not in ('support_correction_gain','support_yaw_correction_gain') or not .1 <= value <= 1.:
                raise ValueError('unsupported residual-gain pilot')
            self.cfg[key] = value
        self.executor.cfg = self.cfg

    def __getattr__(self, name):
        return getattr(self.executor, name)

    def act(self, action, expected):
        # Compatibility for unchanged evaluators. Reject external goal-directed
        # endpoints; only the identical measured-start relative target is legal.
        required = endpoint(action, self.executor.state(), self.cfg)
        if not (np.allclose(expected['xy'],required['xy'],rtol=0,atol=1e-9)
                and abs(expected['yaw']-required['yaw']) < 1e-9
                and abs(expected['pitch']-required['pitch']) < 1e-9):
            raise ValueError('absolute_or_goal_endpoint_override_forbidden')
        return self.execute_action(action, caller='policy')

    def execute_action(self, action, *, caller):
        if action not in ACTIONS or caller not in ('expert', 'policy'):
            raise ValueError('invalid action/caller')
        before = self.executor.state()
        expected = endpoint(action, before, self.cfg)
        if not -20 <= expected['pitch'] <= 90:
            raise ValueError('gimbal_limit_before_execution')
        result = self.executor.act(action, expected)
        result['shared_semantics'] = {
            'version': 'relative_measured_start_user_approved_baseline_v1',
            'caller_audit_only': caller,
            'action_id': ACTIONS.index(action),
            'endpoint_function': 'dedicated_sim_validation.endpoint',
            'executor': type(self.executor).__name__ + '.act -> FootReferencedExpertExecutor.act',
            'goal_feedback': False,
            'max_servo_passes': 1,
            'internal_xy_tolerance_m': self.cfg['primitive_xy_tolerance_m'],
            'internal_yaw_tolerance_deg': self.cfg['primitive_yaw_tolerance_deg'],
            'relative_target_computed_at': 'immediately before physical act',
        }
        result['servo_passes'] = [{k:result.get(k) for k in ('success','duration_s','failure_reason','xy_error_m','yaw_error_deg')}]
        after = result['after']
        result['settled'] = bool(np.linalg.norm(after['velocity_body']) < .025 and abs(after['yaw_rate']) < .04)
        # The old executor has no explicit loop-exhausted flag. Preserve the
        # distinction between duration evidence and an unknown timeout cause.
        cap = 120 * self.rt.dt if action.startswith('GimbalPitch') else self.cfg['support_max_seconds']
        result['timeout'] = bool(not result['action_completed'] and result['duration_s'] >= cap - self.rt.dt)
        result['timeout_evidence'] = 'duration_at_or_above_legacy_servo_cap; failure_reason retained'
        result['return_code'] = 0 if result['success'] else 1
        return result

"""One in-process inspection manager attached to sim_main's existing objects.

All transitions are evidence based. Initialization loads but does not predict.
A Stop proposal is not accepted if its actual executor reports failure.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
import copy
from .live_camera import CameraAdapter, FreshFramePending
from .number_color_reader import DoubaoNumberColorReader

def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

class LiveSinglePoint:
    MAX_VIEW_STEPS = 50
    READ_EVERY_VIEW_STEPS = 5

    def __init__(self, *, env, stage, controller, provider, navigation, config_path, ready_path, trigger_path, app):
        self.config = json.loads(Path(config_path).read_text())
        (self.env, self.stage, self.controller, self.navigation, self.app) = (env, stage, controller, navigation, app)
        (self.ready_path, self.trigger_path) = (Path(ready_path), Path(trigger_path))
        self.nonce = uuid.uuid4().hex
        self.output = Path(self.config['output_root']) / ('single_demo_' + time.strftime('%Y%m%d_%H%M%S') + '_' + self.nonce[:6])
        self.output.mkdir(parents=True)
        self.state = 'INITIALIZING'
        (self.history, self.actions, self.readings) = ([], [], [])
        self.action_history = []
        (self.retry_count, self.action_count) = (0, 0)
        (self.view_step_count, self.best_view) = (0, None)
        (self.view_observations, self.display) = ([], None)
        (self.worker, self.after_frame, self.frame) = (None, None, None)
        self.created = time.monotonic()
        self.view_started = None
        self.last_nav_trace = 0.0
        self.editor_view_switched = False
        self.navigation.stop_navigation()
        print('[CAMERA] validating existing inspection_camera buffer', flush=True)
        self.camera = CameraAdapter(env)
        first = self.camera.get_latest_rgb()
        self.camera_path = first['prim_path']
        from pxr import UsdPhysics
        roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI) and '/Robot/' in str(p.GetPath())]
        if len(roots) != 1:
            raise RuntimeError(f'expected_one_G1_articulation_got_{len(roots)}')
        self.robot_path = str(env.scene['robot'].cfg.prim_path).replace('env_.*', 'env_0')
        if not self.camera_path.startswith('/World/envs/env_0/Robot/'):
            raise RuntimeError('inspection_camera_not_on_live_G1')
        policy_cfg = self.config['viewpoint_policy']
        self.checkpoint = Path(policy_cfg['checkpoint'])
        if hashlib.sha256(self.checkpoint.read_bytes()).hexdigest() != policy_cfg['checkpoint_sha256']:
            raise RuntimeError('GRPO_checkpoint_SHA256_mismatch')
        self.target = Path(self.config['inspection_points'][0]['target_image_path'])
        if hashlib.sha256(self.target.read_bytes()).hexdigest() != self.config['target_metadata']['sha256']:
            raise RuntimeError('target_image_SHA256_mismatch')
        self.policy_root = Path(self.config['execution_root'])
        print('[STATE] LOADING_VIEWPOINT_POLICY', flush=True)
        from g1_active_view.policies import OfflinePolicy
        from g1_active_view.actions import ACTIONS
        from .action_semantics import POLICY_TO_DEPLOYMENT
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(42)
        self.policy = OfflinePolicy(str(self.checkpoint), policy_cfg['dino_source'], policy_cfg['dino_weights'], device=policy_cfg.get('device', 'cuda'))
        weights = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        self.policy.policy.load_state_dict(weights['policy'], strict=True)
        if len(ACTIONS) != 9 or ACTIONS[8] != 'Stop':
            raise RuntimeError('nine_action_contract_mismatch')
        self.action_names = list(ACTIONS)
        self.action_semantics = {name: POLICY_TO_DEPLOYMENT[name] for name in self.action_names}
        print(f'[VIEWPOINT] loader=OfflinePolicy\n[VIEWPOINT] checkpoint={self.checkpoint}\n[VIEWPOINT] action_dim=9\n[VIEWPOINT] STOP_ACTION=8', flush=True)
        from .live_grpo_binding import bind_executor
        self.executor = bind_executor(env, stage, controller, provider, navigation, self.policy_root, self.output, app)
        self._locate_goal()
        print('[STATE] INITIALIZING_VLM_READER', flush=True)
        self.model = self.config['reader']['model']
        self.reader = DoubaoNumberColorReader(self.model)
        self.vlm_credential_present = bool(os.getenv('ARK_API_KEY'))
        print(f'[VLM] reader initialized; credential_present={self.vlm_credential_present}; API not called', flush=True)
        self.reader_python = Path(self.config['reader']['python'])
        if not self.reader_python.is_file():
            raise RuntimeError('isolated_Ark_runtime_missing')
        self.record = {'run_id': self.nonce, 'scene': self.config['scene_path'], 'robot_prim': self.robot_path, 'g1_articulation_roots': roots, 'g1_count': len(roots), 'camera_prim': self.camera_path, 'checkpoint': str(self.checkpoint), 'checkpoint_sha256': policy_cfg['checkpoint_sha256'], 'loader': 'OfflinePolicy', 'policy_action_semantics': self.action_semantics, 'body_yaw_owner': 'navigation_only', 'target_image': str(self.target), 'target_sha256': self.config['target_metadata']['sha256'], 'initial_pose': self.executor.state(), 'goal_geometry': self.goal_geometry, 'actions': self.actions, 'readings': self.readings, 'states': self.history, 'stop_source': None, 'training': False}
        self.record.update(view_steps=0, max_view_steps=self.MAX_VIEW_STEPS, read_every_view_steps=self.READ_EVERY_VIEW_STEPS, view_observations=self.view_observations, similarity_metric='cosine_similarity(DINOv2(current), DINOv2(target)), clamped [-1,1]', better_direction='higher', termination_reason=None)
        self.record['configured_limits'] = {k: self.config.get(k) for k in ('max_view_actions', 'live_navigation_timeout_s', 'live_view_timeout_s')}
        self.record['vlm_credential_present'] = self.vlm_credential_present
        save_json(self.output / 'initialization.json', self.record)
        self._ready_file('INITIALIZING_NAVIGATION', ready=False)

    def _locate_goal(self):
        from pxr import UsdGeom, Gf
        import numpy as np
        suffix = self.config['target_geometry']['label_suffix']
        labels = [p for p in self.stage.Traverse() if str(p.GetPath()).endswith(suffix)]
        if len(labels) != 1:
            raise RuntimeError('cabinet10_relocated_label_not_unique')
        prim = labels[0]
        mesh = UsdGeom.Mesh(prim)
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        points = np.array([list(matrix.Transform(Gf.Vec3d(*map(float, p)))) for p in mesh.GetPointsAttr().Get()])
        center = points.mean(0)
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        normal /= np.linalg.norm(normal)
        inward = np.array(self.config['target_geometry']['factory_center_xy']) - center[:2]
        if np.dot(normal[:2], inward) < 0:
            normal = -normal
        normal[:2] /= np.linalg.norm(normal[:2])
        candidates = []
        for distance in (1.2, 1.5, 1.8, 2.1):
            xy = center[:2] + distance * normal[:2]
            check = self.executor.ground.free(*xy)
            candidates.append({'world_xy': xy.tolist(), 'distance_m': distance, 'ground': check})
        valid = next((c for c in candidates if c['ground']['legal']), None)
        self.goal_geometry = {'label_prim': str(prim.GetPath()), 'center_world': center.tolist(), 'normal_world': normal.tolist(), 'candidates': candidates, 'selection': valid, 'basis': 'existing authored label geometry; PhysX envelope/floor validation'}
        if valid is None:
            raise RuntimeError('no_safe_cabinet10_approach_point')
        xy = valid['world_xy']
        heading_offset = math.radians(float(self.config.get('navigation_heading_offset_deg', 0.0)))
        self.world_goal = [*xy, math.atan2(center[1] - xy[1], center[0] - xy[0]) + heading_offset]
        self.goal_geometry['navigation_heading_offset_deg'] = math.degrees(heading_offset)
        print('[NAV] safe world approach = ' + str(self.world_goal), flush=True)

    def _ready_file(self, status, **extra):
        save_json(self.ready_path, {'status': status, 'ready': status == 'WAIT_FOR_START', 'pid': os.getpid(), 'process_start_ticks': Path(f'/proc/{os.getpid()}/stat').read_text().split()[21], 'nonce': self.nonce, 'trigger_file': str(self.trigger_path), 'output': str(self.output), 'checkpoint_loaded': True, 'camera': self.camera_path, 'updated_at': time.time(), 'result_status': self.record.get('status'), 'failure_reason': self.record.get('failure_reason'), 'missing_navigation_signals': self.record.get('missing_navigation_signals', []), **extra})

    def transition(self, state):
        self.state = state
        self.history.append({'state': state, 'wall_time': time.time(), 'sim_time': float(self.env.sim.current_time)})
        print('[STATE] ' + state, flush=True)
        self._ready_file(state)
        save_json(self.output / 'progress.json', self.record)

    def _capture(self):
        frame = self.camera.get_latest_rgb(after_frame_id=self.after_frame)
        from PIL import Image
        path = self.output / f"rgb_{frame['frame_id']:09d}_{len(self.actions):03d}_{len(self.readings):02d}.png"
        Image.fromarray(frame.pop('rgb')).save(path)
        frame['path'] = str(path)
        frame['captured_sim_time'] = float(self.env.sim.current_time)
        frame['clock_note'] = 'sensor timestamp and physics time are distinct clocks; compare each only to itself'
        frame['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.resolve() == self.target.resolve():
            raise RuntimeError('target_cannot_be_current_RGB')
        self.frame = frame
        self.after_frame = frame['frame_id']
        print(f"[CAMERA] frame_id={frame['frame_id']} timestamp={frame['timestamp']:.4f}", flush=True)
        return frame

    def _finish(self, status, reason=None):
        self.navigation.stop_navigation()
        if self.readings:
            self.record['last_reading_rgb'] = self.readings[-1]['frame']['path']
        if self.readings and (status == 'SUCCESS' or self.state == 'READING'):
            self.frame = self.readings[-1]['frame']
            self.record['reading_image_source'] = self.readings[-1].get('image_source', 'current_rgb_after_policy_stop')
            self.record['final_rgb_source'] = self.record['reading_image_source']
        elif status != 'SUCCESS':
            try:
                self._capture()
                self.record['final_rgb_source'] = 'latest_failure_frame; not a policy Stop'
            except FreshFramePending:
                self.record['final_rgb_source'] = 'last_available_observation; no new frame'
            except Exception as exc:
                self.record['final_rgb_source'] = 'last_available_observation; capture failed:' + type(exc).__name__
        else:
            self.record['final_rgb_source'] = 'exact post-action current RGB submitted to the reader'
        self.record['view_steps'] = getattr(self, 'view_step_count', 0)
        self.record.update(status=status, failure_reason=reason, reading=self.readings[-1]['result'] if self.readings else self.reader._failure('not_attempted'), final_pose=self.executor.state())
        self.transition('SAVING')
        if self.frame:
            shutil.copyfile(self.frame['path'], self.output / 'final_rgb.png')
            self.record['final_rgb'] = str(self.output / 'final_rgb.png')
        self.transition('DONE')
        save_json(self.output / 'result.json', self.record)
        print(f"[RESULT] status={status}\n[RESULT] json={self.output / 'result.json'}\n[RESULT] rgb={self.record.get('final_rgb')}\n[STATE] DONE", flush=True)
        return True

    def _consider_view(self, frame, similarity, step, phase, state):
        if not math.isfinite(float(similarity)):
            raise RuntimeError('non_finite_policy_similarity')
        row = {'step': step, 'phase': phase, 'similarity': float(similarity), 'state': copy.deepcopy(state), 'frame': copy.deepcopy(frame)}
        self.view_observations.append(row)
        if self.best_view is None or row['similarity'] > self.best_view['similarity']:
            self.best_view = copy.deepcopy(row)
            self.record.update(best_step=step, best_similarity=row['similarity'], best_state=row['state'], best_rgb=frame['path'], best_phase=phase)
            shutil.copyfile(frame['path'], self.output / 'best_rgb.png')
            save_json(self.output / 'best_view.json', self.best_view)
        with (self.output / 'view_trace.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(f"[VIEW] best_step={self.best_view['step']:02d} best_similarity={self.best_view['similarity']:.6f}", flush=True)

    def _terminal_similarity(self, frame):
        """Same encoder/preprocessing/formula as OfflinePolicy.predict.

        Evaluate the last reached frame only; no policy forward or 51st action.
        """
        import torch
        from g1_active_view.vision.dinov2_encoder import image_tensor
        with torch.inference_mode():
            current = self.policy.encoder(image_tensor(frame['path'])[None].to(self.policy.device))
            target = self.policy.cache[str(self.target)]
            return float(torch.nn.functional.cosine_similarity(current, target, dim=1)[0].clamp(-1, 1))

    def _launch_reader(self, frame, source):
        (self.frame, self.reading_image_source) = (copy.deepcopy(frame), source)
        self.reading_kind = 'periodic_view_check' if source == 'current_rgb_after_5_view_steps' else 'terminal_reading'
        self.record['reading_image_source'] = source
        self.read_output = self.output / f'reading_{len(self.readings) + 1:02d}.json'
        child_env = os.environ.copy()
        for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'LD_LIBRARY_PATH'):
            child_env.pop(key, None)
        https_proxy = child_env.get('https_proxy') or child_env.get('HTTPS_PROXY', '')
        if https_proxy.lower().startswith(('http://', 'https://')):
            for key in ('ALL_PROXY', 'all_proxy'):
                if child_env.get(key, '').lower().startswith('socks://'):
                    child_env.pop(key)
                    print(f'[VLM] ignored unsupported {key} scheme; existing HTTPS proxy retained', flush=True)
        self.worker = subprocess.Popen([str(self.reader_python), str(Path(__file__).with_name('live_reader_worker.py')), '--image', frame['path'], '--model', self.model, '--output', str(self.read_output)], env=child_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.read_started = time.monotonic()

    def _max_steps_best_view(self):
        return self._finish('VIEW_TIMEOUT', 'view_action_budget_exhausted_without_accepted_reading')

    def _policy_once(self):
        phase = self.state
        if phase == 'VIEW_REPRODUCING' and self.view_step_count >= self.MAX_VIEW_STEPS:
            return self._max_steps_best_view()
        frame = self._capture()
        proposal = self.policy.predict(frame['path'], str(self.target), self.action_history)
        (action_id, action) = (proposal['action_id'], proposal['action_name'])
        if action != self.action_names[action_id]:
            raise RuntimeError('policy_action_contract_mismatch')
        self.action_count += 1
        if phase == 'VIEW_REPRODUCING':
            self.view_step_count += 1
            self.record['view_steps'] = self.view_step_count
            print(f"[VIEW] step={self.view_step_count:02d}/50 action_id={action_id} action_name={action} similarity={proposal['similarity']:.6f}", flush=True)
            self._consider_view(frame, proposal['similarity'], self.view_step_count, 'before_policy_action', self.executor.state())
        else:
            print(f"[VIEW_RETRY] retry={self.retry_count}/5 action_id={action_id} action_name={action} similarity={proposal['similarity']:.6f}", flush=True)
        started = float(self.env.sim.current_time)
        save_json(self.output / f'proposal_{self.action_count:03d}.json', {'proposal': proposal, 'observation': frame, 'history': self.action_history[-8:], 'phase': phase, 'view_step': self.view_step_count, 'started_sim_time': started, 'execution_status': 'pending'})
        if action_id in (4, 5, 6, 7):
            from pxr import UsdGeom
            pan_prim = self.stage.GetPrimAtPath(self.executor.rt.controller.pan_path)
            pitch_prim = self.stage.GetPrimAtPath(self.executor.rt.controller.tilt_path)
            pan_before = UsdGeom.Xformable(pan_prim).GetLocalTransformation()
            pitch_before = UsdGeom.Xformable(pitch_prim).GetLocalTransformation()
            pan_before = [[float(x) for x in row] for row in pan_before]
            pitch_before = [[float(x) for x in row] for row in pitch_before]
        execution = self.executor.execute_action(action, caller='policy')
        self.navigation.run_command_dds.write_run_command(str(self.executor.rt.command))
        row = {'step': self.action_count, 'phase': phase, 'view_step': self.view_step_count, 'retry_step': self.retry_count if phase == 'VIEW_RETRY' else None, 'proposal': proposal, 'history_before': self.action_history[-8:], 'observation': frame, 'execution': execution, 'started_sim_time': started, 'completed_sim_time': float(self.env.sim.current_time)}
        if action_id in (4, 5, 6, 7):
            gimbal = copy.deepcopy(execution.get('gimbal') or getattr(self.executor.rt.controller, 'last_live_command', {}))
            gimbal.update(policy_action=action, deployment_semantic=execution.get('deployment_semantic'), body_yaw_commanded=execution.get('body_yaw_commanded', False), body_yaw_changed=execution.get('body_yaw_changed', abs(float(execution['after']['yaw']) - float(execution['before']['yaw'])) > 1e-06), gimbal_yaw_commanded=execution.get('gimbal_yaw_commanded', action_id in (4, 5)), gimbal_pitch_commanded=execution.get('gimbal_pitch_commanded', action_id in (6, 7)), pan_before=execution['before'].get('pan_deg'), pan_after=execution['after'].get('pan_deg'), pitch_before=execution['before'].get('tilt_deg'), pitch_after=execution['after']['tilt_deg'], local_pan_matrix_before=pan_before, local_pan_matrix_after=[[float(x) for x in r] for r in UsdGeom.Xformable(pan_prim).GetLocalTransformation()], local_pitch_matrix_before=pitch_before, local_pitch_matrix_after=[[float(x) for x in r] for r in UsdGeom.Xformable(pitch_prim).GetLocalTransformation()])
            gimbal['pan_changed'] = abs(float(gimbal.get('pan_before', execution['before'].get('pan_deg', 0.0))) - float(gimbal.get('pan_after', execution['after'].get('pan_deg', 0.0)))) > 1e-06
            gimbal['pitch_changed'] = abs(float(gimbal.get('pitch_before', execution['before']['tilt_deg'])) - float(gimbal.get('pitch_after', execution['after']['tilt_deg']))) > 1e-06
            row['gimbal'] = gimbal
            print(f"[GIMBAL] policy_action={action} deployment_semantic={gimbal.get('deployment_semantic')}\n[GIMBAL] pan_before={gimbal.get('pan_before')} pan_after={gimbal.get('pan_after')} pitch_before={gimbal.get('pitch_before')} pitch_after={gimbal.get('pitch_after')}\n[GIMBAL] body_yaw_commanded={gimbal.get('body_yaw_commanded')} body_yaw_changed={gimbal.get('body_yaw_changed')}", flush=True)
        self.actions.append(row)
        self.action_history.append(action)
        save_json(self.output / f'action_{self.action_count:03d}.json', row)
        if not execution['success']:
            return self._finish('VIEW_FAILED', execution.get('failure_reason') or 'primitive_execution_failed')
        periodic_read = phase == 'VIEW_REPRODUCING' and action_id != 8 and (self.view_step_count % self.READ_EVERY_VIEW_STEPS == 0)
        if phase == 'VIEW_RETRY' or action_id == 8 or periodic_read:
            if action_id == 8 and phase != 'VIEW_RETRY':
                self.record['stop_source'] = 'policy_action'
                self.record['stop_action_id'] = 8
                self.record.update(termination_reason='policy_stop', stop_step=self.view_step_count)
                print('[VIEW] POLICY STOP\n[VIEW] VIEW_REACHED\n[VIEW] stop_source=policy_action\n[VIEW] termination_reason=policy_stop', flush=True)
            if periodic_read:
                self.navigation.stop_navigation()
                print(f'[READ_CHECK] after_view_step={self.view_step_count}; waiting for fresh current RGB', flush=True)
            completion_frame = self.camera.get_latest_rgb()
            self.after_frame = completion_frame['frame_id']
            self.capture_after_sensor_timestamp = completion_frame['timestamp']
            self.capture_after_time = float(self.env.sim.current_time)
            self.await_fresh = True
            self.pending_reading_source = 'current_rgb_after_5_view_steps' if periodic_read else 'current_rgb_after_retry' if phase == 'VIEW_RETRY' else 'current_rgb_after_policy_stop'
            self.fresh_deadline = time.monotonic() + 10.0
            print('[CAMERA] acquiring fresh RGB', flush=True)
            self.transition('READING')
        return False

    def step(self):
        try:
            return self._step()
        except FreshFramePending:
            return False
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return self._finish('FAILED', f'{type(exc).__name__}:{exc}')

    def _step(self):
        if self.state == 'DONE':
            return True
        if self.state == 'INITIALIZING':
            s = self.navigation.status
            required = ('ros_bridge', 'odom', 'lidar', 'slam', 'prior_map_loaded', 'prior_localized', 'final_follower_ready')
            missing = [k for k in required if not s.get(k)]
            if time.monotonic() - getattr(self, 'last_init_diagnostic', 0.0) > 1.0:
                self.last_init_diagnostic = time.monotonic()
                self.record['missing_navigation_signals'] = missing
                self.record['navigation_initialization'] = dict(s)
                save_json(self.output / 'navigation_initialization.json', {'missing': missing, 'health': s})
                self._ready_file('INITIALIZING_NAVIGATION', ready=False)
                if missing != getattr(self, 'previous_init_missing', None):
                    print('[NAV] waiting for: ' + ', '.join(missing), flush=True)
                    self.previous_init_missing = missing
            if not missing:
                self.record['missing_navigation_signals'] = []
                self.record['navigation_initialization'] = dict(s)
                save_json(self.output / 'navigation_initialization.json', {'missing': [], 'health': s})
                self.transition('READY')
                self.transition('WAIT_FOR_START')
                print('========================================\nINSPECTION DEMO READY\nWaiting for START command...\n========================================', flush=True)
            elif time.monotonic() - self.created > 120:
                self.record['missing_navigation_signals'] = missing
                self.record['navigation_initialization'] = dict(s)
                return self._finish('INITIALIZATION_FAILED', 'navigation_feedback_not_ready:' + ','.join(missing))
            return False
        if self.state == 'WAIT_FOR_START':
            if not self.trigger_path.is_file():
                return False
            payload = json.loads(self.trigger_path.read_text())
            self.trigger_path.unlink()
            if payload.get('command') != 'START' or payload.get('nonce') != self.nonce or payload.get('pid') != os.getpid():
                print('[DEMO] invalid or stale START rejected', flush=True)
                return False
            print('[DEMO] START received', flush=True)
            self.transition('INIT')
            self.record['start_rgb'] = self._capture()
            actual = self.executor.state()
            mapped = self.navigation.slam_pose
            if mapped is None:
                raise RuntimeError('live_map_pose_unavailable')
            delta = mapped[2] - actual['yaw']
            dx = self.world_goal[0] - actual['position'][0]
            dy = self.world_goal[1] - actual['position'][1]
            goal = [mapped[0] + math.cos(delta) * dx - math.sin(delta) * dy, mapped[1] + math.sin(delta) * dx + math.cos(delta) * dy, self.world_goal[2] + delta]
            self.transition('NAVIGATING')
            self.nav_started = time.monotonic()
            goal_id = self.navigation.set_goal(*goal)
            self.record['navigation'] = {'goal_id': goal_id, 'map_goal': goal, 'world_goal': self.world_goal, 'frame_conversion': 'measured world/SLAM pose registration at submission (simulation-only)', 'initial_actual': actual, 'initial_map': list(mapped)}
            print(f'[NAV] goal = {goal} frame=map\n[NAV] goal submitted id={goal_id}', flush=True)
            return False
        if self.state == 'NAVIGATING':
            feedback = (self.navigation.goal_feedback or {}).get('feedback', {})
            if time.monotonic() - self.last_nav_trace >= 1.0:
                row = {'wall_time': time.time(), 'sim_time': float(self.env.sim.current_time), 'actual': self.executor.state(), 'feedback': self.navigation.goal_feedback, 'requested_velocity': list(self.navigation.latest_cmd)}
                with (self.output / 'navigation_trace.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                self.last_nav_trace = time.monotonic()
                if row['actual']['collision'] or row['actual']['fall_detected']:
                    return self._finish('NAVIGATION_FAILED', 'measured_collision_or_fall')
            if self.navigation.arrived():
                self.navigation.stop_navigation()
                self.record['navigation']['arrival_feedback'] = self.navigation.goal_feedback
                self.record['navigation']['arrived_actual'] = self.executor.state()
                print('[NAV] ARRIVED', flush=True)
                self.transition('VIEW_REPRODUCING')
                self.view_started = time.monotonic()
                print(f'[VIEW] target_image = {self.target}\n[VIEW] checkpoint = {self.checkpoint}', flush=True)
            elif time.monotonic() - self.nav_started > self.config.get('live_navigation_timeout_s', 600):
                self.record['navigation']['last_feedback'] = self.navigation.goal_feedback
                return self._finish('NAVIGATION_TIMEOUT', 'real_arrival_not_received')
            return False
        if self.state in ('VIEW_REPRODUCING', 'VIEW_RETRY'):
            limit = self.config.get('live_view_timeout_s')
            if limit is not None and time.monotonic() - self.view_started > float(limit):
                return self._finish('VIEW_TIMEOUT', 'view_wall_time_budget_exhausted')
            if self.state == 'VIEW_REPRODUCING' and self.view_step_count >= self.MAX_VIEW_STEPS:
                return self._max_steps_best_view()
            return self._policy_once()
        if self.state == 'READING':
            if self.await_fresh:
                if time.monotonic() > self.fresh_deadline:
                    return self._finish('READ_FAILED', 'no_fresh_post_action_RGB')
                frame = self._capture()
                if frame['timestamp'] <= self.capture_after_sensor_timestamp or frame['captured_sim_time'] <= self.capture_after_time:
                    raise FreshFramePending('waiting_for_sensor_timestamp_after_action')
                self.await_fresh = False
                self._launch_reader(frame, self.pending_reading_source)
                return False
            if self.worker.poll() is None:
                if time.monotonic() - self.read_started < 45:
                    return False
                self.worker.kill()
                self.worker.wait()
            result = self.reader._parse(self.read_output.read_text()) if self.read_output.is_file() else self.reader._failure('reader_worker_failed_or_timeout')
            self.readings.append({'frame': copy.deepcopy(self.frame), 'result': result, 'view_step': getattr(self, 'view_step_count', None), 'kind': getattr(self, 'reading_kind', 'terminal_reading'), 'image_source': getattr(self, 'reading_image_source', None)})
            print('[VLM] ' + json.dumps(result, ensure_ascii=False), flush=True)
            if result.get('failure_reason') in ('ArkAuthenticationError', 'missing_ARK_API_KEY'):
                return self._finish('READ_FAILED', result['failure_reason'])
            if result['readable']:
                self.record['completion_source'] = 'vlm_readable_number_and_background_color'
                self.record['reading_success_step'] = getattr(self, 'view_step_count', None)
                self.record['reading_acceptance'] = 'fresh RGB; existing Reader strict schema; readable=true; finite number; nonempty color; failure_reason=null'
                if getattr(self, 'reading_kind', None) == 'periodic_view_check':
                    self.record.update(termination_reason='periodic_reading_success', stop_source=None, stop_step=None, motion_halted_by='reading_success')
                    print('[READ_CHECK] reliable reading accepted; no further policy actions; SAVING -> DONE', flush=True)
                return self._finish('SUCCESS')
            if getattr(self, 'reading_kind', None) == 'periodic_view_check':
                print(f'[READ_CHECK] not readable after step {self.view_step_count}; resume policy search', flush=True)
                self.transition('VIEW_REPRODUCING')
                return False
            if self.retry_count >= 5:
                return self._finish('READ_FAILED', result['failure_reason'])
            self.retry_count += 1
            self.transition('VIEW_RETRY')
        return False

    def close(self):
        if self.worker and self.worker.poll() is None:
            self.worker.terminate()
            self.worker.wait(timeout=5)
        self.navigation.stop_navigation()
        if self.state != 'DONE':
            self._finish('CANCELLED', 'process_shutdown_before_completion')
        self._ready_file('CLOSED', ready=False)
        self.executor.null.close()

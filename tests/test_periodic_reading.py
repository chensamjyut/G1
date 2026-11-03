"""Periodic reading control-flow tests; doubles are not live Demo evidence."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from planner.live_single_point import LiveSinglePoint
from planner.number_color_reader import DoubaoNumberColorReader


class PeriodicReading(unittest.TestCase):
    def manager(self, output, before_step):
        m=LiveSinglePoint.__new__(LiveSinglePoint)
        m.output,m.target=Path(output),Path('policy_target_only.png')
        m.config={}
        m.state='VIEW_REPRODUCING';m.view_step_count=m.action_count=before_step
        m.action_history=[];m.actions=[];m.readings=[];m.retry_count=0
        m.action_names=['MoveXPos']+['unused']*7+['Stop'];m.record={'stop_source':None}
        m.policy,m.executor,m.navigation,m.camera=Mock(),Mock(),Mock(),Mock()
        m.policy.predict.return_value={'action_id':0,'action_name':'MoveXPos','similarity':.5}
        m.executor.execute_action.return_value={'success':True};m.executor.rt.command=[0,0,0,.8]
        m.env=SimpleNamespace(sim=SimpleNamespace(current_time=10.))
        m.camera.get_latest_rgb.return_value={'frame_id':40,'timestamp':100.}
        m._capture=Mock(return_value={'path':'pre_action_current.png'})
        m._consider_view=Mock();m.transition=Mock(side_effect=lambda s:setattr(m,'state',s))
        m._finish=Mock(side_effect=lambda *a:(setattr(m,'state','DONE') or True))
        return m

    def test_reads_after_completed_steps_5_and_10_not_step4(self):
        for before in (3,4,9):
            with self.subTest(before=before),TemporaryDirectory() as out:
                m=self.manager(out,before);m._policy_once()
                self.assertEqual(m.view_step_count,before+1)
                self.assertEqual(m.action_history,['MoveXPos'])
                if before in (4,9):
                    self.assertEqual(m.state,'READING');self.assertTrue(m.await_fresh)
                    self.assertEqual(m.pending_reading_source,'current_rgb_after_5_view_steps')
                    m.navigation.stop_navigation.assert_called_once()
                else:
                    self.assertEqual(m.state,'VIEW_REPRODUCING');m.camera.get_latest_rgb.assert_not_called()
                self.assertIsNone(m.record['stop_source'])

    def test_failed_fifth_action_does_not_attempt_read(self):
        with TemporaryDirectory() as out:
            m=self.manager(out,4)
            m.executor.execute_action.return_value={'success':False,'failure_reason':'unsafe'}
            self.assertTrue(m._policy_once());m._finish.assert_called_once_with('VIEW_FAILED','unsafe')
            m.camera.get_latest_rgb.assert_not_called()

    def test_periodic_reader_uses_fresh_post_action_image(self):
        with TemporaryDirectory() as out:
            m=self.manager(out,4);m._policy_once()
            m._capture.return_value={'path':'fresh_after_step5.png','timestamp':101.,'captured_sim_time':11.}
            m.reader_python=Path('/test/python');m.model='test-model'
            with patch('planner.live_single_point.subprocess.Popen') as launch:
                m._step();argv=launch.call_args.args[0]
            self.assertEqual(argv[argv.index('--image')+1],'fresh_after_step5.png')
            self.assertNotIn(str(m.target),argv);self.assertEqual(m.reading_kind,'periodic_view_check')

    def read_result(self,m,result):
        m.state='READING';m.await_fresh=False;m.reading_kind='periodic_view_check'
        m.reading_image_source='current_rgb_after_5_view_steps'
        m.reader=DoubaoNumberColorReader('test');m.worker=Mock();m.worker.poll.return_value=0
        m.frame={'path':'fresh_step5.png','frame_id':41};m.read_output=m.output/'reading.json'
        m.read_output.write_text(json.dumps(result))

    def test_worker_retains_https_proxy_drops_only_invalid_socks_fallback(self):
        with TemporaryDirectory() as out:
            m=self.manager(out,4);m._policy_once()
            m.reader_python=Path('/test/python');m.model='test-model'
            env={'HTTPS_PROXY':'http://127.0.0.1:7890',
                 'ALL_PROXY':'socks://127.0.0.1:7890',
                 'all_proxy':'socks://127.0.0.1:7890', 'ARK_API_KEY':'test-only'}
            with patch.dict('os.environ',env,clear=True),patch('planner.live_single_point.subprocess.Popen') as launch:
                m._launch_reader({'path':'fresh_step5.png'},'current_rgb_after_5_view_steps')
                worker_env=launch.call_args.kwargs['env']
                self.assertEqual(worker_env['HTTPS_PROXY'],env['HTTPS_PROXY'])
                self.assertEqual(worker_env['ARK_API_KEY'],'test-only')
                self.assertNotIn('ALL_PROXY',worker_env);self.assertNotIn('all_proxy',worker_env)
                import os
                self.assertEqual(os.environ['ALL_PROXY'],env['ALL_PROXY'])

    def test_readable_ends_without_step6_or_fake_policy_stop(self):
        with TemporaryDirectory() as out:
            m=self.manager(out,5)
            self.read_result(m,{'readable':True,'number':10,'background_color':'blue','failure_reason':None})
            self.assertTrue(m._step());m._finish.assert_called_once_with('SUCCESS')
            self.assertEqual(m.record['termination_reason'],'periodic_reading_success')
            self.assertEqual(m.record['reading_success_step'],5);self.assertIsNone(m.record['stop_source'])
            self.assertIsNone(m.record['stop_step']);self.assertTrue(m._step())
            m.policy.predict.assert_not_called();m.executor.execute_action.assert_not_called()

    def test_unreadable_resumes_search_without_consuming_terminal_retries(self):
        with TemporaryDirectory() as out:
            m=self.manager(out,5);self.read_result(m,DoubaoNumberColorReader._failure('unreadable'))
            self.assertFalse(m._step());self.assertEqual(m.state,'VIEW_REPRODUCING')
            self.assertEqual(m.view_step_count,5);self.assertEqual(m.retry_count,0)
            m._finish.assert_not_called();m.policy.predict.assert_not_called()
            m._step();self.assertEqual(m.view_step_count,6)
            self.assertEqual(m.state,'VIEW_REPRODUCING');m.policy.predict.assert_called_once()

    def test_invalid_reader_json_cannot_complete_task(self):
        with TemporaryDirectory() as out:
            m=self.manager(out,5);self.read_result(m,{'readable':True,'number':10,'background_color':None})
            self.assertFalse(m._step());m._finish.assert_not_called()
            self.assertEqual(m.state,'VIEW_REPRODUCING');self.assertFalse(m.readings[-1]['result']['readable'])

    def test_authentication_failure_stops_instead_of_searching_more(self):
        with TemporaryDirectory() as out:
            m=self.manager(out,5)
            self.read_result(m,DoubaoNumberColorReader._failure('ArkAuthenticationError'))
            self.assertTrue(m._step())
            m._finish.assert_called_once_with('READ_FAILED','ArkAuthenticationError')
            m.policy.predict.assert_not_called();m.executor.execute_action.assert_not_called()

    def test_real_stop_at_fifth_step_keeps_real_stop_semantics(self):
        with TemporaryDirectory() as out:
            m=self.manager(out,4);m.policy.predict.return_value={'action_id':8,'action_name':'Stop','similarity':.5}
            m._policy_once();self.assertEqual(m.record['termination_reason'],'policy_stop')
            self.assertEqual(m.record['stop_step'],5)
            self.assertEqual(m.pending_reading_source,'current_rgb_after_policy_stop')


if __name__=='__main__':
    unittest.main()

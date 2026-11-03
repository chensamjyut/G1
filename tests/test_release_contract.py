"""Boundary tests with doubles; these are NOT simulation integration results."""
import ast,importlib.util, json, math,time
from pathlib import Path
from types import SimpleNamespace
import pytest
from g1_release.resources import verify_file,sha256,mission_config
from planner.live_single_point import LiveSinglePoint
import planner.live_grpo_binding as binding

ROOT=Path(__file__).resolve().parents[1]

def extracted_class(name,base,**scope):
    tree=ast.parse(Path(binding.__file__).read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name==name)
    scope[node.bases[0].id]=base
    exec(compile(ast.Module(body=[node],type_ignores=[]),binding.__file__,'exec'),scope)
    return scope[name]

def test_xy_and_yaw_requests_cannot_command_body_yaw():
    class Base:
        def _velocity_tick(self,request,yaw_rate):return (request,yaw_rate,self.heading_hold_active)
    cls=extracted_class('LiveRuntime',Base)
    obj=cls.__new__(cls)
    for request in [(0.,0.),(.2,0.),(0.,-.2)]:
        for yaw in [-1.2,0.,1.2]:
            obj.heading_hold_active=True
            assert obj._velocity_tick(request,yaw)==(request,0.,False)
            assert obj.suppressed_body_yaw_request==yaw

def test_yaw_routing_never_reaches_body_executor():
    from planner.action_semantics import GIMBAL_YAW_ACTIONS
    class Body:
        def execute_action(self,*a,**k):raise AssertionError('body executor reached by yaw')
        def act(self,*a,**k):raise AssertionError('legacy body act reached by yaw')
    cls=extracted_class('PlannerSharedRelativeExecutor',Body,GIMBAL_YAW_ACTIONS=GIMBAL_YAW_ACTIONS)
    obj=cls.__new__(cls);seen=[]
    obj._execute_gimbal_yaw=lambda a:seen.append(a) or {'axis':'gimbal_yaw'}
    for name in sorted(GIMBAL_YAW_ACTIONS):
        assert obj.execute_action(name,caller='policy')['axis']=='gimbal_yaw'
        assert obj.act(name,{})['axis']=='gimbal_yaw'
    assert len(seen)==4

def test_budget_does_not_read_historical_frame_or_claim_stop():
    obj=LiveSinglePoint.__new__(LiveSinglePoint);out=[]
    obj._finish=lambda *args:out.append(args)
    obj._launch_reader=lambda *args:pytest.fail('historical frame passed to API')
    obj._max_steps_best_view()
    assert out==[('VIEW_TIMEOUT','view_action_budget_exhausted_without_accepted_reading')]

def test_navigation_arrival_is_feedback_not_coordinates():
    spec=importlib.util.spec_from_file_location('bridge',ROOT/'sim_runtime/navigation/isaac_navigation_bridge.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    nav=mod.IsaacNavigationBridge.__new__(mod.IsaacNavigationBridge)
    nav.goal={'goal_id':'new','x':1,'y':2,'yaw':0};nav.goal_feedback={};nav.goal_feedback_at=time.monotonic()
    assert not nav.arrived()
    nav.goal_feedback={'goal_id':'old','feedback':{'reached':True,'reason':'arrival_latched'}}
    assert not nav.arrived()
    nav.goal_feedback['goal_id']='new';assert nav.arrived()
    nav.goal_feedback_at-=2;assert not nav.arrived()

def test_nav_to_view_requires_arrived_and_stops_navigation():
    m=LiveSinglePoint.__new__(LiveSinglePoint);m.state='NAVIGATING';m.last_nav_trace=time.monotonic();m.nav_started=time.monotonic();m.config={};m.record={'navigation':{}};m.target='target.png';m.checkpoint='model.pt';m.executor=SimpleNamespace(state=lambda:{'measured':True});events=[]
    m.navigation=SimpleNamespace(goal_feedback={},arrived=lambda:False,stop_navigation=lambda:events.append('nav_stop'))
    m.transition=lambda state:events.append(state)
    assert m._step() is False;assert events==[]
    m.navigation.arrived=lambda:True
    m._step();assert events==['nav_stop','VIEW_REPRODUCING']

def test_missing_corrupt_and_lfs_weights(tmp_path):
    p=tmp_path/'weight.pt'
    with pytest.raises(FileNotFoundError):verify_file(p,'0'*64)
    p.write_bytes(b'not a model')
    with pytest.raises(ValueError,match='sha256_mismatch'):verify_file(p,'0'*64)
    p.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:'+'0'*64+'\nsize 123\n')
    with pytest.raises(ValueError,match='LFS_pointer'):verify_file(p,sha256(p))

def test_missing_resources_fail_before_simulator():
    with pytest.raises(ValueError,match='unconfigured:scene_path'):mission_config(ROOT,ROOT/'configs/mission.example.json')

def test_history_recent_eight_and_reset():
    from g1_active_view.actions import ACTIONS,encode_history
    assert encode_history([])==[9]*8
    names=list(ACTIONS)+list(ACTIONS[:3])
    assert encode_history(names)==[ACTIONS.index(n) for n in names[-8:]]
    assert encode_history([])==[9]*8

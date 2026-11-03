"""Single-point branch extracted from sim_main, using the existing G1 task.

Simulation only, loopback DDS, one existing gimbal RGB sensor. No training,
robot construction replacement, root teleport, or physical hardware backend.
"""
import argparse,json,os,sys,signal
from pathlib import Path
from types import SimpleNamespace

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--release-root',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path)
    # Parse AppLauncher arguments only after environment preparation.
    known,_=parser.parse_known_args();root=known.release_root.resolve();simroot=root/'sim_runtime'
    sys.path.insert(0,str(root/'src'))
    from g1_release.resources import mission_config
    cfg=mission_config(root,known.config)
    from gimbal_tvr.simulation_lock import acquire_simulation_lock
    lock=acquire_simulation_lock()  # Same lock as existing training: refuse a second Isaac.
    out=Path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True)
    resolved=out/'resolved_mission.local.json';resolved.write_text(json.dumps(cfg,indent=2))
    os.environ.update(PROJECT_ROOT=str(simroot),UNITREE_SIM_PROJECT_ROOT=str(simroot),
        SIM_INSPECTION_ENVIRONMENT_USD=cfg['scene_path'],G1_ROBOT_USD=cfg['robot_usd'],
        SIM_ENABLE_TASK_CAMERAS='1',DISABLE_SCENE_CAMERAS='0',DISABLE_CAMERA_OBSERVATION='1',
        G1_SINGLE_POINT_DEMO='1',ENABLE_NAVIGATION_RVIZ='0',ROS_LOCALHOST_ONLY='1',
        G1_SIM_ODOM_SCALE='1,1,1',G1_PRIOR_MAP=str(out/'geometry_prior.ply'),
        G1_EXPORT_GEOMETRY_PRIOR=str(out/'geometry_prior.ply'),ENABLE_WORLD_CAMERA='0')
    dds_home=os.environ.get('CYCLONEDDS_HOME')
    if not dds_home or not (Path(dds_home)/'lib/libddsc.so').is_file():
        raise RuntimeError('CYCLONEDDS_HOME must name the pinned CycloneDDS installation')
    dds_lib=str(Path(dds_home).resolve()/'lib')
    ros_lib=str(Path(sys.prefix)/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib')
    if os.environ.get('G1_RELEASE_DDS_REEXEC')!='1':
        ld=[dds_lib,*[x for x in os.environ.get('LD_LIBRARY_PATH','').split(':') if x and x not in (dds_lib,ros_lib)]]
        if Path(ros_lib).is_dir():ld.append(ros_lib)
        child=os.environ.copy();child.update(LD_LIBRARY_PATH=':'.join(ld),G1_RELEASE_DDS_REEXEC='1',ROS_DISTRO='humble',RMW_IMPLEMENTATION='rmw_fastrtps_cpp')
        lock.close()
        os.execve(sys.executable,[sys.executable,*sys.argv],child)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
    if not args.headless:parser.error('release single-point entry requires --headless')
    args.enable_cameras=True
    launcher=AppLauncher(args);app=launcher.app
    env=controller=navigation=manager=mission=None
    try:
        # pxr is supplied by Kit and becomes importable after AppLauncher.
        from pxr import Usd,UsdGeom,UsdUtils
        for path in (cfg['scene_path'],cfg['robot_usd']):
            layers,assets,unresolved=UsdUtils.ComputeAllDependencies(path)
            builtin={'OmniPBR.mdl','OmniGlass.mdl','OmniSurface.mdl','gltf/pbr.mdl'}
            unknown=set(unresolved)-builtin
            if unknown:raise RuntimeError('unresolved_USD_dependencies:'+repr(sorted(unknown)))
            if unresolved:
                # USD asset resolver cannot resolve MDL module identifiers; Kit's MDL resolver does.
                # Verify the required installed shader source files exist, then leave render validation explicit.
                site=Path(sys.prefix)/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
                for module in unresolved:
                    matches=[x for x in site.rglob(Path(module).name) if x.as_posix().endswith(module)]
                    if not matches:raise RuntimeError('required_Kit_MDL_module_missing:'+module)
        source=Usd.Stage.Open(cfg['scene_path'])
        if UsdGeom.GetStageMetersPerUnit(source)!=1.0:raise ValueError('scene_must_be_authored_in_metres')
        import gymnasium as gym,omni.usd
        import tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from tools.gimbal_controller import set_gimbal_angles
        from dds.dds_create import create_dds_objects
        from action_provider.action_provider_wh_dds import DDSRLActionProvider
        from layeredcontrol.robot_control_system import RobotController,ControlConfig
        from navigation.isaac_navigation_bridge import IsaacNavigationBridge
        from planner.live_single_point import LiveSinglePoint
        task='Isaac-Move-Cylinder-G129-Dex1-Wholebody'
        ecfg=parse_env_cfg(task,device=args.device,num_envs=1);ecfg.env_name=task
        ecfg.seed=42;ecfg.sim.enable_scene_query_support=True;ecfg.wait_for_textures=False
        env=gym.make(task,cfg=ecfg).unwrapped;env.seed(42)
        stage=omni.usd.get_context().get_stage()
        for path in ('/World/Looks','/Render'):
            if source.GetPrimAtPath(path):stage.OverridePrim(path).GetReferences().AddReference(cfg['scene_path'],path)
        if set_gimbal_angles(stage,0.0,float(cfg.get('gimbal_pitch_deg',-10)))!=1:
            raise RuntimeError('expected_one_existing_G1_gimbal')
        env.sim.reset();env.reset()
        a=SimpleNamespace(task=task,robot_type='g129',enable_dex1_dds=True,enable_dex3_dds=False,
            enable_inspire_dds=False,enable_wholebody_dds=True,model_path=cfg['walking_policy'])
        _,_,manager=create_dds_objects(a,env)
        if not manager.dds_initialized:raise RuntimeError('loopback_DDS_initialization_failed')
        navigation=IsaacNavigationBridge(env,stage,manager.get_object('run_command'),str(simroot))
        navigation.stop_navigation()
        provider=DDSRLActionProvider(env,a)
        controller=RobotController(env,ControlConfig(step_hz=100,use_rl_action_mode=True))
        controller.set_action_provider(provider);controller.set_profiling(False);controller.start()
        mission=LiveSinglePoint(env=env,stage=stage,controller=controller,provider=provider,
            navigation=navigation,config_path=resolved,ready_path=out/'ready.json',
            trigger_path=out/'start.json',app=app)
        def stop(signum,frame):raise KeyboardInterrupt()
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        while app.is_running():
            navigation.step(write_commands=mission.state in ('INITIALIZING','WAIT_FOR_START','INIT','NAVIGATING','DONE'))
            controller.step()
            if provider._error_trace_printed:raise RuntimeError('walking_policy_execution_failed')
            if mission.step():break
        if mission.state!='DONE':raise RuntimeError('simulation_closed_before_result')
        return 0 if mission.record.get('status')=='SUCCESS' else 2
    finally:
        # Each cleanup attempted even if a previous cleanup fails.
        for obj,method in [(mission,'close'),(navigation,'close'),(controller,'stop'),(manager,'stop_all_communication'),(env,'close'),(app,'close')]:
            if obj is not None:
                try:getattr(obj,method)()
                except Exception as exc:print('cleanup_failed',method,type(exc).__name__,file=sys.stderr)
        lock.close()
if __name__=='__main__':raise SystemExit(main())

from dds.dds_master import dds_manager

def create_dds_objects(args_cli, env):
    publish_names = []
    subscribe_names = []
    if args_cli.robot_type == 'g129' or args_cli.robot_type == 'h1_2':
        from dds.g1_robot_dds import G1RobotDDS
        g1_robot = G1RobotDDS()
        dds_manager.register_object('g129', g1_robot)
        publish_names.append('g129')
        subscribe_names.append('g129')
    if args_cli.enable_dex1_dds:
        from dds.gripper_dds import GripperDDS
        gripper = GripperDDS()
        dds_manager.register_object('dex1', gripper)
        publish_names.append('dex1')
        subscribe_names.append('dex1')
    if 'Wholebody' in args_cli.task or args_cli.enable_wholebody_dds:
        from dds.commands_dds import RunCommandDDS
        run_command_dds = RunCommandDDS()
        dds_manager.register_object('run_command', run_command_dds)
        publish_names.append('run_command')
        subscribe_names.append('run_command')
    from dds.reset_pose_dds import ResetPoseCmdDDS
    reset_pose_dds = ResetPoseCmdDDS()
    dds_manager.register_object('reset_pose', reset_pose_dds)
    subscribe_names.append('reset_pose')
    from dds.sim_state_dds import SimStateDDS
    sim_state_dds = SimStateDDS(env, args_cli.task)
    dds_manager.register_object('sim_state', sim_state_dds)
    publish_names.append('sim_state')
    from dds.rewards_dds import RewardsDDS
    rewards_dds = RewardsDDS(env, args_cli.task)
    dds_manager.register_object('rewards', rewards_dds)
    publish_names.append('rewards')
    dds_manager.start_publishing(publish_names)
    dds_manager.start_subscribing(subscribe_names)
    return (reset_pose_dds, sim_state_dds, dds_manager)

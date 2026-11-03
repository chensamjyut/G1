# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
"""
public base scene configuration module
provides reusable scene element configurations, such as tables, objects, ground, lights, etc.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import os
project_root = os.environ.get("PROJECT_ROOT")
_use_original_warehouse = os.getenv("SIM_USE_ORIGINAL_WAREHOUSE") == "1"
_inspection_environment = os.getenv("SIM_INSPECTION_ENVIRONMENT_USD", "")
@configclass
class TableCylinderSceneCfgWH(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
      # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],  # room center point
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            # The source is centimetre-authored, but its demo layout is very
            # large relative to a 1.8 m G1.  The inspection scene intentionally
            # uses the user-confirmed 0.006 scale so the robot/factory visual
            # proportion is suitable for this compact navigation demo.
            usd_path=_inspection_environment or (
                f"{project_root}/assets/objects/small_warehouse/small_warehouse_digital_twin.usd"
                if _use_original_warehouse
                else f"{project_root}/assets/objects/factory_inspection.usda"
            ),
            # fac_num.usd already contains the factory's 0.006 transform.
            scale=(1.0, 1.0, 1.0) if (_inspection_environment or _use_original_warehouse) else (0.006, 0.006, 0.006)
        ),
    )


        # 1. table configuration

    # packing_table1 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/PackingTable_1",    # table in the scene
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[-2.35644,-3.45572,-0.2],   # initial position [x, y, z]
    #                                             rot=[0.70091, 0.0, 0.0, 0.71325]), # initial rotation [x, y, z, w]
    #     spawn=UsdFileCfg(
    #         usd_path=f"{project_root}/assets/objects/PackingTable_2/PackingTable.usd",    # table model file
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),    # set to kinematic object
    #     ),
    # )


    # packing_table2 = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/PackingTable_2",    # table in the scene
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[-3.97225,-4.3424,-0.2],   # initial position [x, y, z]
    #                                             rot=[1.0, 0.0, 0.0, 0.0]), # initial rotation [x, y, z, w]
    #     spawn=UsdFileCfg(
    #         usd_path=f"{project_root}/assets/objects/PackingTable/PackingTable.usd",    # table model file
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),    # set to kinematic object
    #     ),
    # )
    # Inspection uses only objects authored by factory.usd.  Do not add the
    # old task cylinder, inventory cubes, shelf supports, or marker geometry.
    # factory.usd's floor is primarily visual. Add the standard zero-height
    # collision plane underneath it so the locomotion policy has the same
    # stable contact surface it had in the previous warehouse scene.
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=GroundPlaneCfg(
            # Keep only the collision plane.  The default Isaac Lab ground
            # asset contains a visible black grid exactly at z=0, which
            # z-fights with and visually replaces factory.usd's authored
            # coloured floor when both are rendered.
            visible=False,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        ),
    )


    # factory.usd already supplies its complete authored lighting rig (15
    # sphere lights plus rect, distant and dome lights).  Adding the generic
    # 3000-intensity dome light used by the old warehouse washed out the
    # factory's paint colours and metal materials, so no extra light is added.

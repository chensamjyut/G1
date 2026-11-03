"""
public camera configuration
include the basic configuration for different types of cameras, support scene-specific parameter customization
"""
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
import os

@configclass
class CameraBaseCfg:
    """camera base configuration class
    
    provide the default configuration for different types of cameras, support scene-specific parameter customization
    """

    @classmethod
    def get_camera_config(cls, prim_path: str='/World/envs/env_.*/Robot/d435_link/front_cam', update_period: float=0.02, height: int=480, width: int=640, focal_length: float=7.6, focus_distance: float=400.0, horizontal_aperture: float=20.0, clipping_range: tuple=(0.1, 100000.0), pos_offset: tuple=(0, 0.0, 0), rot_offset: tuple=(0.5, -0.5, 0.5, -0.5), convention: str='ros', data_types: list=None) -> CameraCfg:
        """get the front camera configuration
        
        Args:
            prim_path: the path of the camera in the scene
            update_period: update period (seconds)
            height: image height (pixels)
            width: image width (pixels)
            focal_length: focal length
            focus_distance: focus distance
            horizontal_aperture: horizontal aperture
            clipping_range: clipping range (near clipping plane, far clipping plane)
            pos_offset: position offset (x, y, z)
            rot_offset: rotation offset quaternion
            data_types: data type list
            
        Returns:
            CameraCfg: camera configuration
        """
        if data_types is None:
            data_types = ['rgb']
        return CameraCfg(prim_path=prim_path, update_period=update_period, height=height, width=width, data_types=data_types, spawn=sim_utils.PinholeCameraCfg(focal_length=focal_length, focus_distance=focus_distance, horizontal_aperture=horizontal_aperture, clipping_range=clipping_range), offset=CameraCfg.OffsetCfg(pos=pos_offset, rot=rot_offset, convention=convention))

@configclass
class CameraPresets:

    @classmethod
    def g1_comp_inspection_camera(cls) -> CameraCfg:
        """RGB/depth camera mounted on the simulated g1_comp pan/tilt head."""
        return CameraBaseCfg.get_camera_config(prim_path='/World/envs/env_.*/Robot/torso_link/inspection_gimbal/gimbal_yaw_joint/gimbal_pitch_joint/inspection_camera_mount/inspection_camera', height=480, width=640, update_period=0.05, data_types=['rgb', 'distance_to_image_plane'], focal_length=12.0, focus_distance=4.0, horizontal_aperture=20.0, clipping_range=(0.1, 30.0), rot_offset=(1.0, 0.0, 0.0, 0.0), convention='world')

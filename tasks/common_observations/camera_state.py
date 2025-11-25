# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
camera state
"""     

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import numpy as np
import sys
import os

# add the project root directory to the path, so that the shared memory tool can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from image_server.shared_memory_utils import MultiImageWriter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# create the global multi-image shared memory writer
multi_image_writer = MultiImageWriter()

def get_camera_image(
    env: ManagerBasedRLEnv,
) -> dict:
    # pass
    """get multiple camera images and write them to shared memory
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
    
    Returns:
        dict: dictionary containing multiple camera images
    """
    # get the camera images
    images = {}
    # env.sim.render()
    
    head_sources = []
    if "front_camera_left" in env.scene.keys():
        head_sources.append(env.scene["front_camera_left"].data.output["rgb"][0].cpu().numpy())
    if "front_camera_right" in env.scene.keys():
        head_sources.append(env.scene["front_camera_right"].data.output["rgb"][0].cpu().numpy())

    # Fallback to single camera when stereo pair doesn't exist
    if not head_sources and "front_camera" in env.scene.keys():
        head_sources.append(env.scene["front_camera"].data.output["rgb"][0].cpu().numpy())

    if head_sources:
        head_image = np.concatenate(head_sources, axis=1) if len(head_sources) > 1 else head_sources[0]
        images["head"] = head_image
    
    # Left camera (left wrist camera)
    if "left_wrist_camera" in env.scene.keys():
        left_image = env.scene["left_wrist_camera"].data.output["rgb"][0]
        images["left"] = left_image.cpu().numpy()
    
    # Right camera (right wrist camera)  
    if "right_wrist_camera" in env.scene.keys():
        right_image = env.scene["right_wrist_camera"].data.output["rgb"][0]
        images["right"] = right_image.cpu().numpy()

    # if no camera with the specified name is found, try other common camera names
    if not images:
        # try to find other possible camera names
        available_cameras = [name for name in env.scene.keys() if "camera" in name.lower()]
        print(f"[camera_state] No standard cameras found. Available cameras: {available_cameras}")
        
        # if there are available cameras, use the first three as head, left, right
        for i, camera_name in enumerate(available_cameras[:3]):
            camera_image = env.scene[camera_name].data.output["rgb"][0]
            
            if i == 0:
                images["head"] = camera_image.cpu().numpy()
            elif i == 1:
                images["left"] = camera_image.cpu().numpy()
            elif i == 2:
                images["right"] = camera_image.cpu().numpy()
    
    # write the multi-image data to shared memory
    if images:
        success = multi_image_writer.write_images(images)
    else:
        print("[camera_state] No camera images found in the environment")
    
    return torch.zeros((1, 480, 640, 3))

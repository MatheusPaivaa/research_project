# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import torch

import isaaclab.utils.math as math_utils


def camera_follow(env):
    """
    Smoothly updates the simulation camera to follow the robot in the scene.

    This function computes a position behind and slightly above the robot
    using its current world position and orientation, applies smoothing 
    over several frames, and updates the camera view accordingly.

    Args:
        env: The simulation environment object, expected to have a scene and camera controller.
    """

    # Initialize persistent list for storing camera positions (used for smoothing)
    if not hasattr(camera_follow, "smooth_camera_positions"):
        camera_follow.smooth_camera_positions = []

    # Get the robot's world position and orientation (first robot in the scene)
    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]

    # Define a camera offset (behind and above the robot)
    camera_offset = torch.tensor([-5.0, 0.0, 1.0], dtype=torch.float32, device=env.device)

    # Transform the offset to world space using the robot's pose
    camera_pos = math_utils.transform_points(
        camera_offset.unsqueeze(0), 
        pos=robot_pos.unsqueeze(0), 
        quat=robot_quat.unsqueeze(0)
    ).squeeze(0)

    # Ensure the camera stays above the ground (z >= 0.1)
    camera_pos[2] = torch.clamp(camera_pos[2], min=0.1)

    # Smoothing over the last N camera positions
    window_size = 50
    camera_follow.smooth_camera_positions.append(camera_pos)

    # Keep the buffer size fixed
    if len(camera_follow.smooth_camera_positions) > window_size:
        camera_follow.smooth_camera_positions.pop(0)

    # Compute the smoothed camera position
    smooth_camera_pos = torch.mean(torch.stack(camera_follow.smooth_camera_positions), dim=0)

    # Set camera to follow the first environment index (assumes single-agent env)
    env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)

    # Update the camera view to the smoothed position, looking at the robot
    env.unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.cpu().numpy(), 
        lookat=robot_pos.cpu().numpy()
    )

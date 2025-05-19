# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Fancy sensor setup with ANYmal-D:
- RGB + Semantic Camera
- Height Scanner
- Contact Force Sensor

Run with:
    ./isaaclab.sh -p scripts/custom/anymal_d_sensors.py --enable_cameras
"""

import argparse
from isaaclab.app import AppLauncher

# CLI argument parsing
parser = argparse.ArgumentParser(description="Fancy sensor setup on ANYmal-D.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Isaac Lab Imports ---
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

# Load ANYmal-D config
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  #Use ANYmal-D

@configclass
class FancySceneCfg(InteractiveSceneCfg):
    """Scene config with ANYmal-D and multiple sensors."""
    
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3500.0, color=(1.0, 0.9, 0.8))
    )

    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    rgb_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/rgb_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.1, 1000.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.5, 0.0, 0.2), rot=(0.707, 0.0, 0.707, 0.0), convention="ros"),
    )

    semantic_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/seg_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.1, 1000.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.5, 0.2, 0.2), rot=(0.707, 0.0, 0.707, 0.0), convention="ros"),
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 1.5]),
        debug_vis=True,
        mesh_prim_paths=["/World/Ground"],
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
    )


def run_sim(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        if count % 500 == 0:
            print("\n[RESET] Resetting robot and sensors...\n")
            count = 0
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = scene["robot"].data.default_joint_pos.clone() + torch.rand_like(scene["robot"].data.default_joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, scene["robot"].data.default_joint_vel)
            scene.reset()

        # Apply action
        scene["robot"].set_joint_position_target(scene["robot"].data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # --- Sensor logs ---
        if count % 100 == 0:
            print(f"[Time: {sim_time:.2f}s]")
            print(f"RGB:       {scene['rgb_camera'].data.output['rgb'].shape}")
            print(f"Semantic:  {scene['semantic_camera'].data.output['semantic_segmentation'].shape}")
            print(f"Height Z+: {torch.max(scene['height_scanner'].data.ray_hits_w[..., -1]).item():.3f}")
            print(f"Contact F: {torch.max(scene['contact_sensor'].data.net_forces_w).item():.2f} N")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[4, 3, 3], target=[0.0, 0.0, 0.5])

    scene_cfg = FancySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[Setup complete. Starting simulation...]")
    run_sim(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()

"""
Fancy sensor setup with ANYmal-D:
- Height Scanner

Run with:
    ./isaaclab.sh -p scripts/custom/anymal_d_sensors.py --num_envs ...
"""

import sys
import os
import argparse
from isaaclab.app import AppLauncher

# CLI argument parsing
parser = argparse.ArgumentParser(description="Setup ANYmal-D and Terrains.")
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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from terrain.terrain_generator_cfg import get_unique_terrain_cfg, get_multiple_terrains_cfg
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

@configclass
class FancySceneCfg(InteractiveSceneCfg):
    """Scene config with ANYmal-D and terrain config."""
    
    # Terrain config 
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=get_unique_terrain_cfg(num_rows=1, num_cols=3),
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # Add global lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Add robot (Anymal_C)
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # --- Adding ensors ---

    # Heigh scanner config
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
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
        if count % 20 == 0:
            print(f"[Time: {sim_time:.2f}s]")
            # print(f"RGB:       {scene['rgb_camera'].data.output['rgb'].shape}")
            # print(f"Semantic:  {scene['semantic_camera'].data.output['semantic_segmentation'].shape}")
            print(f"Height Z+: {torch.max(scene['height_scanner'].data.ray_hits_w[..., -1]).item():.3f}")
            # print(f"Contact F: {torch.max(scene['contact_sensor'].data.net_forces_w).item():.2f} N")


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
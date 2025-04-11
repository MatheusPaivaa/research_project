from isaaclab.app import AppLauncher

app_launcher = AppLauncher()
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from terrain_generator_cfg import MULTIPLE_TERRAINS_CFG, UNIQUE_TERRAINS_CFG

@configclass
class TerrainSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=UNIQUE_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=2.0,
        ),
        debug_vis=True,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

def main():
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([10.0, 10.0, 5.0], [0.0, 0.0, 0.0])

    scene_cfg = TerrainSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Terreno gerado. Pressione ESC para fechar.")

    while simulation_app.is_running():
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()

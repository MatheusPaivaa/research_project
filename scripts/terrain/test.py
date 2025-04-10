# SPDX-License-Identifier: BSD-3-Clause

"""Script to visualize only the ROUGH terrain defined in ROUGH_TERRAINS_CFG."""

from isaaclab.app import AppLauncher

# inicia a simulação
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from terrain_generator import WAVE_ONLY_CFG

@configclass
class TerrainSceneCfg(InteractiveSceneCfg):
    """Cena com apenas o terreno rough gerado."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=WAVE_ONLY_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAACLAB_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


def main():
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([10.0, 10.0, 5.0], [0.0, 0.0, 0.0])

    scene_cfg = TerrainSceneCfg(num_envs=1, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Terreno rough gerado. Pressione ESC para fechar.")

    while simulation_app.is_running():
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()

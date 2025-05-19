import isaaclab.terrains as terrain_gen
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.utils import configclass
import trimesh
import numpy as np

MeshPlaneTerrainCfg = terrain_gen.MeshPlaneTerrainCfg

@configclass
class FlatOilTerrainCfg(MeshPlaneTerrainCfg):
    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg(
        static_friction=0.03,
        dynamic_friction=0.01,
        restitution=0.0,
        friction_combine_mode="min",
        restitution_combine_mode="multiply",
    )

def get_multiple_terrains_cfg(num_rows=10, num_cols=20, size=(8.0, 8.0)) -> terrain_gen.TerrainGeneratorCfg:
    return terrain_gen.TerrainGeneratorCfg(

        # Curriculum learning activation
        # curriculum=True,
        # difficulty_range=(0.0, 1.0),

        size=size,
        border_width=20.0,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={

            "wave_terrain": terrain_gen.HfWaveTerrainCfg(
                proportion=0.2, 
                amplitude_range=(0.02, 0.06), 
                num_waves=8, 
                border_width=0.25
            ),

            "stepping_stones_terrain": terrain_gen.HfSteppingStonesTerrainCfg(
                proportion=0.2,
                stone_height_max=0.05,
                stone_width_range=(0.2, 0.4),
                stone_distance_range=(0.03, 0.08),
                border_width=0.25
            ),

            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.2, 
                grid_width=0.45, 
                grid_height_range=(0.05, 0.08), 
                platform_width=1.0
            ),

            "flat": terrain_gen.MeshPlaneTerrainCfg(
                proportion=0.2,
            ),

            "flat_oil": FlatOilTerrainCfg(
                proportion=0.2
            )
        }
    )

def get_unique_terrain_cfg(
    num_rows=4,
    num_cols=4,
    size=(8.0, 8.0),
    selected_terrain: str = "boxes",  # padrão: "boxes"
) -> terrain_gen.TerrainGeneratorCfg:

    all_sub_terrains = {
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=1.0,
            amplitude_range=(0.065, 0.065),
            num_waves=8,
            border_width=0.25
        ),
        "stepping_stones_terrain": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=1.0,
            stone_height_max=0.065,
            stone_width_range=(0.4, 0.4),
            stone_distance_range=(0.05, 0.05),
            border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,
            grid_width=0.45,
            grid_height_range=(0.065, 0.065),
            platform_width=1.0
        ),
        "flat_oil": FlatOilTerrainCfg(
            proportion=1.0
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=1.0
        )
    }

    if selected_terrain not in all_sub_terrains:
        raise ValueError(f"Terreno '{selected_terrain}' não encontrado. Opções disponíveis: {list(all_sub_terrains.keys())}")

    return terrain_gen.TerrainGeneratorCfg(
        seed=42,
        size=size,
        border_width=10.0,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={selected_terrain: all_sub_terrains[selected_terrain]}
    )


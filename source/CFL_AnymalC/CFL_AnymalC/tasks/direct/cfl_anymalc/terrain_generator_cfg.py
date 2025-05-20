import isaaclab.terrains as terrain_gen
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.utils import configclass
import trimesh
import numpy as np

MeshPlaneTerrainCfg = terrain_gen.MeshPlaneTerrainCfg

@configclass
class FlatOilTerrainCfg(MeshPlaneTerrainCfg):
    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg(
        static_friction=0.0,
        dynamic_friction=0.0,
        restitution=0.0,
        friction_combine_mode="min",
        restitution_combine_mode="multiply",
    )

def get_terrain_cfg(
    selected_terrain: str = "flat", 
    num_rows: int = 4, 
    num_cols: int = 4, 
    size: tuple = (8.0, 8.0)
) -> terrain_gen.TerrainGeneratorCfg:

    base_terrains = {
        "waves": terrain_gen.HfWaveTerrainCfg(
            amplitude_range=(0.08, 0.08),
            num_waves=8,
            border_width=0.25
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            stone_height_max=0.05,
            stone_width_range=(0.2, 0.4),
            stone_distance_range=(0.03, 0.08),
            border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            grid_width=0.45,
            grid_height_range=(0.05, 0.08),
            platform_width=1.0
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(),
        "flat_oil": FlatOilTerrainCfg()
    }

    if selected_terrain == "all":
        sub_terrains = {
            name: terrain.__class__(**{**vars(terrain), "proportion": 0.2})
            for name, terrain in base_terrains.items()
        }
    else:
        if selected_terrain not in base_terrains:
            raise ValueError(f"Terrain '{selected_terrain}' not found. Avaliable option: {list(base_terrains.keys())}")
        terrain_cfg = base_terrains[selected_terrain]
        terrain_cfg.proportion = 1.0
        sub_terrains = {selected_terrain: terrain_cfg}

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
        sub_terrains=sub_terrains
    )

import isaaclab.terrains as terrain_gen
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.utils import configclass
import trimesh
import numpy as np

MeshPlaneTerrainCfg = terrain_gen.MeshPlaneTerrainCfg

def get_terrain_cfg(
    selected_terrain: str = "flat",
    eval: bool = False, 
    num_rows: int = 4, 
    num_cols: int = 4, 
    size: tuple = (8.0, 8.0)
) -> terrain_gen.TerrainGeneratorCfg:

    base_terrains = {
        "waves": terrain_gen.HfWaveTerrainCfg(
            amplitude_range = (0.10, 0.10) if eval else (0.03, 0.12),
            num_waves = 8,
            border_width = 0.25
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            stone_height_max = 0.05,
            stone_width_range = (0.3, 0.3) if eval else (0.2, 0.4),
            stone_distance_range = (0.06, 0.06) if eval else (0.03, 0.08),
            border_width = 0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            grid_width = 0.45,
            grid_height_range = (0.08, 0.08) if eval else (0.05, 0.08),
            platform_width = 1.0
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(),
        "flat_oil": terrain_gen.MeshPlaneTerrainCfg()
    }

    extra_terrains = {
        "pyramid": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        )
    }

    all_terrains = {**base_terrains, **extra_terrains}

    if selected_terrain == "all":
        sub_terrains = {}
        for name, terrain in base_terrains.items():
            terrain_cfg = terrain.__class__(**{**vars(terrain), "proportion": 0.2})
            terrain_cfg.name = name
            sub_terrains[name] = terrain_cfg
    else:
        if selected_terrain not in all_terrains:
            raise ValueError(f"Terrain '{selected_terrain}' not found. Available options: {list(all_terrains.keys())}")
        terrain_cfg = all_terrains[selected_terrain]
        terrain_cfg.proportion = 1.0
        terrain_cfg.name = selected_terrain
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

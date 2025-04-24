import isaaclab.terrains as terrain_gen


def get_multiple_terrains_cfg(num_rows=4, num_cols=4, size=(8.0, 8.0)) -> terrain_gen.TerrainGeneratorCfg:
    return terrain_gen.TerrainGeneratorCfg(
        size=size,
        border_width=10.0,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "wave_terrain": terrain_gen.HfWaveTerrainCfg(
                proportion=0.25, 
                amplitude_range=(0.0, 0.2), 
                num_waves=8, 
                border_width=0.25
            ),
            "stepping_stone_terrain": terrain_gen.HfSteppingStonesTerrainCfg(
                proportion=0.25,
                stone_height_max=0.05,
                stone_width_range=(0.2, 0.4),
                stone_distance_range=(0.03, 0.08),
                border_width=0.25
            ),
            "gravel_terrain": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.25,
                noise_range=(0.02, 0.05),
                noise_step=0.01,
                border_width=0.25
            ),
            "rails_terrain": terrain_gen.MeshRailsTerrainCfg(
                proportion=0.25,
                rail_thickness_range=(0.05, 0.1),
                rail_height_range=(0.05, 0.15),
                platform_width=1.0
            )
        }
    )


def get_unique_terrain_cfg(num_rows=4, num_cols=4, size=(8.0, 8.0)) -> terrain_gen.TerrainGeneratorCfg:
    return terrain_gen.TerrainGeneratorCfg(
        size=size,
        border_width=10.0,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "wave_terrain": terrain_gen.HfWaveTerrainCfg(
                proportion=1.0,
                amplitude_range=(0.06, 0.1),
                num_waves=8,
                border_width=0.25
            )
        }
    )

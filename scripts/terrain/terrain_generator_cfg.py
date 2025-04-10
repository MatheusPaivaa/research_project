from omni.isaac.lab.terrains import TerrainGeneratorCfg, HfWaveTerrainCfg

WAVE_ONLY_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "wave_terrain": HfWaveTerrainCfg(
            proportion=1.0, amplitude_range=(0.1, 0.2), num_waves=4, border_width=0.0
        )
    },
)

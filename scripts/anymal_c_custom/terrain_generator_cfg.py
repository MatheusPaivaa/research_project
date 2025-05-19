import isaaclab.terrains as terrain_gen
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.utils import configclass
import trimesh
import numpy as np

MeshPlaneTerrainCfg = terrain_gen.MeshPlaneTerrainCfg

# def generate_flat_oil_with_difficulty(difficulty: float, cfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
#     size_x, size_y = cfg.size
#     thickness = 0.02

#     # Cria um plano simples como mesh
#     ground = trimesh.creation.box(extents=(size_x, size_y, thickness))
#     ground.apply_translation((size_x / 2, size_y / 2, -thickness / 2))

#     # Define atritos variáveis conforme a dificuldade
#     max_friction = 0.3  # mais fácil
#     min_friction = 0.02  # mais difícil
#     static_friction = np.clip(max_friction - difficulty * (max_friction - min_friction), min_friction, max_friction)
#     dynamic_friction = static_friction * 0.75

#     # Atribui o material diretamente no cfg (se for suportado)
#     cfg.physics_material.static_friction = static_friction
#     cfg.physics_material.dynamic_friction = dynamic_friction

#     return [ground], np.array([0.0, 0.0, 0.0])

# @configclass
# class FlatOilTerrainCfg(SubTerrainBaseCfg):
#     function = staticmethod(generate_flat_oil_with_difficulty)
#     size: tuple[float, float] = (8.0, 8.0)
#     proportion: float = 0.2
#     physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg(
#         static_friction=0.2,
#         dynamic_friction=0.15,
#         restitution=0.0,
#         compliant_contact_stiffness=1e5,
#         compliant_contact_damping=1e3,
#         friction_combine_mode="min",
#         restitution_combine_mode="multiply",
#     )

@configclass
class FlatOilTerrainCfg(MeshPlaneTerrainCfg):
    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg(
        static_friction=0.2,
        dynamic_friction=0.15,
        restitution=0.0,
        friction_combine_mode="min",
        restitution_combine_mode="multiply",
    )

def get_multiple_terrains_cfg(num_rows=10, num_cols=20, size=(8.0, 8.0)) -> terrain_gen.TerrainGeneratorCfg:
    return terrain_gen.TerrainGeneratorCfg(
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

def get_unique_terrain_cfg(num_rows=4, num_cols=4, size=(8.0, 8.0)) -> terrain_gen.TerrainGeneratorCfg:
    return terrain_gen.TerrainGeneratorCfg(
        # curriculum=True,  # ativa o curriculum
        # difficulty_range=(0.0, 1.0),  # mesmo intervalo
        size=size,
        border_width=10.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            # "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            #     proportion=1.0,
            #     amplitude_range=(0.02, 0.06),
            #     num_waves=8,
            #     border_width=0.25
            # )
            # "stepping_stone_terrain": terrain_gen.HfSteppingStonesTerrainCfg(
            #     proportion=1.0,
            #     stone_height_max=0.05,
            #     stone_width_range=(0.2, 0.4),
            #     stone_distance_range=(0.03, 0.08),
            #     border_width=0.25
            # )
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.2, 
                grid_width=0.45, 
                grid_height_range=(0.05, 0.08), 
                platform_width=1.0
            )
        }
    )

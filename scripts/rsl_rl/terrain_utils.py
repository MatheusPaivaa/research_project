# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import isaaclab.sim as sim_utils

from CFLAnymalC.tasks.manager_based.cflanymalc.mdp.terrain import get_terrain_cfg


def apply_overrides_play(env_cfg, args) -> None:
    ter = args.tested_terrain
    
    if ter not in {"sand", "stepping_ice", "flat"}:
        rows, cols = (1, 1) if (args.unique or args.keyboard) else (10, 20)
        env_cfg.scene.terrain.terrain_generator = get_terrain_cfg(
            selected_terrain=ter, num_rows=rows, num_cols=cols, eval=True
        )
    elif ter == "stepping_ice":
        env_cfg.scene.terrain.terrain_generator = get_terrain_cfg(
            selected_terrain="boxes", num_rows=1, num_cols=1
        )

    if ter == "flat":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None

    elif ter == "flat_oil":
        env_cfg.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.15, dynamic_friction=0.10,
            friction_combine_mode="multiply", restitution_combine_mode="multiply"
        )

    elif ter == "stepping_ice":
        env_cfg.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.2, dynamic_friction=0.15,
            friction_combine_mode="multiply", restitution_combine_mode="multiply"
        )

    elif ter == "sand":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.2, dynamic_friction=0.9, restitution=0.0,
            improve_patch_friction=True,
            friction_combine_mode="average", restitution_combine_mode="min",
            compliant_contact_stiffness=500.0, compliant_contact_damping=5.0
        )
    else:  # default
        env_cfg.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0,
            friction_combine_mode="multiply", restitution_combine_mode="multiply"
        )

def apply_overrides_train(env_cfg, args) -> None:
    if args.terrain is not None and args.terrain != "sand" and args.terrain != "stepping_ice":
        env_cfg.scene.terrain.terrain_generator = get_terrain_cfg(
            selected_terrain=args.terrain,
            num_rows=10,
            num_cols=20,
        )

    if args.terrain == "flat_oil":
        env_cfg.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.2,
            dynamic_friction=0.15,
        )
    elif args.terrain == "flat":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
    elif args.terrain == "stepping_ice":

        env_cfg.scene.terrain.terrain_generator = get_terrain_cfg(
            selected_terrain="boxes",
            num_rows=10,
            num_cols=20,
        )
        env_cfg.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.2,
            dynamic_friction=0.15,
        )
    elif args.terrain == "sand":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.2,
            dynamic_friction=0.9, 
            restitution=0.3,  

            improve_patch_friction=True, 

            friction_combine_mode="average",   
            restitution_combine_mode="min",          

            compliant_contact_stiffness=600.0,        
            compliant_contact_damping=5.0             
        )
    else:
        env_cfg.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1,
            dynamic_friction=1,
        )
        
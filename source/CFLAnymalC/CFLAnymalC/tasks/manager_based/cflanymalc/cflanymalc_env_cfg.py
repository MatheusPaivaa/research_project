# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from . import mdp

from CFLAnymalC.tasks.manager_based.cflanymalc.mdp.terrain import get_terrain_cfg

@configclass
class AnymalCDefaultEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: mdp.MySceneCfg = mdp.MySceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: mdp.ObservationsCfg = mdp.ObservationsCfg()
    actions: mdp.ActionsCfg = mdp.ActionsCfg()
    commands: mdp.CommandsCfg = mdp.CommandsCfg()

    # MDP settings
    rewards: mdp.RewardsCfg = mdp.RewardsCfg()
    terminations: mdp.TerminationsCfg = mdp.TerminationsCfg()
    events: mdp.EventCfg = mdp.EventCfg()
    curriculum: mdp.CurriculumCfg = mdp.CurriculumCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class AnymalCFlatEnvCfg(AnymalCDefaultEnvCfg):

    def __post_init__(self):

        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None


@configclass
class AnymalCRoughEnvCfg(AnymalCDefaultEnvCfg):

    def __post_init__(self):
        super().__post_init__()


@configclass
class AnymalCPlayFlatEnvCfg(AnymalCFlatEnvCfg):
    curriculum: None = None

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = False

        self.scene.terrain.max_init_terrain_level = None

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.terrain_perf = None
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material, mode="startup",
            params=dict(
                asset_cfg=SceneEntityCfg("robot", body_names=".*"),
                static_friction_range=(0.8, 0.8),     
                dynamic_friction_range=(0.6, 0.6),
                restitution_range=(0.0, 0.0),
                num_buckets=64,
            ),
        )
        
        self.scene.terrain.terrain_type = "generator"

    
@configclass
class AnymalCPlayRoughEnvCfg(AnymalCRoughEnvCfg):
    curriculum: None = None

    def __post_init__(self):
        super().__post_init__()

        self.scene.env_spacing = 2.5

        self.scene.terrain.max_init_terrain_level = None

        self.observations.policy.enable_corruption = False
        self.terminations.terrain_out_of_bounds = None

        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material, mode="startup",
            params=dict(
                asset_cfg=SceneEntityCfg("robot", body_names=".*"),
                static_friction_range=(0.8, 0.8),     
                dynamic_friction_range=(0.6, 0.6),
                restitution_range=(0.0, 0.0),
                num_buckets=64,
            ),
        )


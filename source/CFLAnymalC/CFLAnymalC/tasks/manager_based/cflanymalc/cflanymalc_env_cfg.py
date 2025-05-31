# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from . import mdp

from terrain_generator_cfg import get_terrain_cfg

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
        """Post initialization."""
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
class AnymalCPlayEnvCfg(AnymalCFlatEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    curriculum: None = None

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

    


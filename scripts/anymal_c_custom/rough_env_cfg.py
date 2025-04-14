from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

# Pre-defined configs
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

@configclass
class AnymalCRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Switch robot to anymal-c
        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class AnymalCRoughEnvCfg_PLAY(AnymalCRoughEnvCfg):
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # Reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Disable randomization for play
        self.observations.policy.enable_corruption = False

        # Remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

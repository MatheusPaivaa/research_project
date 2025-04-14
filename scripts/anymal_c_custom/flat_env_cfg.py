from isaaclab.utils import configclass

from .rough_env_cfg import AnymalCRoughEnvCfg

@configclass
class AnymalCCustomFlatEnvCfg(AnymalCRoughEnvCfg):
    def __post_init__(self):

        # Post init of parent
        super().__post_init__()

        # Override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5

        # Change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # No height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # No terrain curriculum
        self.curriculum.terrain_levels = None


class AnymalCCustomFlatEnvCfg_PLAY(AnymalCFlatEnvCfg):
    def __post_init__(self) -> None:
        # Post init of parent
        super().__post_init__()

        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable randomization for play
        self.observations.policy.enable_corruption = False

        # Remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

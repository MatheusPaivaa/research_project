import math
import torch
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers.command_manager import CommandTerm, CommandTermCfg

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

class FixedVelocityCommand(CommandTerm):
    def __init__(self, cfg: "FixedVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self._commands = torch.tensor(cfg.default_command, device=self.device).repeat(self.num_envs, 1)

    def reset(self, env_ids):
        self._commands[env_ids.to(dtype=torch.long)] = self._commands[env_ids.to(dtype=torch.long)]
        return {}

    def update_commands(self, new_commands: torch.Tensor):
        assert new_commands.shape == self._commands.shape, \
            f"Expected shape {self._commands.shape}, got {new_commands.shape}"
        self._commands = new_commands.clone()

    def update_commands_for_ids(self, env_ids: torch.Tensor, new_cmds: torch.Tensor):
        """
        Atualiza os comandos somente para os ambientes especificados.

        Args:
            env_ids (torch.Tensor): IDs dos ambientes a serem atualizados. Shape: (N,)
            new_cmds (torch.Tensor): Novos comandos. Shape: (N, 3)
        """
        assert new_cmds.shape[0] == env_ids.shape[0], "new_cmds and env_ids must have matching batch sizes"
        assert new_cmds.shape[1] == self._commands.shape[1], "Command dimension mismatch"

        self._commands[env_ids.to(dtype=torch.long)] = new_cmds.to(self.device, dtype=self._commands.dtype)

    def _resample_command(self, env_ids: torch.Tensor):
        pass

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

    @property
    def command(self):
        return self._commands

@configclass
class FixedVelocityCommandCfg(CommandTermCfg):
    """Configuration for fixed velocity command."""
    class_type = FixedVelocityCommand
    asset_name: str = "robot"
    default_command: list = [1.0, 0.0, 0.0]
    resampling_time_range: tuple[float, float] = (10.0, 10.0) 



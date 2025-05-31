from isaaclab.utils import configclass
from typing import List, Tuple, Set
from dataclasses import field
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import torch
import numpy as np
import sys
import pickle
import os

from isaaclab.envs import ManagerBasedRLEnv

curriculum_step_counter = 0
SAVE_INTERVAL = 3000 
SAVE_DIR = "./logs/curriculum_logs" 
os.makedirs(SAVE_DIR, exist_ok=True)


grid_curriculum = None  # Global variable to hold the curriculum manager instance

class GridCurriculumManager:
    """
    Manages curriculum based on discretized command bins in a 2D grid.

    Attributes:
        device: Target device for tensors.
        steps: Step size between bins.
        max_range: Max absolute value for velocity ranges.
        start_range: Initial command range.
        unlocked_bins: Active unlocked grid bins.
    """

    def __init__(
        self,
        device: torch.device,
        num_envs: int,
        steps: float = 0.2,
        max_range: float = 3.0,
        start_range: float = 0.5,
    ) -> None:
        
        self.device = device
        self.steps = steps
        self.max_range = max_range
        self.start_range = start_range
  
        self.EPS_V = 0.15
        self.EPS_W = 0.25

        self.cmd_map, self.lin_vals, self.ang_vals = self._build_command_grid()
        self.dx = (self.lin_vals[1] - self.lin_vals[0]).item()
        self.dz = (self.ang_vals[1] - self.ang_vals[0]).item()
        self.unlocked_bins: Set[Tuple[float, float]] = set()

    def _build_command_grid(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds a grid of discretized [lin_vel_x, ang_vel_z] command bins.

        Returns:
            Tuple of:
                cmd_map: Flattened grid of command pairs (N, 2).
                lin_vals: Discrete values along lin_vel_x.
                ang_vals: Discrete values along ang_vel_z.
        """
        steps = int(2 * self.max_range / self.steps) + 1
        lin_vals = torch.linspace(-self.max_range, self.max_range, steps=steps, device=self.device)
        ang_vals = torch.linspace(-self.max_range, self.max_range, steps=steps, device=self.device)
        grid_lin, grid_ang = torch.meshgrid(lin_vals, ang_vals, indexing='ij')
        cmd_map = torch.stack([grid_lin.reshape(-1), grid_ang.reshape(-1)], dim=1)
        return cmd_map, lin_vals, ang_vals

    def map_commands_to_bins(self, cmds: torch.Tensor) -> torch.Tensor:
        """
        Maps each command to its nearest bin in the discretized command grid.

        Args:
            cmds: Commands of shape (E, 2).

        Returns:
            Tensor: Closest bin for each input command (E, 2).
        """
        distances = torch.norm(cmds.unsqueeze(1) - self.cmd_map.unsqueeze(0), dim=2)
        nearest_idx = torch.argmin(distances, dim=1)
        return self.cmd_map[nearest_idx]

    def update_unlocked_bins(self, matched_cmds: torch.Tensor, v_error: torch.Tensor, w_error: torch.Tensor):
        """
        Updates the set of unlocked bins using tracking errors (instead of rewards).

        Args:
            matched_cmds: Mapped command bins (E, 2).
            v_error: Absolute error of linear velocity (E,).
            w_error: Absolute error of angular velocity (E,).
        """
        cell_perf = {}
        for i in range(matched_cmds.shape[0]):
            key = tuple(matched_cmds[i].tolist())
            cell_perf.setdefault(key, []).append((v_error[i].item(), w_error[i].item()))

        cell_avg = {
            k: (sum(e[0] for e in v_w_list) / len(v_w_list),
                sum(e[1] for e in v_w_list) / len(v_w_list))
            for k, v_w_list in cell_perf.items()
        }

        newly_unlocked = {
            k for k, (v_avg, w_avg) in cell_avg.items()
            if v_avg < self.EPS_V and w_avg < self.EPS_W
        }

        expanded = self._expand_neighbors(newly_unlocked)
        self.unlocked_bins.update(expanded)

    def _expand_neighbors(self, cells: Set[Tuple[float, float]]) -> Set[Tuple[float, float]]:
        """
        Expands a set of cells to include their 8-connected neighbors.

        Args:
            cells: Cells to expand.

        Returns:
            Set: Expanded set including neighbors.
        """
        expanded = set(cells)
        for x, z in cells:
            for dx_off in [-self.dx, 0, self.dx]:
                for dz_off in [-self.dz, 0, self.dz]:
                    neighbor = (round(x + dx_off, 4), round(z + dz_off, 4))
                    if -self.max_range <= neighbor[0] <= self.max_range and -self.max_range <= neighbor[1] <= self.max_range:
                        expanded.add(neighbor)
        return expanded

    def get_range_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Computes the min and max values from unlocked bins for both lin_vel_x and ang_vel_z.

        Returns:
            Tuple: ((min_lin, max_lin), (min_ang, max_ang))
        """
        if not self.unlocked_bins:
            return (-self.start_range, self.start_range), (-self.start_range, self.start_range)
        unlocked_array = torch.tensor(list(self.unlocked_bins), device=self.device).reshape(-1, 2)
        lin_bounds = (unlocked_array[:, 0].min().item(), unlocked_array[:, 0].max().item())
        ang_bounds = (unlocked_array[:, 1].min().item(), unlocked_array[:, 1].max().item())
        return lin_bounds, ang_bounds

    def save_unlocked_bins(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.unlocked_bins, f)


# Curriculum hook for updating command ranges based on performance
def command_levels(env: ManagerBasedRLEnv, env_ids: List[int]) -> torch.Tensor:
    """
    Curriculum hook for updating velocity command ranges based on bin-wise performance.

    Args:
        env: IsaacLab RL environment.
        env_ids : Active environment indices.

    Returns:
        torch.Tensor: Monitoring tensor, here max lin_vel_x.
    """

    global grid_curriculum
    global curriculum_step_counter

    if grid_curriculum is None:
        grid_curriculum = GridCurriculumManager(
            device=env.device,
            num_envs=env.num_envs,  # ← passa explicitamente
        )

    rm = env.reward_manager

    if not isinstance(env.obs_buf, dict) or "policy" not in env.obs_buf:
        return {}

    # Collect current commands and velocities
    cmds = env.command_manager.get_command("base_velocity")[env_ids][:, [0, 2]]
    
    obs = env.obs_buf["policy"]

    v_actual = obs[env_ids, 0]   # base_lin_vel_x
    w_actual = obs[env_ids, 5]   # base_ang_vel_z

    v_cmd = obs[env_ids, 9]      # velocity_commands_x
    w_cmd = obs[env_ids, 11]     # velocity_commands_z
    
    v_error = torch.abs(v_cmd - v_actual)  
    w_error = torch.abs(w_cmd - w_actual)   

    # grid_curriculum.EPS_V = max(0.03, grid_curriculum.EPS_V0 * 0.98 ** (curriculum_step_counter/1e4))
    # grid_curriculum.EPS_W = max(0.03, grid_curriculum.EPS_W0 * 0.98 ** (curriculum_step_counter/1e4))

    # Update curriculum
    matched_bins = grid_curriculum.map_commands_to_bins(cmds)
    grid_curriculum.update_unlocked_bins(matched_bins, v_error, w_error)

    # Apply updated command bounds
    lin_range, ang_range = grid_curriculum.get_range_bounds()
    cmd_cfg = env.command_manager.cfg.base_velocity
    cmd_cfg.ranges.lin_vel_x = lin_range
    cmd_cfg.ranges.ang_vel_z = ang_range

    # env.command_manager.resample_commands()

    # Step counter & periodic saving
    curriculum_step_counter += len(env_ids)

    # Salva se atingir múltiplo
    if curriculum_step_counter % SAVE_INTERVAL < len(env_ids):
        os.makedirs(SAVE_DIR, exist_ok=True)
        step_path = os.path.join(SAVE_DIR, f"bins_step_{curriculum_step_counter}.pkl")
        grid_curriculum.save_unlocked_bins(step_path)

    return {
        "lin_vel_min": torch.tensor(lin_range[0], device=env.device).item(),
        "lin_vel_max": torch.tensor(lin_range[1], device=env.device).item(),
        "ang_vel_min": torch.tensor(ang_range[0], device=env.device).item(),
        "ang_vel_max": torch.tensor(ang_range[1], device=env.device).item(),
        "unlocked_bins": torch.tensor(len(grid_curriculum.unlocked_bins), device=env.device).item(),
        # "erro_lin_threshold": torch.tensor(grid_curriculum.EPS_V, device=env.device).item(),
        # "erro_ang_threshold": torch.tensor(grid_curriculum.EPS_W, device=env.device).item(),
    }

@configclass
class CurriculumCfg:
    command_ranges = CurrTerm(func=command_levels)



def get_term_weight(term_cfgs, term_name: str) -> float:
    """
    Extracts the weight of a reward term by its name.

    Args:
        term_cfgs: List of reward term configurations.
        term_name: Name of the reward term.

    Returns:
        float: Weight of the reward term.
    """
    for term in term_cfgs:
        if term.func.__name__ == term_name:
            return term.weight
    raise ValueError(f"Reward term '{term_name}' not found.")


def get_term_index(term_names: List[str], term_name: str) -> int:
    """
    Finds the index of a reward term name in the term list.

    Args:
        term_names: List of reward term names.
        term_name: Term name to find.

    Returns:
        int: Index of the term name.
    """
    try:
        return term_names.index(term_name)
    except ValueError:
        raise ValueError(f"Reward term '{term_name}' not found in term names.")
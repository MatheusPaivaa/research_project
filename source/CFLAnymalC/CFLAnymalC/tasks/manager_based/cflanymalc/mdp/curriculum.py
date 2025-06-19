from isaaclab.utils import configclass
from typing import List, Tuple, Set
from dataclasses import field
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import torch
import numpy as np
import sys
import pickle
import os
import random
from collections import Counter
from collections import defaultdict, deque

from isaaclab.envs import ManagerBasedRLEnv

# Variables for managing curriculum logging and persistence
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
        steps: float = 0.5,
        max_range: float = 3.0,
        start_range: float = 0.5,
        eps_v: float = 0.15,
        eps_w: float = 0.25,
        delay: bool = False,
        window_size: int = 2,
        log_error_decay: bool = False,
    ) -> None:
        
        self.device = device
        self.steps = steps
        self.max_range = max_range
        self.start_range = start_range
  
        self.EPS_V = eps_v
        self.EPS_W = eps_w
          
        self.EPS_V0 = eps_v
        self.EPS_W0 = eps_w

        self.increment_confidence = 0.5
        self.window_size = window_size
        self.win_buffers = defaultdict(lambda: deque(maxlen=self.window_size))

        self.delay = delay
        self.log_error_decay = log_error_decay

        self.cmd_map, self.lin_vals, self.ang_vals = self._build_command_grid()
        self.dx = (self.lin_vals[1] - self.lin_vals[0]).item()
        self.dz = (self.ang_vals[1] - self.ang_vals[0]).item()
        self.unlocked_bins: Set[Tuple[float, float]] = set()
        self.bin_confidence = {tuple(cmd.tolist()): 0.0 for cmd in self.cmd_map}

    def _build_command_grid(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Builds a grid of discretized [lin_vel_x, ang_vel_z] command bins.

        Returns:
            Tuple of:
                cmd_map: Flattened grid of command pairs (N, 2).
                lin_vals: Discrete values along lin_vel_x.
                ang_vals: Discrete values along ang_vel_z.

        """

        lin_vals = torch.tensor(
            [round(i, 4) for i in np.arange(-self.max_range, self.max_range + self.steps, self.steps)],
            device=self.device
        )
        ang_vals = torch.tensor(
            [round(i, 4) for i in np.arange(-self.max_range, self.max_range + self.steps, self.steps)],
            device=self.device
        )

        grid_lin, grid_ang = torch.meshgrid(lin_vals, ang_vals, indexing='ij')
        cmd_map = torch.stack([grid_lin.reshape(-1), grid_ang.reshape(-1)], dim=1)

        return cmd_map, lin_vals, ang_vals
    
    def _record_errors_window(self, key, v_err, w_err):
        """
        Stores recent tracking errors (v_err, w_err) in a sliding window buffer for a specific command bin.

        Args:
            key: Tuple representing the command bin.
            v_err: Linear velocity error.
            w_err: Angular velocity error.
        """
        self.win_buffers[key].append((v_err, w_err))

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
    
    def update_bin_confidence(self, newly_unlocked: Set[Tuple[float, float]]) -> None:
        """
        Updates confidence levels for each bin based on recent performance.

        Args:
            newly_unlocked: Bins that achieved good performance in the current step.
        """

        expanded_neighbors = self._expand_neighbors(newly_unlocked)
        for bin in expanded_neighbors:
            mapped_bin = self.map_commands_to_bins(torch.tensor([bin], device=self.device))[0]
            mapped_key = tuple(mapped_bin.tolist())
            self.bin_confidence[mapped_key] += self.increment_confidence

        for bin, score in self.bin_confidence.items():
            if score >= 1.0:
                self.unlocked_bins.add(bin)

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
            key = normalize_bin_key(tuple(matched_cmds[i].tolist()), self.steps)
            cell_perf.setdefault(key, []).append((v_error[i].item(), w_error[i].item()))

        cell_avg = {
            k: (sum(e[0] for e in v_w_list) / len(v_w_list),
                sum(e[1] for e in v_w_list) / len(v_w_list))
            for k, v_w_list in cell_perf.items()
        }

        if self.delay:
            newly_unlocked = {
                k for k, (v_avg, w_avg) in cell_avg.items()
                if v_avg < self.EPS_V and w_avg < self.EPS_W
            }
        else: 
            newly_unlocked = set()
            for cmd, ev, ew in zip(matched_cmds, v_error, w_error):
                key = normalize_bin_key(tuple(cmd.tolist()), self.steps)

                self._record_errors_window(key, ev.item(), ew.item())
                buf = self.win_buffers[key]

                if len(buf) == self.window_size:
                    v_avg = sum(e[0] for e in buf) / self.window_size
                    w_avg = sum(e[1] for e in buf) / self.window_size
                    if v_avg < self.EPS_V and w_avg < self.EPS_W:
                        newly_unlocked.add(key)

        # Activate delay
        if self.delay:
            self.update_bin_confidence(newly_unlocked)
        else:
            expanded = self._expand_neighbors(newly_unlocked)
            self.unlocked_bins.update(expanded)

    def _expand_neighbors(self, cells: Set[Tuple[float, float]]) -> Set[Tuple[float, float]]:
        """
        Expands the given set of bins to include their 8-connected neighbors in the grid.

        Args:
            cells: Set of unlocked bins (tuples) to expand.

        Returns:
            Set[Tuple[float, float]]: Expanded set including original bins and their valid neighbors.
        """
        expanded = set()
        for x, z in cells:
            for dx_off in [-self.dx, 0, self.dx]:
                for dz_off in [-self.dz, 0, self.dz]:
                    neighbor = (x + dx_off, z + dz_off)
                    if -self.max_range <= neighbor[0] <= self.max_range and -self.max_range <= neighbor[1] <= self.max_range:
                        norm = normalize_bin_key(neighbor, self.steps)
                        expanded.add(norm)
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
        """
        Saves the current set of unlocked bins to a file.

        Args:
            path: File path to save the unlocked bin data.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.unlocked_bins, f)

    def save_bin_confidences(self, path: str):
        """
        Saves the current bin confidence scores to a file.

        Args:
            path: File path to save the bin confidence data.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.bin_confidence, f)

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
            num_envs=env.num_envs,
        )

    rm = env.reward_manager

    if not isinstance(env.obs_buf, dict) or "policy" not in env.obs_buf:
        return {}

    # Collect current commands and velocities
    cmds = env.command_manager.get_command("base_velocity")[env_ids][:, [0, 2]]
    
    obs = env.obs_buf["policy"]

    v_actual = obs[env_ids, 0] # base_lin_vel_x
    w_actual = obs[env_ids, 5] # base_ang_vel_z

    v_cmd = obs[env_ids, 9] # velocity_commands_x
    w_cmd = obs[env_ids, 11] # velocity_commands_z
    
    v_error = torch.abs(v_cmd - v_actual)  
    w_error = torch.abs(w_cmd - w_actual)   

    # Activate log decay on errors
    if grid_curriculum.log_error_decay:
        grid_curriculum.EPS_V = max(0.03, grid_curriculum.EPS_V0 * 0.98 ** (curriculum_step_counter/1e4))
        grid_curriculum.EPS_W = max(0.03, grid_curriculum.EPS_W0 * 0.98 ** (curriculum_step_counter/1e4))

    # Update curriculum
    matched_bins = grid_curriculum.map_commands_to_bins(cmds)
    matched_bins = torch.tensor([
        [round_to_step(x.item(), grid_curriculum.steps), round_to_step(z.item(), grid_curriculum.steps)]
        for x, z in matched_bins
    ], device=grid_curriculum.device)
    grid_curriculum.update_unlocked_bins(matched_bins, v_error, w_error)

    # Apply updated command bounds
    lin_range, ang_range = grid_curriculum.get_range_bounds()
    cmd_cfg = env.command_manager.cfg.base_velocity
    cmd_cfg.ranges.lin_vel_x = lin_range
    cmd_cfg.ranges.ang_vel_z = ang_range

    # Step counter & periodic saving
    curriculum_step_counter += len(env_ids)

    if curriculum_step_counter % SAVE_INTERVAL < len(env_ids):
        step_path = os.path.join(SAVE_DIR, f"unlocked_bin_step_{curriculum_step_counter}.pkl")
        grid_curriculum.save_unlocked_bins(step_path)

        if grid_curriculum.delay:
            step_path = os.path.join(SAVE_DIR, f"bins_confidence_step_{curriculum_step_counter}.pkl")
            grid_curriculum.save_bin_confidences(step_path)

    result = {
        "lin_vel_min": torch.tensor(lin_range[0], device=env.device).item(),
        "lin_vel_max": torch.tensor(lin_range[1], device=env.device).item(),
        "ang_vel_min": torch.tensor(ang_range[0], device=env.device).item(),
        "ang_vel_max": torch.tensor(ang_range[1], device=env.device).item(),
        "unlocked_bins": torch.tensor(len(grid_curriculum.unlocked_bins), device=env.device).item(),
    }

    if grid_curriculum.log_error_decay:
        result["erro_lin_threshold"] = torch.tensor(grid_curriculum.EPS_V, device=env.device).item()
        result["erro_ang_threshold"] = torch.tensor(grid_curriculum.EPS_W, device=env.device).item()

    return result

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


def round_to_step(val, step):
    """
    
    Rounds a value to the nearest multiple of the given step.

    Args:
        val: The value to be rounded.
        step: The step size used for rounding.

    Returns:
        float: The value rounded to the nearest step, with 4 decimal precision.
    
    """
    return round(round(val / step) * step, 4)


def normalize_bin_key(bin_tuple: Tuple[float, float], step: float) -> Tuple[float, float]:
    """

    Normalizes a tuple of float values to the nearest multiples of a given step.

    Args:
        bin_tuple: A tuple containing two float values.
        step: The step size used for normalization.

    Returns:
        Tuple[float, float]: A tuple with both values normalized and rounded to 4 decimal places.
    
    """
    return (
        round(round(bin_tuple[0] / step) * step, 4),
        round(round(bin_tuple[1] / step) * step, 4),
    )

"""
Terrain tracking and logging module for environment performance analysis.

Purpose:
    This experimental function collects terrain-specific statistics during training,
    such as velocity tracking errors and success rates, and logs them to TensorBoard.

Status:
    UNDER DEVELOPMENT – subject to change and testing.

Main features:
    - Computes MAE for linear and angular velocity commands per terrain type.
    - Tracks success rates (timeouts vs terminations) per terrain.
    - Logs stats to TensorBoard for visualization over time.

Note:
    Requires a ManagerBasedRLEnv with terrain and policy observations enabled.
"""

import os, collections, omni.log as log
import torch, numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains.terrain_generator import TerrainGenerator
import inspect
from types import ModuleType, FunctionType, MethodType

def active_terrain_names(env):
    """
    Returns the name of the active terrain for each environment instance.

    Args:
        env (ManagerBasedRLEnv): The simulation environment.

    Returns:
        List[str]: List of terrain names mapped from levels and types.
    """
    if not hasattr(env, "_terrain_name_lut"):
        tg = TerrainGenerator(
            cfg=env.scene.terrain.cfg.terrain_generator,
            device="cpu"
        )
        env._terrain_name_lut = tg.sub_name 

    levels = env.scene.terrain.terrain_levels.cpu().numpy()
    types  = env.scene.terrain.terrain_types.cpu().numpy()
    lut    = env._terrain_name_lut
    return [lut[l, t] for l, t in zip(levels, types)]

def terrain_stats(env, env_ids, window=500, tb_root="terrain"):
    """
    Collects and logs terrain-wise tracking errors and success rates.

    Args:
        env (ManagerBasedRLEnv): The simulation environment.
        env_ids (List[int]): List of environment indices to evaluate.
        window (int): Smoothing window size for moving averages.
        tb_root (str): TensorBoard log root group name.
    """
    run_dir = Path(getattr(env, "run_dir", ".")) 
    log_dir = run_dir / "logs" / "tracking"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logging cache if not already present
    if not hasattr(env, "TERRAIN_ERR_CACHE"):
        terr_names = active_terrain_names(env)
        uniq = sorted(set(terr_names))
        env.TERRAIN_ERR_CACHE = {
            "step":   0,
            "writer": SummaryWriter(log_dir=str(log_dir)),
            "lin":  {t: collections.deque(maxlen=window) for t in uniq},
            "ang":  {t: collections.deque(maxlen=window) for t in uniq},
            "succ": collections.Counter(),
            "trials": collections.Counter(),
        }

    C = env.TERRAIN_ERR_CACHE
    step, writer = C["step"], C["writer"]

    if not isinstance(env.obs_buf, dict) or "policy" not in env.obs_buf:
        return {}

    # Extract observation data
    obs = env.obs_buf["policy"]
    v_act, w_act = obs[env_ids, 0], obs[env_ids, 5]
    v_cmd, w_cmd = obs[env_ids, 9], obs[env_ids, 11]
    v_err, w_err = torch.abs(v_cmd - v_act), torch.abs(w_cmd - w_act)

    robot = env.scene["robot"]
    torques = robot.data.applied_torque[env_ids]
    qd = robot.data.joint_vel[env_ids]
    # power = torch.sum(torch.abs(torques * qd), dim=1)

    # mass, g = robot.data.mass, 9.81
    lin_vel = obs[env_ids, 0]
    # cot = power / ((mass * g) * torch.clamp(torch.abs(lin_vel), min=1e-3))

    terr_names = active_terrain_names(env)
    for li, eid in enumerate(env_ids):
        terr = terr_names[eid]

        C["lin"][terr].append(v_err[li].item())
        C["ang"][terr].append(w_err[li].item())
        # C["cot"][terr].append(cot[li].item())

        if env.termination_manager.terminated[eid]:
            C["trials"][terr] += 1
            if env.termination_manager.time_outs[eid]:
                C["succ"][terr] += 1

    # Log stats to TensorBoard
    for terr in C["lin"]:
        if C["lin"][terr]:
            writer.add_scalar(f"{tb_root}/{terr}/lin_err_mae",
                              np.mean(C["lin"][terr]), step)
            writer.add_scalar(f"{tb_root}/{terr}/ang_err_mae",
                              np.mean(C["ang"][terr]), step)
            # writer.add_scalar(f"{tb_root}/{terr}/cot",
            #                   np.mean(C["cot"][terr]), step)
            succ = C["succ"][terr] / max(C["trials"][terr], 1)
            writer.add_scalar(f"{tb_root}/{terr}/success_rate",
                              succ, step)

    C["step"] += 1

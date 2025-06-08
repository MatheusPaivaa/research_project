import os, torch, collections, omni.log as log
from torch.utils.tensorboard import SummaryWriter
from isaaclab.managers import EventTermCfg
from isaaclab.envs import ManagerBasedRLEnv

def _discover_terrain_ids(env: ManagerBasedRLEnv):
    """
    
    Returns:
        terrain_name_of_env : list[str] length == env.num_envs

    """
    tgen = env.scene.terrain.terrain_generator

    if not hasattr(tgen, "sub_terrains") or not tgen.sub_terrains:
        return ["default"] * env.num_envs

    names = list(tgen.sub_terrains.keys())
    num_rows, num_cols = tgen.num_rows, tgen.num_cols

    # Row/col is encoded in env.scene.terrain.env_origins
    env_origins = env.scene.terrain.env_origins   # (num_envs, 3)
    x_coords = env_origins[:, 0] # use X & Y to recover row/col
    y_coords = env_origins[:, 1]

    patch_w = (tgen.size[0] / num_rows)
    patch_h = (tgen.size[1] / num_cols)

    rows = torch.floor((x_coords - x_coords.min()) / patch_w).to(torch.long)
    cols = torch.floor((y_coords - y_coords.min()) / patch_h).to(torch.long)
    idx  = (rows * num_cols + cols) % len(names)  

    return [names[i] for i in idx]

def terrain_stats(
    env: ManagerBasedRLEnv,
    env_ids: list[int],
    window: int        = 500,
    print_every: int   = 10_000,
    tb_tag_root: str   = "terrain",
):
    """

    Called every sim-step by the EventManager (mode='step').
    It accumulates stats in env.TERRAIN_STAT_CACHE and writes them out.

    """
    if not hasattr(env, "TERRAIN_STAT_CACHE"):
        env.TERRAIN_STAT_CACHE = {
            "terrain_name": _discover_terrain_ids(env),
            "step": 0,
            "writer": SummaryWriter(os.path.join(os.getenv("OUTPUT_DIR", "./tb"), "terrain")),
            "buf": collections.defaultdict(
                lambda: collections.deque(maxlen=window)
            ),
            "err_lin": collections.defaultdict(lambda: collections.deque(maxlen=window)),
            "err_ang": collections.defaultdict(lambda: collections.deque(maxlen=window)),
            "ep_len":  collections.defaultdict(lambda: collections.deque(maxlen=window)),
        }

    C = env.TERRAIN_STAT_CACHE
    step = C["step"]
    names = C["terrain_name"]

    rew_buf  = env.reward_manager.rew_buf
    done_buf = env.termination_manager.terminated

    obs      = env.obs_buf["policy"]
    v_err    = torch.abs(obs[env_ids, 9]  - obs[env_ids, 0])  # cmd_x  – vel_x
    w_err    = torch.abs(obs[env_ids, 11] - obs[env_ids, 5])  # cmd_z  – vel_z

    for i, env_id in enumerate(env_ids):
        tname = names[env_id]
        C["buf"][tname].append(rew_buf[env_id].item())
        C["err_lin"][tname].append(v_err[i].item())
        C["err_ang"][tname].append(w_err[i].item())
        if done_buf[env_id]:
            C["ep_len"][tname].append(env.episode_length_buf[env_id])

    C["step"] += len(env_ids)

    if C["step"] % print_every < len(env_ids):
        hdr = "│ {:>10} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │".format(
            "terrain", "R̄", "v_err", "w_err", "ep_len"
        )
        log.info("-" * len(hdr))
        log.info(hdr)
        for tn in C["buf"]:
            mean_r  = torch.tensor(C["buf"][tn]).mean().item()
            mean_v  = torch.tensor(C["err_lin"][tn]).mean().item()
            mean_w  = torch.tensor(C["err_ang"][tn]).mean().item()
            mean_ep = (torch.tensor(C["ep_len"][tn]).float().mean().item()
                       if C["ep_len"][tn] else 0.0)
            log.info("│ {:>10} │ {:7.3f} │ {:7.3f} │ {:7.3f} │ {:7.1f} │"
                     .format(tn, mean_r, mean_v, mean_w, mean_ep))
        log.info("-" * len(hdr))

    w = C["writer"]
    for tn in C["buf"]:
        w.add_scalar(f"{tb_tag_root}/{tn}/reward_mean",
                     torch.tensor(C["buf"][tn]).mean().item(), global_step=step)
        w.add_scalar(f"{tb_tag_root}/{tn}/lin_err",
                     torch.tensor(C["err_lin"][tn]).mean().item(), global_step=step)
        w.add_scalar(f"{tb_tag_root}/{tn}/ang_err",
                     torch.tensor(C["err_ang"][tn]).mean().item(), global_step=step)

    env.extras.update({
        f"{tb_tag_root}/{tn}/reward": torch.tensor(C["buf"][tn]).mean()
        for tn in C["buf"]
    })

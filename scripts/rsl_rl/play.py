# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import collections
from itertools import product
import math

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--unique", action="store_true", default=False, help="Enable unique env for testing")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--trained_terrain", type=str, default="flat", help="Name of the trained terrain.")
parser.add_argument("--tested_terrain", type=str, default="flat", help="Name of the tested terrain.")
parser.add_argument("--log_name", type=str, default=None, help="Name of the log name.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np
import json

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
import isaaclab.sim as sim_utils
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.devices import Se2Keyboard
from isaaclab.managers import ObservationTermCfg as ObsTerm

# PLACEHOLDER: Extension template (do not remove this comment)

from CFLAnymalC.tasks.manager_based.cflanymalc.mdp.command import FixedVelocityCommandCfg
import rsl_rl_utils

import CFLAnymalC.tasks
from terrain_utils import apply_overrides_play

def main():
    """Play with RSL-RL agent."""

    if args_cli.unique or args_cli.keyboard: 
        args_cli.num_envs = 1

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    if not args_cli.keyboard:
        env_cfg.commands.base_velocity = FixedVelocityCommandCfg(
            asset_name="robot",
            default_command=[0.0, 0.0, 0.0]
        )

    if args_cli.keyboard:
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        controller = Se2Keyboard(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    apply_overrides_play(env_cfg, args_cli)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if not hasattr(env.unwrapped, "_commands"):
        env.unwrapped._commands = torch.zeros((args_cli.num_envs, 3), device=args_cli.device)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join("./logs/"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # Export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()

    # Evalaluation configuration
    trials, trial_steps, warmup_steps = 5, 900, 50
    results = []
    buffer = []

    # Evaluation for each pair
    v_x_vals = np.linspace(-4, 4, 30)  # Antes
    omega_z_vals = np.linspace(-4, 4, 30)  # Antes

    combinations = list(product(v_x_vals, omega_z_vals))
    chunks = np.array_split(combinations, args_cli.num_envs)
    env_command_queues = {i: list(chunk) for i, chunk in enumerate(chunks)}
    max_steps = max(len(queue) for queue in env_command_queues.values())

    total_start_time = time.time()

    for step_index in range(max_steps):
        commands = []
        for i in range(args_cli.num_envs):
            if step_index < len(env_command_queues[i]):
                v_x_cmd, omega_z_cmd = env_command_queues[i][step_index]
            else:
                v_x_cmd, omega_z_cmd = 0.0, 0.0

            if args_cli.unique: commands.append([3.0, 0.0, 0.0])
            else: commands.append([v_x_cmd, 0.0, omega_z_cmd])

        command_tensor = torch.tensor(commands, device=env.device)

        print(f"\n[INFO] Evaluating step {step_index + 1}/{len(env_command_queues[0])}")
        for i, cmd in enumerate(commands):
            print(f"  Env {i}: v_x={cmd[0]:.2f}, omega_z={cmd[2]:.2f}")

        for trial in range(trials):
            buffer = []
            print(f"[INFO]  Trial {trial + 1}/{trials}")
            obs, _ = env.get_observations()
            timestep = 0

            while timestep < trial_steps and simulation_app.is_running():
                start_time = time.time()
                with torch.inference_mode():
                    if trial == 0 and timestep == 0: env.reset()

                    if not args_cli.keyboard:
                        command_term = env.env.unwrapped.command_manager.get_term("base_velocity")
                        command_term.update_commands(command_tensor)

                    actions = policy(obs.float())
                    obs, _, _, _ = env.step(actions)

                    for i in range(args_cli.num_envs):

                        v_x = obs[i, 0].item()
                        omega_z = obs[i, 5].item()

                        target_v_x = obs[i, 9].item()
                        target_omega_z = obs[i, 11].item()

                        error_x = (v_x - target_v_x) ** 2
                        error_omega_z = (omega_z - target_omega_z) ** 2

                        if timestep >= warmup_steps:
                            buffer.append({
                                "env": i,
                                "v_x_cmd": v_x,
                                "omega_z_cmd": omega_z,
                                "error_x": error_x,
                                "error_omega_z": error_omega_z,
                            })

                    timestep += 1
                    if timestep == trial_steps:
                        obs, _ = env.reset()

                    # Delay for real-time execution (optional)
                    sleep_time = dt - (time.time() - start_time)
                    if args_cli.real_time and sleep_time > 0:
                        time.sleep(sleep_time)

                    if args_cli.keyboard:
                        rsl_rl_utils.camera_follow(env)
                        current_cmd = controller.advance()
                        print(f"[DEBUG] Teclado: v_x={current_cmd[0]:.2f}, omega_z={current_cmd[2]:.2f}")

            results_per_env = collections.defaultdict(lambda: {
                "sum_error_x": 0.0,
                "sum_error_omega_z": 0.0,
                "count": 0
            })

            for entry in buffer:
                env_idx = entry["env"]
                results_per_env[env_idx]["sum_error_x"]        += entry["error_x"]
                results_per_env[env_idx]["sum_error_omega_z"] += entry["error_omega_z"]
                results_per_env[env_idx]["count"]             += 1

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break

        for env_id, data in results_per_env.items():             
            count              = data["count"]
            avg_error_x        = data["sum_error_x"]        / count
            avg_error_omega_z  = data["sum_error_omega_z"]  / count
            mse_x              = data["sum_error_x"] / count
            mse_omega_z        = data["sum_error_omega_z"] / count

            rmse_x             = mse_x ** 0.5
            rmse_omega_z       = mse_omega_z ** 0.5

            v_x_cmd, _, omega_z_cmd = commands[env_id]   

            print(f"[RESULT][ENV {env_id}] => v_x={v_x_cmd:.2f}, "
                f"omega_z={omega_z_cmd:.2f} | "
                f"error_x={avg_error_x:.4f}, "
                f"error_omega_z={avg_error_omega_z:.4f}, "
                f"count={count}")

            results.append({                                      
                "v_x_cmd": v_x_cmd,
                "omega_z_cmd": omega_z_cmd,
                "avg_error_x": avg_error_x,
                "avg_error_omega_z": avg_error_omega_z,
                "rmse_x": rmse_x,
                "rmse_omega_z": rmse_omega_z,
                "count": count
            })

    results.sort(key=lambda x: (x["v_x_cmd"], x["omega_z_cmd"]))

    os.makedirs("./logs/tracking_log", exist_ok=True)

    if args_cli.log_name is not None:
        args_cli.tested_terrain = args_cli.log_name
        
    output_path = os.path.join(f"./logs/tracking_log/average_error_log_{args_cli.trained_terrain}_in_{args_cli.tested_terrain}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n[INFO] Saved path: {output_path}")

    total_elapsed_time = time.time() - total_start_time
    print(f"\n[INFO] Complete evaluation in {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes).")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
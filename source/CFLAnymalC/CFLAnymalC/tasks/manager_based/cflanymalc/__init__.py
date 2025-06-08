# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="CFL-AnymalC_Flat_Task",
    entry_point=f"isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker = True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cflanymalc_env_cfg:AnymalCFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
    },
)

gym.register(
    id="CFL-AnymalC_Rough_Task",
    entry_point=f"isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker = True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cflanymalc_env_cfg:AnymalCRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

gym.register(
    id="CFL-AnymalC_Rough_LSTM_Task",
    entry_point=f"isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker = True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cflanymalc_env_cfg:AnymalCRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPOLSTMRunnerCfg",
    },
)

gym.register(
    id="CFL-AnymalC_Play_Flat_Task",
    entry_point=f"isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker = True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cflanymalc_env_cfg:AnymalCPlayFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
    },
)

gym.register(
    id="CFL-AnymalC_Play_Rough_Task",
    entry_point=f"isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker = True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cflanymalc_env_cfg:AnymalCPlayRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
    },
)

gym.register(
    id="CFL-AnymalC_Play_Rough_LSTM_Task",
    entry_point=f"isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker = True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cflanymalc_env_cfg:AnymalCPlayRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPOLSTMRunnerCfg",
    },
)

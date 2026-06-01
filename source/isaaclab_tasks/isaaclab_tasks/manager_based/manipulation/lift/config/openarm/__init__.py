# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control
##

# ── 完整 Lift 任務（階段二：Grasping + Lifting，使用 DDPG）──
gym.register(
    id="Isaac-Lift-Cube-OpenArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OpenArmCubeLiftEnvCfg",
        # 舊有的 PPO 設定（保留相容性，可改為使用 rsl_rl 或 rl_games 搭配 PPO）
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmLiftCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # 新增：SKRL DDPG 設定（階段二主要使用）
        "skrl_ddpg_cfg_entry_point": f"{agents.__name__}:skrl_ddpg_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-OpenArm-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OpenArmCubeLiftEnvCfg_PLAY",
        # 舊有的 PPO 設定（保留相容性）
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmLiftCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # 新增：SKRL DDPG 設定
        "skrl_ddpg_cfg_entry_point": f"{agents.__name__}:skrl_ddpg_cfg.yaml",
    },
    disable_env_checker=True,
)

# ── Reach 任務（階段一：只學習接近物體，使用 DDPG）──
gym.register(
    id="Isaac-Reach-Cube-OpenArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OpenArmCubeReachEnvCfg",
        # SKRL DDPG 設定（階段一主要使用）
        "skrl_ddpg_cfg_entry_point": f"{agents.__name__}:skrl_ddpg_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Reach-Cube-OpenArm-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OpenArmCubeReachEnvCfg_PLAY",
        # SKRL DDPG 設定
        "skrl_ddpg_cfg_entry_point": f"{agents.__name__}:skrl_ddpg_cfg.yaml",
    },
    disable_env_checker=True,
)

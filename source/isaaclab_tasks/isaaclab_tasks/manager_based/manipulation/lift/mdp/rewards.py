# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    offset: tuple = (0.0, 0.0, 0.0),  # 相對於物件重心的偏移，例如 (0,0,0.06) 指向杯口
    ee_index: int = 0,  # 0=左手, 1=右手
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position with optional offset: (num_envs, 3)
    offset_t = torch.tensor(offset, device=env.device).unsqueeze(0)
    cube_pos_w = object.data.root_pos_w + offset_t
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., ee_index, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_ee_distance_bimanual(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward whichever hand (left or right) is closest to the object.
    Each hand gets its own reward contribution, encouraging both arms to learn to reach.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = object.data.root_pos_w  # (num_envs, 3)

    # Left hand (index 0) distance
    ee_left = ee_frame.data.target_pos_w[..., 0, :]
    dist_left = torch.norm(cube_pos_w - ee_left, dim=1)

    # Right hand (index 1) distance
    ee_right = ee_frame.data.target_pos_w[..., 1, :]
    dist_right = torch.norm(cube_pos_w - ee_right, dim=1)

    # Reward both hands: sum of tanh rewards so both arms are incentivized
    reward_left = 1 - torch.tanh(dist_left / std)
    reward_right = 1 - torch.tanh(dist_right / std)

    return reward_left + reward_right


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_lifting_verticality(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object vertically without horizontal movement.
    It penalizes the X-Y displacement from the object's default/initial position.
    """
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get current object position: (num_envs, 3)
    curr_pos_w = object.data.root_pos_w
    
    # Get initial/default object position: (num_envs, 3)
    # default_root_state is (num_envs, 13) where first 3 are pos x,y,z
    init_pos_w = object.data.default_root_state[:, :3]
    
    # Calculate XY displacement only
    xy_displacement = torch.norm(curr_pos_w[:, :2] - init_pos_w[:, :2], dim=1)
    
    # Reward is given only if the object is lifted
    is_lifted = object.data.root_pos_w[:, 2] > minimal_height
    
    # High reward for small XY displacement, drops off with distance
    return is_lifted * (1.0 - torch.tanh(xy_displacement / std))


def object_upright_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize the object for tilting from the upright (Z) axis."""
    object: RigidObject = env.scene[object_cfg.name]
    # quaternion is [w, x, y, z]
    w = object.data.root_quat_w[:, 0]
    x = object.data.root_quat_w[:, 1]
    y = object.data.root_quat_w[:, 2]
    z = object.data.root_quat_w[:, 3]
    # The true vertical direction is [0, 0, 1]. Rotated by quat, the Z-component is:
    up_z = w*w - x*x - y*y + z*z
    # Penalize deviation from 1.0 (perfectly upright)
    tilt = 1.0 - up_z
    # 給予 0.05 的容錯區間（大約 18 度），超過這個傾斜角度才開始計算懲罰
    return torch.relu(tilt - 0.05)


def object_safe_trajectory(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    safe_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward moving straight up to a safe height, then translating to the goal.
    This prevents diagonal movements that might spill liquids.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # compute the desired goal position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    
    curr_pos_w = object.data.root_pos_w
    # init_pos_w comes from default state
    init_pos_w = object.data.default_root_state[:, :3]
    
    is_below_safe_height = curr_pos_w[:, 2] < safe_height
    
    # Target position depends on current height
    target_pos_w = torch.zeros_like(curr_pos_w)
    
    # If below safe height: XY is locked to initial XY, Z target is safe_height
    # If above safe height: XY is the goal XY, Z target is the goal Z
    target_pos_w[:, :2] = torch.where(is_below_safe_height.unsqueeze(1), init_pos_w[:, :2], des_pos_w[:, :2])
    target_pos_w[:, 2] = torch.where(is_below_safe_height, safe_height, des_pos_w[:, 2])
    
    distance = torch.norm(target_pos_w - curr_pos_w, dim=1)
    
    # rewarded heavily if close to this customized trajectory ONLY WHEN LIFTED
    is_lifted = curr_pos_w[:, 2] > minimal_height
    return is_lifted * (1.0 - torch.tanh(distance / std))


def object_velocity_penalty(
    env: ManagerBasedRLEnv,
    max_velocity: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize the object for moving too fast (e.g., being knocked away).

    This discourages the robot from hitting/flinging the object instead of grasping it.
    The penalty is proportional to the object's linear velocity when it exceeds max_velocity.
    """
    object: RigidObject = env.scene[object_cfg.name]
    # Linear velocity magnitude: (num_envs,)
    obj_vel = torch.norm(object.data.root_lin_vel_w, dim=1)
    # Only penalize when velocity exceeds the threshold
    return torch.relu(obj_vel - max_velocity)


def object_grasping(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    finger_trigger_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for closing the gripper when close to the object."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    obj_w = object.data.root_pos_w
    distance = torch.norm(obj_w - ee_w, dim=1)
    
    is_close = distance <= distance_threshold
    
    # finger joints are typically the last joints 6-11 in Kinova 6dof (6-11 mapping to 3 fingers, 3 tips)
    # We take the mean of the gripper joints. 
    finger_pos = torch.mean(robot.data.joint_pos[:, 6:], dim=1)
    is_grasping = finger_pos > finger_trigger_pos
    
    # Give reward of 1.0 if it is both close to the object and squeezing the fingers!
    return torch.where(is_close & is_grasping, torch.ones_like(distance), torch.zeros_like(distance))


def openarm_object_grasping(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    finger_trigger_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """OpenArm 雙臂版夾取獎勵。

    與通用版 object_grasping 不同的地方：
    - 動態過濾名稱含 'finger' 的關節，避免將手臂關節誤算進夾取偵測。
    - OpenArm 夾爪「閉合」時關節位置接近 0.0，
      因此條件改為 finger_pos < finger_trigger_pos（值越小代表夾得越緊）。
    - 同時偵測左手（index 0）與右手（index 1）到物體的距離，
      只要任一隻手靠近且夾爪閉合，就給予獎勵。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    obj_w = object.data.root_pos_w  # (num_envs, 3)

    # 分別計算左手（index 0）與右手（index 1）到物體的距離
    ee_left = ee_frame.data.target_pos_w[..., 0, :]
    ee_right = ee_frame.data.target_pos_w[..., 1, :]
    dist_left = torch.norm(obj_w - ee_left, dim=1)
    dist_right = torch.norm(obj_w - ee_right, dim=1)

    # 任一手靠近就算成立
    is_close = (dist_left <= distance_threshold) | (dist_right <= distance_threshold)

    # 動態找出所有 finger 關節的索引（避免硬編碼）
    finger_indices = [
        i for i, name in enumerate(robot.data.joint_names)
        if "finger" in name.lower()
    ]

    if len(finger_indices) == 0:
        # 若找不到 finger 關節，退回預設行為（不給獎勵）
        return torch.zeros(env.num_envs, device=env.device)

    # OpenArm 夾爪閉合時關節位置 ≈ 0.0，因此用 < 判斷
    finger_pos = torch.mean(robot.data.joint_pos[:, finger_indices], dim=1)
    is_grasping = finger_pos < finger_trigger_pos

    # 同時靠近且夾爪閉合才給 1.0 獎勵
    return torch.where(is_close & is_grasping, torch.ones(env.num_envs, device=env.device), torch.zeros(env.num_envs, device=env.device))


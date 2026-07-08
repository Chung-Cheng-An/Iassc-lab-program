# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
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
    z_offset: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel.

    Args:
        env: The environment.
        std: Standard deviation for the tanh kernel.
        z_offset: Vertical offset (m) added to the object root position.
            Use this to aim at a point above the cup base (e.g. 0.06 for
            3/5 of a ~10 cm tall cup).
    """
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    target_pos = object.data.root_pos_w.clone()
    target_pos[:, 2] += z_offset  # shift target up along z-axis
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(target_pos - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


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


def object_tilt_penalty(
    env: ManagerBasedRLEnv,
    tilt_angle_limit: float = 45.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize the object for tilting beyond a certain angle (in degrees)."""
    import math
    object: RigidObject = env.scene[object_cfg.name]
    # quaternion is [w, x, y, z]
    # Local Z-axis projected onto world Z-axis is: 1.0 - 2.0 * (x^2 + y^2)
    x = object.data.root_quat_w[:, 1]
    y = object.data.root_quat_w[:, 2]
    z_proj = 1.0 - 2.0 * (x * x + y * y)
    
    threshold_proj = math.cos(math.radians(tilt_angle_limit))
    
    # Return penalty only if the tilt is greater than the limit (i.e. z_proj < threshold_proj)
    return torch.clamp(threshold_proj - z_proj, min=0.0)


def gripper_closed_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_finger_joint.*"]),
    open_val: float = 0.044,
) -> torch.Tensor:
    """Reward the agent for closing the gripper (proxy for grasping).

    Closure ratio: 1.0 when fully closed, 0.0 when fully open.
    Typically used in the grasp stage to encourage the agent to squeeze the cup.

    Args:
        env: The environment.
        robot_cfg: Configuration for the robot and its finger joints.
        open_val: The joint position value corresponding to a fully open gripper.
    """
    robot = env.scene[robot_cfg.name]
    finger_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    # 0 = fully closed, open_val = fully open → invert to get closure
    closure = 1.0 - torch.clamp(finger_pos.mean(dim=-1) / open_val, 0.0, 1.0)
    return closure


def object_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.4,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent proportionally to how high the object has been lifted.

    Reward is clamped to [0, 1]. Used in the lift stage.

    Args:
        env: The environment.
        target_height: The world-frame z-height considered as "fully lifted" (reward = 1.0).
        object_cfg: Configuration for the object asset.
    """
    object: RigidObject = env.scene[object_cfg.name]
    height = object.data.root_pos_w[:, 2]
    return torch.clamp(height / target_height, 0.0, 1.0)


def object_upright_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for keeping the object upright.

    Returns a value in [0, 1] based on how aligned the object's local Z-axis
    is with the world Z-axis (1.0 = perfectly upright, 0.0 = horizontal or inverted).

    Args:
        env: The environment.
        object_cfg: Configuration for the object asset.
    """
    object: RigidObject = env.scene[object_cfg.name]
    # quaternion is [w, x, y, z]; projection of local-Z onto world-Z = 1 - 2*(x²+y²)
    x = object.data.root_quat_w[:, 1]
    y = object.data.root_quat_w[:, 2]
    z_proj = 1.0 - 2.0 * (x * x + y * y)
    return torch.clamp(z_proj, 0.0, 1.0)


def gripper_closed_near_object_reward(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    z_offset: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_finger_joint.*"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    open_val: float = 0.044,
) -> torch.Tensor:
    """Reward the agent for closing the gripper, but only when it is close to the object.

    Prevents reward hacking (closing in the air). Uses exponential decay gating based on distance.

    Args:
        env: The environment.
        threshold: The distance scale (m) within which the closure reward is active.
        z_offset: Vertical offset (m) added to the object root position to check distance.
        robot_cfg: Configuration for the robot.
        object_cfg: Configuration for the target object.
        ee_frame_cfg: Configuration for the end-effector frame.
        open_val: Joint position for fully open gripper.
    """
    # 1. Gripper closure (0.0 = open, 1.0 = closed)
    robot = env.scene[robot_cfg.name]
    finger_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    closure = 1.0 - torch.clamp(finger_pos.mean(dim=-1) / open_val, 0.0, 1.0)

    # 2. EE to object target position distance
    object: RigidObject = env.scene[object_cfg.name]
    from isaaclab.sensors import FrameTransformer
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    target_pos = object.data.root_pos_w.clone()
    target_pos[:, 2] += z_offset
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(target_pos - ee_w, dim=1)

    # 3. Gating based on distance (gating = 1.0 when distance = 0, decays to 0.13 at distance = 2*threshold)
    gating = torch.exp(-distance / threshold)

    return closure * gating


def gripper_premature_close_penalty(
    env: ManagerBasedRLEnv,
    penalty_dist: float = 0.08,
    z_offset: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_finger_joint.*"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    open_val: float = 0.044,
) -> torch.Tensor:
    """Penalise closing the gripper while still far from the cup.

    Returns a positive value (used with a negative weight) proportional to the
    gripper closure when the EE is more than ``penalty_dist`` away from the cup.
    This eliminates the reward-hacking behaviour of closing in the air before
    approaching the object.

    Args:
        env: The environment.
        penalty_dist: Distance (m) beyond which closing the gripper is penalised.
        z_offset: Vertical offset (m) added to the object root position to check distance.
        robot_cfg: Configuration for the robot finger joints.
        object_cfg: Configuration for the target object.
        ee_frame_cfg: Configuration for the end-effector frame.
        open_val: Joint position for fully open gripper (used to compute closure ratio).
    """
    # Gripper closure  (0 = open, 1 = closed)
    robot = env.scene[robot_cfg.name]
    finger_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    closure = 1.0 - torch.clamp(finger_pos.mean(dim=-1) / open_val, 0.0, 1.0)

    # EE-to-object distance
    object: RigidObject = env.scene[object_cfg.name]
    from isaaclab.sensors import FrameTransformer
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    target_pos = object.data.root_pos_w.clone()
    target_pos[:, 2] += z_offset
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(target_pos - ee_w, dim=1)

    # Hard mask: 1.0 when farther than penalty_dist, 0.0 otherwise
    far_mask = (distance > penalty_dist).float()

    # Penalty = closure * far_mask  (large when closed AND far, 0 when open OR near)
    return closure * far_mask


def ee_align_cup_grasp_flat(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for keeping the end-effector aligned horizontally to grab the cup.

    Encourages:
    1. The tool direction (local Z-axis) to point forward along world +X axis.
    2. The gripper width/clamping direction (local Y-axis) to remain completely horizontal (world Z component = 0).
    """
    from isaaclab.sensors import FrameTransformer
    from isaaclab.utils.math import quat_apply
    
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    
    # 1. Get tool direction (local Z) in world frame
    z_axis_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    z_axis_w = quat_apply(ee_quat_w, z_axis_local)
    
    # We want local Z to align with world +X [1, 0, 0]
    # The dot product of z_axis_w and [1, 0, 0] is simply z_axis_w[:, 0]
    z_align = torch.clamp(z_axis_w[:, 0], min=0.0) # only positive alignment
    
    # 2. Get gripper width direction (local Y) in world frame
    y_axis_local = torch.tensor([0.0, 1.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    y_axis_w = quat_apply(ee_quat_w, y_axis_local)
    
    # We want local Y to be strictly horizontal (world Z component = 0)
    y_align_penalty = torch.square(y_axis_w[:, 2]) # 0 when horizontal, 1 when vertical
    
    # Combine rewards: encourage z_align (1.0 is best) and penalise non-horizontal y (0.0 is best)
    # Map to [0, 1] range
    reward = z_align * (1.0 - y_align_penalty)
    return reward


def object_lift_height_reward(
    env: ManagerBasedRLEnv,
    table_height: float = 0.055,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent continuously for lifting the object above the table.

    Returns the height difference above table_height, clamped to positive values.
    """
    object = env.scene[object_cfg.name]
    height = object.data.root_pos_w[:, 2]
    # Continuous height difference
    lift_height = torch.clamp(height - table_height, min=0.0)
    return lift_height




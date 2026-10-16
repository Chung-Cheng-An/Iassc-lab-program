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
    approach_offset: float = 0.0,
    approach_yaw: float = 45.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel.

    Args:
        env: The environment.
        std: Standard deviation for the tanh kernel.
        z_offset: Vertical offset (m) added to the object root position.
        approach_offset: Distance (m) to offset backward along the approach angle to prevent palm collisions.
        approach_yaw: Approach yaw angle in degrees (default 45 deg).
    """
    import math
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    target_pos = object.data.root_pos_w.clone()
    target_pos[:, 2] += z_offset  # shift target up along z-axis
    if approach_offset > 0.0:
        yaw_rad = math.radians(approach_yaw)
        target_pos[:, 0] -= approach_offset * math.cos(yaw_rad)
        target_pos[:, 1] -= approach_offset * math.sin(yaw_rad)
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(target_pos - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def ee_soft_landing_near_object(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.12,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the robot for decelerating when close to the cup (< 12cm). Prevents impact toppling."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    from isaaclab.assets import Articulation
    robot: Articulation = env.scene[robot_cfg.name]

    target_pos = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(target_pos - ee_w, dim=1)

    joint_vel = torch.norm(robot.data.joint_vel[:, :7], dim=1)
    is_near = distance < distance_threshold
    soft_landing = torch.where(is_near, 1.0 / (1.0 + 2.0 * joint_vel), torch.zeros_like(distance))
    return soft_landing


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
    approach_yaw: float = 0.0,
    tool_axis: str = "z",
    clamping_axis: str = "y",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for keeping the end-effector aligned horizontally to grab the cup.

    Encourages:
    1. The tool direction (local Z-axis or X-axis) to point along the approach direction in the world XY plane.
    2. The gripper width/clamping direction (local Y-axis or X-axis) to remain completely horizontal (world Z component = 0).
    """
    from isaaclab.sensors import FrameTransformer
    from isaaclab.utils.math import quat_apply
    import math
    
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    
    # 1. Get tool direction in world frame based on the robot type's local tool axis
    if tool_axis == "x":
        tool_axis_local = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    elif tool_axis == "y":
        tool_axis_local = torch.tensor([0.0, 1.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    else:
        tool_axis_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    z_axis_w = quat_apply(ee_quat_w, tool_axis_local)
    
    # Target alignment vector in world XY plane based on approach_yaw (in degrees)
    yaw_rad = math.radians(approach_yaw)
    target_dir = torch.tensor([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], device=env.device)
    
    # Dot product of local tool direction (z_axis_w) and target_dir (clamped to positive alignment)
    z_align = torch.clamp(torch.sum(z_axis_w * target_dir, dim=-1), min=0.0)
    
    # 2. Get gripper width direction in world frame based on clamping axis
    if clamping_axis == "x":
        y_axis_local = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    elif clamping_axis == "y":
        y_axis_local = torch.tensor([0.0, 1.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    else:
        y_axis_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    y_axis_w = quat_apply(ee_quat_w, y_axis_local)
    
    # We want local clamping axis to be strictly horizontal (world Z component = 0)
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




def object_at_target_reward(
    env,
    target_pos: tuple,
    std: float = 0.15,
    min_height: float = 0.10,
    object_cfg=None,
) -> "torch.Tensor":
    """Reward the agent for moving the object to a fixed world-frame target.

    Uses a tanh kernel: reward = 1.0 at target, decays smoothly with distance.
    Only activates when object is above min_height (must be lifted first).

    Args:
        env: The environment.
        target_pos: (x, y, z) target position in world frame.
        std: Tanh length scale (m). Smaller = tighter requirement.
        min_height: Minimum z-height before reward activates.
        object_cfg: Configuration for the object asset.
    """
    import torch
    from isaaclab.assets import RigidObject
    from isaaclab.managers import SceneEntityCfg as _SceneEntityCfg
    if object_cfg is None:
        object_cfg = _SceneEntityCfg("object")
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w  # (num_envs, 3)

    target = torch.tensor(list(target_pos), device=obj_pos.device, dtype=obj_pos.dtype)
    target = target.unsqueeze(0).expand_as(obj_pos)

    distance = torch.norm(obj_pos - target, dim=1)
    lifted = (obj_pos[:, 2] > min_height).float()
    return lifted * (1.0 - torch.tanh(distance / std))

def arm_joint1_back_reward(
    env,
    target_angle: float = 3.14159,
    min_height: float = 0.20,
    joint_name: str = "openarm_right_joint1",
    object_cfg=None,
) -> "torch.Tensor":
    """Reward the agent for rotating the shoulder Joint 1 backwards (towards 180 degrees) after lifting the cup."""
    import torch
    from isaaclab.assets import RigidObject, Articulation
    from isaaclab.managers import SceneEntityCfg as _SceneEntityCfg
    if object_cfg is None:
        object_cfg = _SceneEntityCfg("object")
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene["robot"]

    lifted = (obj.data.root_pos_w[:, 2] > min_height).float()

    joint_idx = robot.find_joints(joint_name)[0][0]
    j1_pos = robot.data.joint_pos[:, joint_idx]

    # Encourage Joint 1 angle to approach target_angle (+pi or -pi)
    angle_diff = torch.abs(j1_pos - target_angle)
    reward = 1.0 - torch.tanh(angle_diff / 1.0)

    return lifted * reward


def object_pre_lift_xy_penalty(
    env,
    start_xy: tuple = (0.25, 0.0),
    lift_height_threshold: float = 0.35,
    object_cfg=None,
) -> "torch.Tensor":
    """Penalize horizontal XY movement of the cup before it is lifted to lift_height_threshold (0.35m).

    Forces the robot to lift the cup STRAIGHT UP vertically without any sideways drift.
    """
    import torch
    from isaaclab.assets import RigidObject
    from isaaclab.managers import SceneEntityCfg as _SceneEntityCfg
    if object_cfg is None:
        object_cfg = _SceneEntityCfg("object")
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w

    start_xy_t = torch.tensor(list(start_xy), device=obj_pos.device, dtype=obj_pos.dtype)
    xy_dist = torch.norm(obj_pos[:, :2] - start_xy_t, dim=1)

    # Active ONLY before cup reaches lift_height_threshold (0.35m)
    not_yet_lifted = (obj_pos[:, 2] < lift_height_threshold).float()
    return not_yet_lifted * xy_dist


def object_at_target_if_grasped_and_lifted(
    env,
    target_pos: tuple,
    std: float = 0.15,
    min_height: float = 0.35,
    gripper_threshold: float = 0.02,
    object_cfg=None,
    robot_cfg=None,
) -> "torch.Tensor":
    """Reward moving to target_pos ONLY IF:
       1. Gripper is CLOSED (finger joint angles <= gripper_threshold)
       2. Cup is LIFTED (object z >= min_height)

    If either condition is NOT met, reward is EXACTLY 0.0!
    """
    import torch
    from isaaclab.assets import RigidObject, Articulation
    from isaaclab.managers import SceneEntityCfg as _SceneEntityCfg

    if object_cfg is None:
        object_cfg = _SceneEntityCfg("object")
    if robot_cfg is None:
        robot_cfg = _SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"])

    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    obj_pos = obj.data.root_pos_w

    # Condition 1: Cup is lifted >= min_height
    is_lifted = (obj_pos[:, 2] >= min_height).float()

    # Condition 2: Gripper is closed (finger joint angles <= gripper_threshold)
    finger_ids, _ = robot.find_joints(robot_cfg.joint_names)
    finger_pos = robot.data.joint_pos[:, finger_ids]
    is_closed = (torch.max(finger_pos, dim=1)[0] <= gripper_threshold).float()

    # Gate: BOTH conditions must be met (1.0)
    gate = is_lifted * is_closed

    target = torch.tensor(list(target_pos), device=obj_pos.device, dtype=obj_pos.dtype)
    target = target.unsqueeze(0).expand_as(obj_pos)

    distance = torch.norm(obj_pos - target, dim=1)
    reward = 1.0 - torch.tanh(distance / std)

    return gate * reward


def joint1_transport_hint(
    env,
    target_joint1_rad: float = -1.57,
    min_height: float = 0.35,
    gripper_threshold: float = 0.02,
    std: float = 0.8,
    robot_cfg=None,
    object_cfg=None,
) -> "torch.Tensor":
    """Guide joint1 (shoulder rotation) toward target angle ONLY when cup is lifted + grasped.

    Acts as a 'path hint' telling the arm which way to rotate during transport.
    Without this, the policy has to discover the correct rotation direction by chance.

    Args:
        target_joint1_rad: Target angle for joint1 in radians.
            -1.57 = 90° right (toward via-point)
            -3.14 = 180° backward (toward final target)
        min_height: Cup must be above this height to activate (same gate as transport reward).
        gripper_threshold: Gripper finger joint must be below this to be considered closed.
        std: Gaussian width for the reward (larger = softer hint).
    """
    import torch
    from isaaclab.assets import RigidObject, Articulation
    from isaaclab.managers import SceneEntityCfg as _SceneEntityCfg

    if robot_cfg is None:
        robot_cfg = _SceneEntityCfg("robot", joint_names=["openarm_right_joint1"])
    if object_cfg is None:
        object_cfg = _SceneEntityCfg("object")

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Gate: cup lifted AND gripper closed
    obj_pos = obj.data.root_pos_w
    is_lifted = (obj_pos[:, 2] >= min_height).float()

    finger_cfg = _SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"])
    finger_cfg.resolve(env.scene)
    finger_ids = finger_cfg.joint_ids
    finger_pos = robot.data.joint_pos[:, finger_ids]
    is_closed = (torch.max(finger_pos, dim=1)[0] <= gripper_threshold).float()

    gate = is_lifted * is_closed

    # Reward: how close is joint1 to the target angle?
    joint1_ids, _ = robot.find_joints(robot_cfg.joint_names)
    joint1_angle = robot.data.joint_pos[:, joint1_ids[0]]

    target = torch.tensor(target_joint1_rad, device=joint1_angle.device, dtype=joint1_angle.dtype)
    angle_diff = torch.abs(joint1_angle - target)
    reward = 1.0 - torch.tanh(angle_diff / std)

    return gate * reward


def ee_self_clearance_penalty(
    env: ManagerBasedRLEnv,
    min_distance: float = 0.15,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the agent if the end-effector gets too close to the robot base/torso."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot = env.scene[robot_cfg.name]

    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    robot_base_w = robot.data.root_pos_w

    dist = torch.norm(ee_w - robot_base_w, dim=1)
    penalty = torch.clamp(min_distance - dist, min=0.0)
    return penalty


# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold


def ee_reached_object(
    env: ManagerBasedRLEnv,
    threshold: float = 0.03,
    z_offset: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Success termination for stage 1 (Reach): end-effector is close enough to the cup.

    Args:
        env: The environment.
        threshold: Maximum allowed distance (m) between EE and target point on the object.
        z_offset: Vertical offset (m) added to the object root position (same as in reward).
        object_cfg: Configuration for the object asset.
        ee_frame_cfg: Configuration for the end-effector frame sensor.
    """
    object: RigidObject = env.scene[object_cfg.name]
    from isaaclab.sensors import FrameTransformer
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    target_pos = object.data.root_pos_w.clone()
    target_pos[:, 2] += z_offset
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(target_pos - ee_w, dim=1)
    return distance < threshold


def object_grasped(
    env: ManagerBasedRLEnv,
    min_height: float = 0.09,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Success termination for stage 2 (Grasp): object has been lifted slightly off the table.

    Args:
        env: The environment.
        min_height: Minimum world-frame z-height (m) for the object to be considered grasped.
        object_cfg: Configuration for the object asset.
    """
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_pos_w[:, 2] > min_height


def object_lifted_success(
    env: ManagerBasedRLEnv,
    min_height: float = 0.25,
    tilt_limit: float = 45.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Success termination for stage 3 (Lift): object is held high and upright.

    Args:
        env: The environment.
        min_height: Minimum world-frame z-height (m) required.
        tilt_limit: Maximum allowed tilt angle (degrees) from vertical.
        object_cfg: Configuration for the object asset.
    """
    import math
    object: RigidObject = env.scene[object_cfg.name]
    height_ok = object.data.root_pos_w[:, 2] > min_height
    # Check upright: projection of local-Z onto world-Z
    x = object.data.root_quat_w[:, 1]
    y = object.data.root_quat_w[:, 2]
    z_proj = 1.0 - 2.0 * (x * x + y * y)
    tilt_ok = z_proj > math.cos(math.radians(tilt_limit))
    return height_ok & tilt_ok


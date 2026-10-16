# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm cup rotation task.

This task reuses the existing reach/grasp logic from the cup curriculum and adds
an explicit side-place rotation objective: grasp the cup, lift it, rotate it by
90 degrees, and move it to the robot side (and place it down).
"""

import math

import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.openarm.joint_pos_env_cfg import OpenArmCubeLiftEnvCfg


def quat_yaw_from_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Compute yaw angle from quaternion in [w, x, y, z] format."""
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def cup_rotate_to_side_reward(
    env,
    target_yaw: float = math.pi / 2.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the cup for aligning its yaw with a 90-degree side-placement target."""
    object = env.scene[object_cfg.name]
    yaw = quat_yaw_from_wxyz(object.data.root_quat_w)
    yaw_err = torch.atan2(torch.sin(yaw - target_yaw), torch.cos(yaw - target_yaw))
    return torch.exp(-torch.abs(yaw_err) / (math.pi / 12.0))


@configclass
class CupRotateRewardsCfg:
    """Rewards for the rotate-and-place task."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "z_offset": 0.055},
        weight=1.0,
    )

    gripper_close = RewTerm(
        func=mdp.gripper_closed_near_object_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
            "open_val": 0.044,
            "threshold": 0.05,
            "z_offset": 0.055,
        },
        weight=3.0,
    )

    ee_align = RewTerm(func=mdp.ee_align_cup_grasp_flat, weight=2.0)

    object_tilt = RewTerm(
        func=mdp.object_tilt_penalty,
        params={"tilt_angle_limit": 40.0},
        weight=-3.0,
    )

    # 保留 lifting_object 獎勵，確保機器人有「拿起」的動作，不會直接在桌面上拖行
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.12},
        weight=4.0,
    )

    rotate_to_side = RewTerm(
        func=cup_rotate_to_side_reward,
        params={"target_yaw": math.pi / 2.0},
        weight=8.0,
    )

    place_on_side = RewTerm(
        func=mdp.object_goal_distance,
        # 【修改點】將 minimal_height 改為 0.0，讓靠近桌面的放置動作能獲得最高權重獎勵
        params={"std": 0.08, "minimal_height": 0.0, "command_name": "object_pose"},
        weight=12.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint.*", "openarm_right_finger_joint.*"])
        },
    )


@configclass
class CupRotateTerminationsCfg:
    """Termination conditions for the rotate-and-place task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    side_placement = DoneTerm(
        func=mdp.object_reached_goal,
        params={"command_name": "object_pose", "threshold": 0.05},
    )


@configclass
class OpenArmRotateEnvCfg(OpenArmCubeLiftEnvCfg):
    """Configuration for the custom OpenArm cup rotation task."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards = CupRotateRewardsCfg()
        self.terminations = CupRotateTerminationsCfg()

        # Keep gripper closed when the agent has reached the cup and is rotating it.
        self.actions.gripper_action.close_command_expr = {"openarm_right_finger_joint.*": 0.0}

        # The task is a combined reach -> grasp -> lift -> rotate -> place task.
        self.episode_length_s = 12.0

        # Set target pose for the object.
        self.commands.object_pose.body_name = "openarm_right_hand"
        self.commands.object_pose.ranges.pos_x = (0.25, 0.35)
        self.commands.object_pose.ranges.pos_y = (-0.15, 0.15) 
        # 【修改點】將目標高度 pos_z 降到接近桌面 (0.02, 0.05)，引導機器人將物體放下
        self.commands.object_pose.ranges.pos_z = (0.02, 0.05)
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (math.pi / 2.0, math.pi / 2.0)

        # Increase grip stability for the rotation stage.
        self.scene.robot.actuators["openarm_gripper"].stiffness = 5e3
        self.scene.robot.actuators["openarm_gripper"].damping = 3e2


@configclass
class OpenArmRotateEnvCfg_PLAY(OpenArmRotateEnvCfg):
    """Play variant for the custom OpenArm rotate task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 1 — Cup Reach: train the robot to move its end-effector close to the cup.

Success condition: EE distance to cup < 3 cm.
No goal-pose command is used; the cup stays at a fixed position.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.openarm.joint_pos_env_cfg import OpenArmCubeLiftEnvCfg

##
# Reward configuration
##


@configclass
class CupReachRewardsCfg:
    """Rewards for Stage 1: approach the cup."""

    # Primary: get the end-effector close to the cup
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1, "z_offset": 0.055}, weight=3.0)

    # Orientation: force horizontal (flat) alignment to align with cup body
    ee_align = RewTerm(func=mdp.ee_align_cup_grasp_flat, weight=2.0)

    # Penalty for tipping the cup over (tilt > 45 degrees)
    object_tilt = RewTerm(func=mdp.object_tilt_penalty, params={"tilt_angle_limit": 45.0}, weight=-3.0)

    # Regularisation
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint.*", "openarm_right_finger_joint.*"])},
    )


##
# Termination configuration
##


@configclass
class CupReachTerminationsCfg:
    """Terminations for Stage 1."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    # Success: EE within 1.5 cm of cup
    ee_reached = DoneTerm(
        func=mdp.ee_reached_object,
        params={"threshold": 0.005, "z_offset": 0.055},
    )


##
# Environment configuration
##


@configclass
class CupReachEnvCfg(OpenArmCubeLiftEnvCfg):
    """Configuration for Stage 1 — Cup Reach."""

    def __post_init__(self):
        # Inherit full robot / object / scene setup
        super().__post_init__()

        # Replace rewards and terminations with stage-specific ones
        self.rewards = CupReachRewardsCfg()
        self.terminations = CupReachTerminationsCfg()

        # Force the gripper to always stay open in Stage 1
        # By setting close_command_expr same as open_command_expr, the gripper remains open (0.044)
        # regardless of what the RL policy outputs. This preserves action dimensions.
        self.actions.gripper_action.close_command_expr = {"openarm_right_finger_joint.*": 0.044}

        # Longer episode: agent needs time to navigate
        self.episode_length_s = 8.0


@configclass
class CupReachEnvCfg_PLAY(CupReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

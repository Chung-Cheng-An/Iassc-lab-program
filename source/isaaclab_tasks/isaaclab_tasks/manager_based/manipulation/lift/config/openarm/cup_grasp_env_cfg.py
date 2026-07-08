# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 2 — Cup Grasp: train the robot to close the gripper around the cup and lift it slightly.

Success condition: object z-height > 0.09 m (slightly above table top).
The gripper_closed_reward acts as a proxy for actual contact since no contact sensor is required.
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
class CupGraspRewardsCfg:
    """Rewards for Stage 2: grasp the cup."""

    # Keep EE near the cup so the agent doesn't wander
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1, "z_offset": 0.055}, weight=1.0)

    # Orientation: force horizontal (flat) alignment to align with cup body
    ee_align = RewTerm(func=mdp.ee_align_cup_grasp_flat, weight=2.0)

    # Encourage gripper closure (proxy for grasping), but ONLY when close to the cup
    gripper_closed = RewTerm(
        func=mdp.gripper_closed_near_object_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
            "open_val": 0.044,
            "threshold": 0.05,
            "z_offset": 0.055,
        },
        weight=3.0,
    )

    # Hard penalty for closing the gripper while still far from the cup (prevents reward hacking)
    gripper_premature_close = RewTerm(
        func=mdp.gripper_premature_close_penalty,
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
            "open_val": 0.044,
            "penalty_dist": 0.08,
            "z_offset": 0.055,
        },
        weight=-5.0,
    )

    # Progressive lifting rewards — higher = more reward
    # Cup starts at z=0.055m (table top)
    lifting_tier1 = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.08}, weight=3.0)   # ~2.5cm above table
    lifting_tier2 = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.15}, weight=6.0)   # ~9.5cm above table
    lifting_tier3 = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.25}, weight=12.0)  # ~19.5cm above table

    # Continuous lifting reward to guide the agent smoothly from table height (0.055m) upward
    lifting_continuous = RewTerm(
        func=mdp.object_lift_height_reward,
        params={"table_height": 0.055},
        weight=60.0,
    )

    # Keep the cup upright while grasping
    object_tilt = RewTerm(func=mdp.object_tilt_penalty, params={"tilt_angle_limit": 40.0}, weight=-4.0)

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
class CupGraspTerminationsCfg:
    """Terminations for Stage 2."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    # Success: cup lifted to the highest tier (0.25m)
    object_grasped = DoneTerm(
        func=mdp.object_grasped,
        params={"min_height": 0.25},
    )


##
# Environment configuration
##


@configclass
class CupGraspEnvCfg(OpenArmCubeLiftEnvCfg):
    """Configuration for Stage 2 — Cup Grasp."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards = CupGraspRewardsCfg()
        self.terminations = CupGraspTerminationsCfg()

        # Longer episode to allow the full reach → close → lift motion
        self.episode_length_s = 10.0

        # Warm-start: override robot initial joint positions with
        # the converged Stage 1 pose (placeholder values, needs to be updated after Stage 1 runs).
        self.scene.robot.init_state.joint_pos = {
            "openarm_right_joint1": 1.473,
            "openarm_right_joint2": -0.790,
            "openarm_right_joint3": -1.57,
            "openarm_right_joint4": 1.505,
            "openarm_right_joint5": -0.171,
            "openarm_right_joint6": -0.195,
            "openarm_right_joint7": 0.381,
            "openarm_right_finger_joint.*": 0.044,  # gripper fully open
        }

        # Increase gripper grip force for more reliable grasping
        self.scene.robot.actuators["openarm_gripper"].stiffness = 5e3
        self.scene.robot.actuators["openarm_gripper"].damping = 3e2


@configclass
class CupGraspEnvCfg_PLAY(CupGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 3 — Cup Lift: train the robot to raise the cup to height while keeping it upright.

Success condition: object z-height > 0.25 m AND tilt < 45°.
No goal-pose command is used; success is defined purely by height + orientation.
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
class CupLiftRewardsCfg:
    """Rewards for Stage 3: lift the cup high while keeping it upright."""

    # Encourage the EE to remain near the cup while lifting (using 0.055 grasp offset)
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1, "z_offset": 0.055}, weight=0.5)

    # Orientation: force horizontal (flat) alignment to align with cup body
    ee_align = RewTerm(func=mdp.ee_align_cup_grasp_flat, weight=2.0)

    # Binary bonus when cup is above the target height
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.25}, weight=10.0)

    # Continuous height reward — pushes agent to go higher than the minimum
    object_height = RewTerm(
        func=mdp.object_height_reward,
        params={"target_height": 0.4},
        weight=5.0,
    )

    # Keep cup upright (1 = vertical, 0 = horizontal)
    object_upright = RewTerm(func=mdp.object_upright_reward, weight=3.0)

    # Penalise excessive tilting
    object_tilt = RewTerm(func=mdp.object_tilt_penalty, params={"tilt_angle_limit": 45.0}, weight=-5.0)

    # Regularisation
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_joint.*", "openarm_finger_joint.*"])},
    )


##
# Termination configuration
##


@configclass
class CupLiftTerminationsCfg:
    """Terminations for Stage 3."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    # Success: cup > 25 cm AND upright within 45°
    object_lifted = DoneTerm(
        func=mdp.object_lifted_success,
        params={"min_height": 0.25, "tilt_limit": 45.0},
    )


##
# Environment configuration
##


@configclass
class CupLiftEnvCfg(OpenArmCubeLiftEnvCfg):
    """Configuration for Stage 3 — Cup Lift."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards = CupLiftRewardsCfg()
        self.terminations = CupLiftTerminationsCfg()

        self.episode_length_s = 10.0

        # Warm-start: override robot initial joint positions with
        # the converged Stage 1/2 grasp pose. Gripper starts closed (0.0).
        self.scene.robot.init_state.joint_pos = {
            "openarm_joint1": 1.473,
            "openarm_joint2": -0.790,
            "openarm_joint3": -1.57,
            "openarm_joint4": 1.505,
            "openarm_joint5": -0.171,
            "openarm_joint6": -0.195,
            "openarm_joint7": 0.381,
            "openarm_finger_joint.*": 0.044,  # gripper starts open around the cup to avoid mesh collision at spawn
        }

        # Lock gripper to closed state (0.0) in Stage 3 so the arm focus is purely on lifting
        self.actions.gripper_action.open_command_expr = {"openarm_finger_joint.*": 0.0}

        # Increase gripper stiffness/damping for stable grasp under gravity
        self.scene.robot.actuators["openarm_gripper"].stiffness = 5e3
        self.scene.robot.actuators["openarm_gripper"].damping = 3e2


@configclass
class CupLiftEnvCfg_PLAY(CupLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

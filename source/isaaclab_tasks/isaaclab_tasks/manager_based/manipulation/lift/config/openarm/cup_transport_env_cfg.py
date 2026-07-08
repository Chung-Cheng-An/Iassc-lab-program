# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 4 — Cup Transport: train the robot to carry the cup to a random goal position.

This stage is equivalent to the original full-task environment (joint_pos_env_cfg),
but isolated as the final curriculum step and with a success termination added.

Success condition: cup within 3 cm of the goal position (from the object_pose command).
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
class CupTransportRewardsCfg:
    """Rewards for Stage 4: carry the cup to the goal pose."""

    # Keep EE near cup so the grip is maintained
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.1)

    # Orientation: force horizontal (flat) alignment to align with cup body
    ee_align = RewTerm(func=mdp.ee_align_cup_grasp_flat, weight=2.0)

    # Encourage holding the cup at height
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.15}, weight=15.0)

    # Penalise tilting the cup (it would spill water)
    object_tilt = RewTerm(func=mdp.object_tilt_penalty, params={"tilt_angle_limit": 45.0}, weight=-3.0)

    # Coarse goal-tracking reward
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.15, "command_name": "object_pose"},
        weight=16.0,
    )

    # Fine-grained goal-tracking reward
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.15, "command_name": "object_pose"},
        weight=5.0,
    )

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
class CupTransportTerminationsCfg:
    """Terminations for Stage 4."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    # Success: cup within 3 cm of goal position
    object_reached_goal = DoneTerm(
        func=mdp.object_reached_goal,
        params={"command_name": "object_pose", "threshold": 0.03},
    )


##
# Environment configuration
##


@configclass
class CupTransportEnvCfg(OpenArmCubeLiftEnvCfg):
    """Configuration for Stage 4 — Cup Transport."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards = CupTransportRewardsCfg()
        self.terminations = CupTransportTerminationsCfg()

        # Standard episode length (goal resamples every 5 s anyway)
        self.episode_length_s = 5.0


@configclass
class CupTransportEnvCfg_PLAY(CupTransportEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

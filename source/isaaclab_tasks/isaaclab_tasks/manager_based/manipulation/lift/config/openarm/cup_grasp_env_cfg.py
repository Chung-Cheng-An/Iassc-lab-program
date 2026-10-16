# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 2 -- Cup Grasp: train the robot to approach, envelope, squeeze and lift the cup off the table.

Design principle:
  - Inherits from OpenArmCubeLiftEnvCfg with exact same action scale (0.8).
  - Warm-started directly from Stage 1 (Reach) checkpoints (model_2999.pt).
  - Rewards approach, horizontal orientation, elbow human prior, gripper closure near cup, and upward lift.
  - Hard termination on cup topple (tilt > 35 deg).
  - Success condition: cup lifted off table (z > 0.05m).
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
    """Rewards for Stage 2: approach, envelope, squeeze and lift the cup."""

    # 1. 導引手臂靠近杯身腰部 (延續 Stage 1 最佳高度：z_offset = 0.055, approach_yaw = 0.0)
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.15, "z_offset": 0.055, "approach_offset": 0.0, "approach_yaw": 0.0},
        weight=4.0,
    )

    # 2. 夾爪水平對齊正前方 (雙指正對水杯，杯腰置入兩指正中)
    ee_align = RewTerm(
        func=mdp.ee_align_cup_grasp_flat,
        params={"approach_yaw": 0.0, "tool_axis": "z", "clamping_axis": "y"},
        weight=2.0,
    )

    # 3. 擬人化姿態約束：手肘維持自然垂直角度，嚴禁外翻雞翅！
    elbow_human_prior = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint3"])},
    )

    # 4. 核心抓取獎勵：當夾爪深入杯腰 (距離 < 4.5cm) 時，雙指閉合給予強力正反饋！
    gripper_closed = RewTerm(
        func=mdp.gripper_closed_near_object_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
            "open_val": 0.044,
            "threshold": 0.045,
            "z_offset": 0.055,
        },
        weight=15.0,
    )

    # 5. 空中提早捏拳重罰：距離杯身大於 5.0cm 時提早閉合給予重罰，逼它保持張開滑入杯腰！
    premature_close = RewTerm(
        func=mdp.gripper_premature_close_penalty,
        params={
            "penalty_dist": 0.05,
            "z_offset": 0.055,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
            "open_val": 0.044,
        },
        weight=-6.0,
    )

    # 6. 向上抬起獎勵：當杯子被抓起離地時給予高額獎勵 (目標高度 10cm)
    object_lifted = RewTerm(
        func=mdp.object_height_reward,
        params={"target_height": 0.10, "object_cfg": SceneEntityCfg("object")},
        weight=12.0,
    )

    # 7. 防撞翻/防灑水保護：杯子傾斜超過 15 度漸進扣分
    object_tilt = RewTerm(
        func=mdp.object_tilt_penalty,
        params={"tilt_angle_limit": 15.0},
        weight=-4.0,
    )

    # 正則化項 (平滑動作)
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

    # 硬性失敗終止：杯子若傾斜超過 35 度立刻判失敗結束回合 (GAME OVER)
    cup_toppled = DoneTerm(
        func=mdp.cup_tilted,
        params={"tilt_limit": 35.0, "object_cfg": SceneEntityCfg("object")},
    )

    # 成功判定：杯子被夾起且離地 > 0.05m (5 公分)，宣告抓取成功！
    object_grasped = DoneTerm(
        func=mdp.object_grasped,
        params={"min_height": 0.05},
    )


##
# Environment configuration
##


@configclass
class CupGraspEnvCfg(OpenArmCubeLiftEnvCfg):
    """Configuration for Stage 2 -- Cup Grasp."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards = CupGraspRewardsCfg()
        self.terminations = CupGraspTerminationsCfg()

        # 回合時間 5.0 秒 (250 步：前 35 步伸入手掌，後續收緊合爪並平穩抬起)
        self.episode_length_s = 5.0

        # 維持與 Stage 1 相同的 action scale (0.8)，完美無縫繼承 Stage 1 權重
        self.actions.arm_action.scale = 0.8

        # 夾爪控制：允許閉合至 0.0m (解鎖合爪)
        self.actions.gripper_action.close_command_expr = {"openarm_right_finger_joint.*": 0.0}

        # 提高夾爪抓取剛性與阻尼 (高剛性夾取，牢固抓握)
        self.scene.robot.actuators["openarm_gripper"].stiffness = 5e3
        self.scene.robot.actuators["openarm_gripper"].damping = 3e2

        # 桌面杯子位置隨機化 (與 Stage 1 相同的 6cm x 8cm 範圍)
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.04, 0.04),
            "z": (0.0, 0.0),
        }


@configclass
class CupGraspEnvCfg_PLAY(CupGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

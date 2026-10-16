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
    """Rewards for Stage 1: approach and envelop the cup (ready for immediate grasp)."""

    # Primary: 直接引導末端滑入杯腰中心 (z_offset=0.055, approach_offset=0.0)，雙指深入包覆水杯兩側，讓下一步可直接合爪夾取！
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.15, "z_offset": 0.055, "approach_offset": 0.0, "approach_yaw": 0.0},
        weight=6.0,
    )

    # Orientation: 手掌對齊正面水平推進 (雙指開口水平正對水杯，杯腰置入兩指正中)
    ee_align = RewTerm(
        func=mdp.ee_align_cup_grasp_flat,
        params={"approach_yaw": 0.0, "tool_axis": "z", "clamping_axis": "y"},
        weight=2.0,
    )

    # Human Posture Prior: 懲罰手肘外翻旋轉 (joint3 偏離 0.0 自然下垂垂直角度)，強制維持真人自然姿態！
    elbow_human_prior = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint3"])},
    )

    # Soft Landing: 進入杯身 10cm 內獎勵關節平滑減速，避免撞翻杯子
    soft_landing = RewTerm(func=mdp.ee_soft_landing_near_object, params={"distance_threshold": 0.10}, weight=2.0)

    # Penalty: 杯子傾斜超過 15 度早期扣分警示
    object_tilt = RewTerm(func=mdp.object_tilt_penalty, params={"tilt_angle_limit": 15.0}, weight=-3.0)

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

    # 硬性失敗終止：杯子傾斜超過 20 度立刻判失敗結束回合 (GAME OVER)
    cup_toppled = DoneTerm(
        func=mdp.cup_tilted,
        params={"tilt_limit": 20.0, "object_cfg": SceneEntityCfg("object")},
    )

    # 移除 ee_reached 提早結束！讓手臂抵達杯腰後持續停留在杯旁賺取滿分獎勵，學會「死死定格、絕不後縮」！
    ee_reached = None


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

        # 5.0 秒回合 (250 steps)：手臂前 35 步滑入杯腰，後續 200+ 步全程定格包覆水杯
        self.episode_length_s = 5.0

        # 開啟桌面杯子位置隨機化 (方案 B：擴大抗位移汎化能力)
        # X 範圍: 0.22m ~ 0.28m (基準 0.25m ± 3cm，前後共 6 公分)
        # Y 範圍: -0.04m ~ +0.04m (基準 0.00m ± 4cm，左右共 8 公分)
        # Z 範圍: 0.0 (嚴格緊貼桌面，不彈跳、不掉落)
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.04, 0.04),
            "z": (0.0, 0.0),
        }


@configclass
class CupReachEnvCfg_PLAY(CupReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

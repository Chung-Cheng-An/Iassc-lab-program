# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 3 -- Cup Pick and Place:
Robots smoothly pick up cup, transport to target table zone, gently lower and place on table, then release gripper.
Features smooth curriculum reward shaping: maintains grasp robustness from model_4498 while guiding transport, descent and release.
"""

import math
import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.openarm.joint_pos_env_cfg import OpenArmCubeLiftEnvCfg


def gripper_grasp_phase_reward(
    env,
    open_val: float = 0.044,
    grasp_dist_threshold: float = 0.045,
    z_offset: float = 0.072,
    target_place_xy: tuple = (0.16, -0.28),
    place_xy_radius: float = 0.08,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward closing gripper near cup during grasp & transport.
    Shuts off once cup is safely placed on table inside goal zone to avoid fighting gripper release.
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    finger_ids, _ = robot.find_joints(robot_cfg.joint_names)
    finger_pos = robot.data.joint_pos[:, finger_ids]
    closure = 1.0 - torch.clamp(finger_pos.mean(dim=-1) / open_val, 0.0, 1.0)

    target_pos = obj.data.root_pos_w.clone()
    target_pos[:, 2] += z_offset
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(target_pos - ee_w, dim=1)
    gating = torch.exp(-distance / grasp_dist_threshold)

    # 當杯子已經在目標放置區且已經著地時，關閉夾爪夾緊獎勵 (使用各環境 local 座標)
    target_xy = torch.tensor(list(target_place_xy), device=obj.data.root_pos_w.device, dtype=obj.data.root_pos_w.dtype)
    cup_local_xy = obj.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    dist_to_place = torch.norm(cup_local_xy - target_xy, dim=1)
    is_at_place_on_table = (dist_to_place < place_xy_radius) & (obj.data.root_pos_w[:, 2] < 0.025)
    not_placed = (~is_at_place_on_table).float()

    return not_placed * (closure * gating)


def cup_pick_transport_place_reward(
    env,
    target_pos: tuple = (0.16, -0.28, 0.0),
    xy_std: float = 0.08,
    z_std: float = 0.04,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Smooth multi-phase Pick, Transport and Descent guidance:
    1. Transport phase: Guide XY toward target while cup is lifted and gripped.
    2. Descend phase: Once near target XY (< 0.08m), smoothly guide Z down to table surface (0.0m) and reward touchdown.
    """
    obj = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    obj_pos = obj.data.root_pos_w

    finger_ids, _ = robot.find_joints(robot_cfg.joint_names)
    finger_pos = robot.data.joint_pos[:, finger_ids]
    is_gripping = (finger_pos.mean(dim=-1) <= 0.038).float()

    target_xy = torch.tensor(list(target_pos[:2]), device=obj_pos.device, dtype=obj_pos.dtype)
    cup_local_xy = obj_pos[:, :2] - robot.data.root_pos_w[:, :2]
    xy_dist = torch.norm(cup_local_xy - target_xy, dim=1)

    # 門控：在空中達到安全騰空高度 (>= 4.5cm) 且手爪夾持時給予平移搬運獎勵 (防貼地拖曳摩擦翻倒)
    is_lifted = (obj_pos[:, 2] >= 0.045).float() * is_gripping
    xy_progress = is_lifted * (1.0 - torch.tanh(xy_dist / xy_std))

    # 當 XY 接近放置目標 (< 0.08m) 時，引導高度平緩降至桌面 (Z -> 0.0) 並獎勵著地
    # 注意：此項不可乘以 is_lifted，否則杯子一旦降至桌面 (Z < 0.025) 獎勵歸零，會導致模型懸空在半空中不敢著地
    near_xy = torch.clamp(1.0 - xy_dist / 0.08, min=0.0)
    z_dist = torch.clamp(obj_pos[:, 2] - target_pos[2], min=0.0)
    z_descend_reward = 1.0 - torch.tanh(z_dist / z_std)
    touchdown = (obj_pos[:, 2] <= 0.025).float()

    return xy_progress + near_xy * (1.5 * z_descend_reward + 1.0 * touchdown)


def cup_placed_on_table_reward(
    env,
    target_pos: tuple = (0.16, -0.28, 0.0),
    xy_threshold: float = 0.08,
    max_height: float = 0.025,
    max_tilt_deg: float = 22.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward placing the cup stably and uprightly on the table near target xy."""
    obj = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    obj_pos = obj.data.root_pos_w

    target_xy = torch.tensor(list(target_pos[:2]), device=obj_pos.device, dtype=obj_pos.dtype)
    cup_local_xy = obj_pos[:, :2] - robot.data.root_pos_w[:, :2]
    xy_dist = torch.norm(cup_local_xy - target_xy, dim=1)
    is_at_goal_xy = (xy_dist <= xy_threshold).float()

    # 必須接觸桌面 (Z <= 2.5cm)
    is_on_table = (obj_pos[:, 2] <= max_height).float()

    # 姿態端正 (傾角 <= 22度)
    x = obj.data.root_quat_w[:, 1]
    y = obj.data.root_quat_w[:, 2]
    z_proj = 1.0 - 2.0 * (x * x + y * y)
    cos_thresh = math.cos(math.radians(max_tilt_deg))
    is_upright = (z_proj >= cos_thresh).float()

    return is_at_goal_xy * is_on_table * is_upright


def cup_lift_phase_reward(
    env,
    minimal_height: float = 0.045,
    target_place_xy: tuple = (0.16, -0.28),
    shutoff_radius: float = 0.12,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward lifting the cup at start & during transport.
    Crucially shuts off once entering destination zone so robot is NOT penalized for lowering cup to table!
    """
    obj = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]

    target_xy = torch.tensor(list(target_place_xy), device=obj.data.root_pos_w.device, dtype=obj.data.root_pos_w.dtype)
    cup_local_xy = obj.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    dist_to_place = torch.norm(cup_local_xy - target_xy, dim=1)

    # 只要杯子接近放置區 (< 0.12m) 或 Y <= -0.20m，立即關閉空中懸空獎勵，徹底根除目標區向上抽抬的衝動！
    not_at_goal = (dist_to_place > shutoff_radius) & (cup_local_xy[:, 1] > -0.20)
    is_lifted = (obj.data.root_pos_w[:, 2] > minimal_height).float()

    return not_at_goal.float() * is_lifted


def release_gripper_at_goal_reward(
    env,
    target_pos: tuple = (0.16, -0.28, 0.0),
    xy_threshold: float = 0.09,
    max_height: float = 0.025,
    max_speed: float = 0.35,
    max_tilt_deg: float = 22.0,
    open_val: float = 0.044,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Reward opening the gripper ONLY when cup is safely, softly, and uprightly placed on the table.
    Uses continuous sigmoid action shaping to directly guide the actor's 8th action from -5.5 to +1.0!
    """
    obj = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    obj_pos = obj.data.root_pos_w
    obj_vel = obj.data.root_lin_vel_w

    target_xy = torch.tensor(list(target_pos[:2]), device=obj_pos.device, dtype=obj_pos.dtype)
    cup_local_xy = obj_pos[:, :2] - robot.data.root_pos_w[:, :2]
    xy_dist = torch.norm(cup_local_xy - target_xy, dim=1)

    # 1. 座標在目標圓圈內且高度已著地 (Z <= 2.5cm)
    at_goal_zone = (xy_dist <= xy_threshold) & (obj_pos[:, 2] <= max_height)

    # 2. 嚴格起步隔離保護：杯子必須已自起點 [0.25, 0.0] 搬運平移超過 12cm
    start_xy = torch.tensor([0.25, 0.0], device=obj_pos.device, dtype=obj_pos.dtype)
    dist_from_start = torch.norm(cup_local_xy - start_xy, dim=1)
    has_transported = dist_from_start > 0.12

    # 3. 杯身垂直不傾斜 (傾角 <= 22度)
    x = obj.data.root_quat_w[:, 1]
    y = obj.data.root_quat_w[:, 2]
    z_proj = 1.0 - 2.0 * (x * x + y * y)
    cos_thresh = math.cos(math.radians(max_tilt_deg))
    is_upright = (z_proj >= cos_thresh)

    # 4. 杯子已大致停穩 (速度 <= 35 cm/s)
    speed = torch.norm(obj_vel, dim=1)
    is_stationary = (speed <= max_speed)

    is_placed_properly = at_goal_zone & has_transported & is_upright & is_stationary

    # 5. 主動動作梯度引導 (寬域 Sigmoid 避免在 -5.5 處梯度飽和) 快速將夾爪指令推向 >= 0 開爪
    grip_act = env.action_manager.action[:, 7]
    action_gradient = torch.sigmoid(grip_act * 0.35 + 1.0)

    finger_ids, _ = robot.find_joints(robot_cfg.joint_names)
    finger_pos = robot.data.joint_pos[:, finger_ids]
    open_ratio = torch.clamp(finger_pos.mean(dim=-1) / open_val, 0.0, 1.0)

    return is_placed_properly.float() * (action_gradient + 2.0 * open_ratio)


def arm_still_after_release_reward(
    env,
    target_pos: tuple = (0.16, -0.28, 0.0),
    xy_threshold: float = 0.09,
    max_height: float = 0.025,
    max_tilt_deg: float = 22.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_joint[1-7]"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
) -> torch.Tensor:
    """Penalize arm motion after cup is placed and gripper is open to maintain steady stillness."""
    obj = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    obj_pos = obj.data.root_pos_w

    target_xy = torch.tensor(list(target_pos[:2]), device=obj_pos.device, dtype=obj_pos.dtype)
    cup_local_xy = obj_pos[:, :2] - robot.data.root_pos_w[:, :2]
    xy_dist = torch.norm(cup_local_xy - target_xy, dim=1)

    at_goal_on_table = (xy_dist <= xy_threshold) & (obj_pos[:, 2] <= max_height)
    x = obj.data.root_quat_w[:, 1]
    y = obj.data.root_quat_w[:, 2]
    z_proj = 1.0 - 2.0 * (x * x + y * y)
    cos_thresh = math.cos(math.radians(max_tilt_deg))
    is_upright = (z_proj >= cos_thresh)

    finger_ids, _ = robot.find_joints(finger_cfg.joint_names)
    is_open = (robot.data.joint_pos[:, finger_ids].mean(dim=-1) >= 0.020)

    is_placed_and_released = at_goal_on_table & is_upright & is_open

    arm_ids, _ = robot.find_joints(robot_cfg.joint_names)
    arm_vel = robot.data.joint_vel[:, arm_ids]
    vel_norm = torch.norm(arm_vel, dim=-1)

    stillness = torch.exp(-vel_norm / 0.2)
    return is_placed_and_released.float() * stillness


##
# Reward configuration
##


@configclass
class CupPickPlaceRewardsCfg:
    """Rewards for Stage 3: Pick up, transport to target, gently lower upright, and release."""

    # 1. 導引手臂靠近杯身中上部 (z_offset = 0.072m，避開下半部錐形，防放桌干涉)
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.15, "z_offset": 0.072, "approach_offset": 0.0, "approach_yaw": 0.0},
        weight=4.0,
    )

    # 2. 夾爪水平對齊正前方
    ee_align = RewTerm(
        func=mdp.ee_align_cup_grasp_flat,
        params={"approach_yaw": 0.0, "tool_axis": "z", "clamping_axis": "y"},
        weight=2.0,
    )

    # 3. 擬人化姿態約束：手肘維持自然垂直角度，嚴禁外翻
    elbow_human_prior = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint3"])},
    )

    # 4. 抓取階段閉合夾爪 (在杯身中上部果斷咬死)
    gripper_closed = RewTerm(
        func=gripper_grasp_phase_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
            "open_val": 0.044,
            "grasp_dist_threshold": 0.045,
            "z_offset": 0.072,
            "target_place_xy": (0.16, -0.28),
            "place_xy_radius": 0.08,
        },
        weight=15.0,
    )

    # 5. 空中提早閉爪懲罰 (防提前捏拳推倒杯身)
    premature_close = RewTerm(
        func=mdp.gripper_premature_close_penalty,
        params={
            "penalty_dist": 0.05,
            "z_offset": 0.072,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
            "open_val": 0.044,
        },
        weight=-6.0,
    )

    # 6. 先抬起杯子離地至安全淨空高度 (進入目標平臺自動關閉，徹底根絕抽抬甩杯衝動)
    object_lifted = RewTerm(
        func=cup_lift_phase_reward,
        params={
            "minimal_height": 0.045,
            "target_place_xy": (0.16, -0.28),
            "shutoff_radius": 0.12,
        },
        weight=8.0,
    )

    # 7. 搬運平移 + 靠攏目標時引導下放回桌面
    cup_transport_and_place = RewTerm(
        func=cup_pick_transport_place_reward,
        params={
            "target_pos": (0.16, -0.28, 0.0),
            "xy_std": 0.08,
            "z_std": 0.04,
        },
        weight=15.0,
    )

    # 8. 放置到桌面上穩定端正獎勵 (高度降回桌面 z <= 0.025m 且姿態垂直)
    cup_at_place_goal = RewTerm(
        func=cup_placed_on_table_reward,
        params={
            "target_pos": (0.16, -0.28, 0.0),
            "xy_threshold": 0.09,
            "max_height": 0.025,
            "max_tilt_deg": 22.0,
        },
        weight=15.0,
    )

    # 9. 放置成功後的鬆手開爪主動梯度引導 (sigmoid 動作引導快速扭轉為 +1.0)
    release_at_goal = RewTerm(
        func=release_gripper_at_goal_reward,
        params={
            "target_pos": (0.16, -0.28, 0.0),
            "xy_threshold": 0.09,
            "max_height": 0.025,
            "max_speed": 0.35,
            "max_tilt_deg": 22.0,
            "open_val": 0.044,
        },
        weight=25.0,
    )

    # 10. 放手後手臂安靜定身獎勵 (杜絕放手後手肘晃動推倒杯子)
    post_release_stillness = RewTerm(
        func=arm_still_after_release_reward,
        params={
            "target_pos": (0.16, -0.28, 0.0),
            "xy_threshold": 0.09,
            "max_height": 0.025,
            "max_tilt_deg": 22.0,
        },
        weight=15.0,
    )

    # 11. 肩部旋轉路徑引導 (引導肩部轉向右側/右後側目標，降低探索難度)
    joint1_hint = RewTerm(
        func=mdp.joint1_transport_hint,
        params={
            "target_joint1_rad": -0.65,
            "min_height": 0.03,
            "gripper_threshold": 0.038,
            "std": 0.5,
        },
        weight=4.0,
    )

    # 12. 水杯防傾斜約束 (平滑懲罰，引導垂直運水)
    cup_tilt_penalty = RewTerm(
        func=mdp.object_tilt_penalty,
        params={"tilt_angle_limit": 8.0, "object_cfg": SceneEntityCfg("object")},
        weight=-8.0,
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
class CupPickPlaceTerminationsCfg:
    """Terminations for Stage 3 Pick and Place."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    # 防翻倒終止：杯子若傾斜超過 35 度立刻視為翻倒重置 (放寬傾角探索邊界，允許下放放爪接觸時微晃，不再動輒掐斷訓練)
    cup_toppled = DoneTerm(
        func=mdp.cup_tilted,
        params={"tilt_limit": 35.0, "object_cfg": SceneEntityCfg("object")},
    )


##
# Environment configuration
##


@configclass
class CupPickPlaceEnvCfg(OpenArmCubeLiftEnvCfg):
    """Configuration for Stage 3 -- Cup Pick and Place."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards = CupPickPlaceRewardsCfg()
        self.terminations = CupPickPlaceTerminationsCfg()

        # 回合時間 6.0 秒 (300 步：充足時間靠近、抓起、平移並放置回桌面與鬆開)
        self.episode_length_s = 6.0

        # 維持與 Stage 1/2 相同的 action scale (0.8)
        self.actions.arm_action.scale = 0.8

        # 夾爪控制：允許閉合至 0.0m，張開至 0.044m
        self.actions.gripper_action.close_command_expr = {"openarm_right_finger_joint.*": 0.0}
        self.actions.gripper_action.open_command_expr = {"openarm_right_finger_joint.*": 0.044}

        # 提高夾爪抓取剛性與阻尼
        self.scene.robot.actuators["openarm_gripper"].stiffness = 5e3
        self.scene.robot.actuators["openarm_gripper"].damping = 3e2

        # 桌面杯子初始位置 (微幅隨機化 4cm x 4cm)
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (0.0, 0.0),
        }

        # 核心目標對齊：鎖定放置目標在機器人 Base 座標系 (Base 位於 Z = -0.20m，桌面為 Z = 0.0m，故 Base 下 Z = +0.20m)
        self.commands.object_pose.ranges.pos_x = (0.16, 0.16)
        self.commands.object_pose.ranges.pos_y = (-0.28, -0.28)
        self.commands.object_pose.ranges.pos_z = (0.20, 0.20)


@configclass
class CupPickPlaceEnvCfg_PLAY(CupPickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # 評估與展示模式關閉空中提早終止，確保完整走完平移放杯全流程
        if hasattr(self, "terminations"):
            self.terminations.cup_toppled = None
            self.terminations.object_dropping = None

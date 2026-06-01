# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import math

from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.openarm.lift_openarm_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class OpenArmCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Debug first: glass cup is heavier than DexCube, so start with fewer environments.
        # After the cup works, you can increase this gradually.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Set OpenArm as robot (Bimanual)
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UsdFileCfg(
                usd_path=r"C:\Users\user\Desktop\IsaacLab\openarm_isaac_lab\source\openarm\openarm\tasks\manager_based\openarm_manipulation\usds\openarm_bimanual\openarm_bimanual.usd",
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, -0.15),
                #joint_pos={
                #    "openarm_left_joint.": 0.0,
                #    "openarm_right_joint.": 0.0,
                #    "openarm_left_finger_joint.": 0.0,
                #    "openarm_right_finger_joint.": 0.0,
                #},
                joint_pos={
                # left arm: try to reach forward/down toward the table
                    "openarm_left_joint1": 0.0,
                    "openarm_left_joint2": 0.1,
                    "openarm_left_joint3": 0.4,
                    "openarm_left_joint4": 0.6,
                    "openarm_left_joint5": 0.0,
                    "openarm_left_joint6": 0.4,
                    "openarm_left_joint7": 0.0,
                    # right arm: symmetric-ish initial pose
                    "openarm_right_joint1": 0.0,
                    "openarm_right_joint2": 0.2,
                    "openarm_right_joint3": 0.4,
                    "openarm_right_joint4": 0.6,
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.4,
                    "openarm_right_joint7": 0.0,
                    # open grippers at reset
                    "openarm_left_finger_joint.*": 0.04,
                    "openarm_right_finger_joint.*": 0.04,
                },
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["openarm_left_joint.*", "openarm_right_joint.*"],
                    effort_limit=87.0,
                    velocity_limit=2.175,
                    stiffness=80.0,
                    damping=4.0,
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["openarm_left_finger_joint.*", "openarm_right_finger_joint.*"],
                    effort_limit=20.0,
                    velocity_limit=0.2,
                    stiffness=200.0,
                    damping=5.0,
                ),
            },
        )

        # Set actions for the specific robot type (Bimanual OpenArm)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_left_joint.*",
                "openarm_right_joint.*"
            ],
            scale=0.5,
            use_default_offset=True,
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_left_finger_joint.*",
                "openarm_right_finger_joint.*"
            ],
            open_command_expr={
                "openarm_left_finger_joint.*": 0.044,
                "openarm_right_finger_joint.*": 0.044
            },
            close_command_expr={
                "openarm_left_finger_joint.*": 0.0,
                "openarm_right_finger_joint.*": 0.0
            },
        )

        # Set the body name for the end effector (Using left hand for the lift command tracking by default)
        self.commands.object_pose.body_name = "openarm_left_hand"
        self.commands.object_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # Set Glass Cup as object
        # NOTE:
        # Because glass.usd already has Rigid Body + Collider + Mass from Isaac Sim,
        # keep the Python spawn simple. Do not add rigid_props/collision_props/mass_props here first.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0, 0.055], rot=[1, 1, 0.0, 0.0]),
            spawn=UsdFileCfg(
                #coffee cup rot=[[0.7071, 0.7071, 0.0, 0.0]] and scale = (0.0002, 0.0002,0.0002)
                #usd_path=r"C:\Users\user\Desktop\isaacProject\coffeecup.usd",
                #scale=(0.0002, 0.0002, 0.0002),

                #usd_path=r"C:\Users\user\Desktop\isaacProject\highglasscup.usd",
                #scale=(0.0005, 0.0005,0.0005),

                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand",
                    name="end_effector_left",
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand",
                    name="end_effector_right",
                ),
            ],
        )


@configclass
class OpenArmCubeLiftEnvCfg_PLAY(OpenArmCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


# ─────────────────────────────────────────────────────────────────
# 階段一（Stage 1）：Reaching 環境
# 繼承完整的 Lift 環境設定，但關閉與夾取/抬起相關的獎勵項，
# 讓 DDPG 第一階段只學習讓末端效應器靠近物體。
# ─────────────────────────────────────────────────────────────────
@configclass
class OpenArmCubeReachEnvCfg(OpenArmCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- 關閉第二階段獎勵（夾取、抬起、目標追蹤）---
        # 原先的 grasping_object 獎勵（weight=10.0）→ 設為 0，不在此階段學習
        self.rewards.grasping_object.weight = 0.0
        # 原先的 lifting_object 獎勵（weight=15.0）→ 設為 0，此階段不需要抬起
        self.rewards.lifting_object.weight = 0.0
        # 原先的 object_goal_tracking 獎勵（weight=16.0）→ 設為 0
        self.rewards.object_goal_tracking.weight = 0.0
        # 原先的 object_goal_tracking_fine_grained 獎勵（weight=5.0）→ 設為 0
        self.rewards.object_goal_tracking_fine_grained.weight = 0.0

        # 只保留 reaching_object（已在父類設定 weight=1.1）
        # 以及 action_rate、joint_vel 懲罰項（維持原有值）


@configclass
class OpenArmCubeReachEnvCfg_PLAY(OpenArmCubeReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 縮小場景供播放展示使用
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # 關閉觀測值的隨機雜訊
        self.observations.policy.enable_corruption = False


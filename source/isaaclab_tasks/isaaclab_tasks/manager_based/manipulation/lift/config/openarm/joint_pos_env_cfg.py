# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.openarm.lift_openarm_env_cfg import LiftEnvCfg, ObjectTableSceneCfg

from isaaclab_assets.robots.openarm import OPENARM_BI_CFG

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class OpenArmCubeLiftSceneCfg(ObjectTableSceneCfg):
    # 訓練時預設為 None (避免 Windows usdrt Fabric 崩潰)，視覺測試與視覺訓練腳本動態掛載
    head_cam: CameraCfg | None = None
    head_cam_visual: AssetBaseCfg | None = None


@configclass
class OpenArmCubeLiftEnvCfg(LiftEnvCfg):
    scene: OpenArmCubeLiftSceneCfg = OpenArmCubeLiftSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Bimanual OpenArm as robot
        self.scene.robot = OPENARM_BI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = [0.0, 0.0, -0.2]

        # Configure right arm default joint offsets to match the elbow-up posture of the unimanual arm.
        # This is critical because use_default_offset=True adds the action (scaled by 0.5) to these offsets.
        # If left at 0.0, the arm is physically constrained to [-0.5, 0.5] radians, failing to reach the cup.
        # 擬人化右手臂預備姿態：手肘自然下垂微彎、前臂朝桌面自然延伸，絕不鎖死在天花板雞翅吊臂姿態
        self.scene.robot.init_state.joint_pos = {
            "openarm_left_joint.*": 0.0,  # left arm remains inactive and folded
            "openarm_right_joint1": 0.20, # 肩部微內收指向水杯
            "openarm_right_joint2": 0.40, # 肩部向前微傾
            "openarm_right_joint3": 0.0,  # 手肘垂直自然下垂 (解鎖 -1.57 雞翅)
            "openarm_right_joint4": 0.70, # 手肘自然微彎 40 度，前臂朝桌面延伸
            "openarm_right_joint5": 0.0,  # 手腕正向
            "openarm_right_joint6": -0.30,# 手掌維持水平對齊水杯腰身
            "openarm_right_joint7": 0.0,
            "openarm_left_finger_joint.*": 0.044,
            "openarm_right_finger_joint.*": 0.044,
        }

        # Set actions for the specific robot type (OpenArm Right Arm)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_right_joint[1-7]",
            ],
            scale=0.8,
            use_default_offset=True,
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            open_command_expr={"openarm_right_finger_joint.*": 0.044},
            close_command_expr={"openarm_right_finger_joint.*": 0.0},
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = "openarm_right_hand"
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.debug_vis = False

        # Set Glass as object (slightly further back at 0.25m, rotated -45 degrees to point handle away from approach)
        # Table top is at Z=0.0 in world coords; cup base contact is at Z=-0.0030 so it rests flush with 0 initial drop/bounce
        # 使用原廠標準無水馬克杯 (SM_Mug_D1.usd，純淨陶瓷杯、重心低、底部平穩絕不翻倒)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.25, 0.0, 0.0], rot=[0.9239, 0, 0, -0.3827]),
            spawn=UsdFileCfg(
                usd_path="C:/Users/user/Desktop/isaacProject/SM_Mug_D1.usd",
                scale=(0.007, 0.007, 0.007),
                rigid_props=RigidBodyPropertiesCfg(
                     solver_position_iteration_count=16,
                     solver_velocity_iteration_count=1,
                     max_angular_velocity=1000.0,
                     max_linear_velocity=1000.0,
                     max_depenetration_velocity=5.0,
                     disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.35),
            ),
        )

        # Listens to the required transforms (Right End Effector)
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_right_ee_tcp",
                    name="end_effector",
                ),
            ],
        )

        # Restrict observations and joint velocity rewards to cover only the active right arm
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["openarm_right_joint.*", "openarm_right_finger_joint.*"]
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["openarm_right_joint.*", "openarm_right_finger_joint.*"]
        )
        self.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["openarm_right_joint.*", "openarm_right_finger_joint.*"]
        )

        # Increase joint stiffness and damping for bimanual arm joints to prevent unstable oscillations / physics explosions (NaNs)
        self.scene.robot.actuators["openarm_arm"].stiffness = 400.0
        self.scene.robot.actuators["openarm_arm"].damping = 80.0





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

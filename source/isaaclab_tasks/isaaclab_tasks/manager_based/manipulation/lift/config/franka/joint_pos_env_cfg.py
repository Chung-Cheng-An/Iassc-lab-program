# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.kinova import KINOVA_JACO2_N6S300_CFG  # isort: skip


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Kinova as robot
        self.scene.robot = KINOVA_JACO2_N6S300_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 提供一個不像是僵硬L型、也不會蜷縮偏後的「自然前伸」姿態
        self.scene.robot.init_state.joint_pos = {
            "j2n6s300_joint_1": 0.0,
            "j2n6s300_joint_2": 2.0,  # 稍微前傾 (2.76 是極度摺疊，1.57 是水平前指)
            "j2n6s300_joint_3": 1.5,  # 順勢下垂對著桌面
            "j2n6s300_joint_4": 0.0,
            "j2n6s300_joint_5": 0.0,
            "j2n6s300_joint_6": 0.0,
            "j2n6s300_joint_finger_[1-3]": 0.2,
            "j2n6s300_joint_finger_tip_[1-3]": 0.2,
        }

        # Set actions for the specific robot type (kinova jaco2 6dof)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["j2n6s300_joint_[1-6]"], scale=0.1, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["j2n6s300_joint_finger_[1-3]", "j2n6s300_joint_finger_tip_[1-3]"],
            open_command_expr={
                "j2n6s300_joint_finger_[1-3]": 0.2,
                "j2n6s300_joint_finger_tip_[1-3]": 0.2
            },
            close_command_expr={
                "j2n6s300_joint_finger_[1-3]": 1.0,
                "j2n6s300_joint_finger_tip_[1-3]": 1.0
            },
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "j2n6s300_end_effector"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, 0.055], rot=[1, 0, 0, 0]),
            
            spawn=UsdFileCfg(
                usd_path = r"C:\Users\user\Desktop\isaacProject\SM_Mug_D1.usd",
                #usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.01, 0.01, 0.01),
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
            prim_path="{ENV_REGEX_NS}/Robot/j2n6s300_end_effector",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/j2n6s300_end_effector",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],  # 歸零，直接交給官方 end_effector 座標點
                    ),
                ),
            ],
        )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
OpenArm Bimanual Lift Cube Task with IK (Differential Inverse Kinematics) Controller.

This configuration uses the OpenArm bimanual robot. The right arm is responsible for
lifting the cube using a DifferentialIK controller (relative mode), while the left arm
also uses a DifferentialIK controller. Both grippers have binary position control.

Joint naming conventions for bimanual robot:
  - Left arm joints:   openarm_left_joint[1-7]
  - Right arm joints:  openarm_right_joint[1-7]
  - Left finger:       openarm_left_finger_joint.*
  - Right finger:      openarm_right_finger_joint.*
  - Left hand body:    openarm_left_hand
  - Right hand body:   openarm_right_hand
"""

import math

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.openarm.joint_pos_env_cfg import (
    OpenArmCubeLiftEnvCfg,
)

from isaaclab_assets.robots.openarm import OPENARM_BI_HIGH_PD_CFG

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class OpenArmBimanualCubeLiftIKEnvCfg(OpenArmCubeLiftEnvCfg):
    """
    Lift cube environment using OpenArm bimanual robot with Differential IK controller.

    The right arm performs the lift task. The left arm is also controlled via IK
    (targeting its own pose) but does not directly interact with the cube object.
    The target command is tracked by the right arm's end-effector.
    """

    def __post_init__(self):
        # Post init of parent (sets up scene, object, commands, rewards, etc.)
        super().__post_init__()

        # ------------------------------------------------------------------
        # Switch to bimanual robot with high PD gains (suitable for IK).
        # disable_gravity=True is set in OPENARM_BI_HIGH_PD_CFG for better
        # IK tracking performance.
        # ------------------------------------------------------------------
        self.scene.robot = OPENARM_BI_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Lower the robot base (Z axis), adjust the -0.2 value to lower/raise it more
        self.scene.robot.init_state.pos = [0.0, 0.0, -0.2]

        # Override object's initial position to be closer to the bimanual arms
        self.scene.object.init_state.pos = [0.2, 0.0, 0.055]

        # ------------------------------------------------------------------
        # Override arm_action: RIGHT arm uses DifferentialIK (relative mode)
        # The right arm is the one approaching and lifting the cube.
        # ------------------------------------------------------------------
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_joint[1-7]"],
            body_name="openarm_right_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.5,
        )

        # ------------------------------------------------------------------
        # Override gripper_action: right gripper only (for lifting the cube)
        # ------------------------------------------------------------------
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            open_command_expr={"openarm_right_finger_joint.*": 0.044},
            close_command_expr={"openarm_right_finger_joint.*": 0.0},
        )

        # ------------------------------------------------------------------
        # Override the command body_name to use the RIGHT hand as EE target.
        # The command pose controls where the right EE should go.
        # ------------------------------------------------------------------
        self.commands.object_pose.body_name = "openarm_right_hand"
        self.commands.object_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # ------------------------------------------------------------------
        # Override observation joint names to cover BOTH arms and fingers
        # so the policy can observe the full bimanual state.
        # ------------------------------------------------------------------
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                "openarm_left_joint.*",
                "openarm_right_joint.*",
                "openarm_left_finger_joint.*",
                "openarm_right_finger_joint.*",
            ],
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                "openarm_left_joint.*",
                "openarm_right_joint.*",
                "openarm_left_finger_joint.*",
                "openarm_right_finger_joint.*",
            ],
        )

        # ------------------------------------------------------------------
        # Override reward joint_vel penalty to cover both arms.
        # ------------------------------------------------------------------
        self.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                "openarm_left_joint.*",
                "openarm_right_joint.*",
                "openarm_left_finger_joint.*",
                "openarm_right_finger_joint.*",
            ],
        )

        # ------------------------------------------------------------------
        # Override FrameTransformer to track the RIGHT arm end-effector.
        # The source frame is the right arm's root link (openarm_right_link0),
        # and the target is the right arm's TCP frame.
        # ------------------------------------------------------------------
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


@configclass
class OpenArmBimanualCubeLiftIKEnvCfg_PLAY(OpenArmBimanualCubeLiftIKEnvCfg):
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        # Make a smaller scene for play/evaluation
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable randomization for play
        self.observations.policy.enable_corruption = False

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 3 -- Cup Grasp + Transport (combined).

The policy is trained to perform two sequential behaviours in a single PPO run:

  Phase A -- Grasp (learned first, automatically):
    Arm starts at Stage 1 converged position with gripper OPEN.
    Rewards for approaching the cup and closing the gripper drive Phase A.

  Phase B -- Transport (activated once the cup is lifted):
    Once the gripper is closed and the cup is lifted > 10 cm, the transport
    reward activates and pushes the cup to the target position on the RIGHT side.
    Target: [0.20, -0.35, 0.20] in world frame
      x = 0.20 : same forward distance
      y = -0.35 : 35 cm to the robot's right
      z = 0.20  : 20 cm above the table

    NOTE: cup_at_target uses min_height=0.10, so the policy CANNOT earn
          transport rewards without first lifting the cup.  This creates a
          natural curriculum within a single training run.

Success condition: cup reaches target zone AND is above 0.18 m (lifted + moved).

Design choices:
  - Gripper is NOT locked -- the policy controls open/close.
  - Gripper init = 0.044 (open), same as Stage 2.
  - Arm scale = 0.1: enough for both fine grasping and 35-cm lateral transport.
  - No premature-close penalty: scale=0.1 keeps arm near the cup at init.
  - Warm-started from Stage 1 converged pose (Step 1400).
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CylinderCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.openarm.joint_pos_env_cfg import OpenArmCubeLiftEnvCfg

# ?? Transport target (world frame) ??????????????????????????????????????????
#   Robot base at [0, 0, -0.2].  Cup starts at [0.25, 0.0, 0.055].
#   Target: behind the robot body (-X direction), on the table surface.
_TARGET_POS = (-0.15, -0.15, 0.02)   # Target on table surface: 15cm behind, 15cm right, z=0.02m (table level)
_VIA_POS    = (0.10, -0.35, 0.35)    # Via-point: right side of robot -- guides the arc path
_MID_POS    = (-0.02, -0.28, 0.20)   # Arc mid-point: smooth transition toward table
                                      # Helps fill in the reward gradient gap on the arc

##
# Reward configuration
##


@configclass
class CupLiftRewardsCfg:
    """Rewards for Stage 3: sequential Grasp → Lift → Arc-Transport curriculum.

    Phase A (Grasp):       reaching_object + gripper_closed
    Phase B (Lift):        lifting_object at 0.40 m -- clears robot torso
    Phase C (Transport):   cup moves along arc path to behind the robot
      Step 1: cup_at_via  -- reward for reaching right-side via-point (0.15, -0.35, 0.40)
      Step 2: cup_at_target -- reward for reaching final target (-0.20, 0.0, 0.40)

    Hard constraint (full episode): cup tilt > 45° terminates the episode.
    The via-point guides the arm to swing RIGHT first (like 「稍息」 / at-ease),
    then arc backward -- avoiding the direct path through the robot body.
    """

    # Phase A -- Grasp --------------------------------------------------------
    # NOTE: weights intentionally matched to cup_grasp_env_cfg.py which proved
    # capable of grasping. The ratio reaching:gripper_closed = 2:10 is key.

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "z_offset": 0.055},
        weight=2.0,   # Same as GRASP: keeps EE near cup without overpowering gripper signal
    )

    gripper_closed = RewTerm(
        func=mdp.gripper_closed_near_object_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"]),
            "open_val": 0.044,
            "threshold": 0.06,
            "z_offset": 0.055,
        },
        weight=10.0,  # Same as GRASP: dominant signal to close gripper on cup
    )


    # Phase B -- Lift to 0.35 m (clear table & torso before any transport) -----

    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.35},
        weight=10.0,
    )

    # Phase C -- Arc transport (only active once cup is >= 0.35 m AND gripper closed) ---

    # Step C1: via-point
    cup_at_via = RewTerm(
        func=mdp.object_at_target_if_grasped_and_lifted,
        params={
            "target_pos": _VIA_POS,
            "std": 0.15,
            "min_height": 0.35,
            "gripper_threshold": 0.02,
        },
        weight=3.0,
    )

    # Step C2: final target
    cup_at_target = RewTerm(
        func=mdp.object_at_target_if_grasped_and_lifted,
        params={
            "target_pos": _TARGET_POS,
            "std": 0.15,
            "min_height": 0.35,
            "gripper_threshold": 0.02,
        },
        weight=8.0,
    )

    cup_at_target_bonus = RewTerm(
        func=mdp.object_at_target_if_grasped_and_lifted,
        params={
            "target_pos": _TARGET_POS,
            "std": 0.08,
            "min_height": 0.35,
            "gripper_threshold": 0.02,
        },
        weight=5.0,
    )

    # Arc mid-point: fills the reward gap between via and target
    cup_at_mid = RewTerm(
        func=mdp.object_at_target_if_grasped_and_lifted,
        params={
            "target_pos": _MID_POS,
            "std": 0.15,
            "min_height": 0.35,
            "gripper_threshold": 0.02,
        },
        weight=4.0,
    )

    # Joint1 transport hints: explicitly command shoulder rotation direction (activated only when lifted+grasped)
    # Hint A: joint1 toward +1.20 rad (open-space arc toward via-point)
    joint1_hint_via = RewTerm(
        func=mdp.joint1_transport_hint,
        params={
            "target_joint1_rad": +1.20,   # Positive direction -- swings away from pillar toward open space
            "min_height": 0.35,
            "gripper_threshold": 0.02,
            "std": 0.6,
        },
        weight=3.5,   # Strong hint: directly tells joint1 to rotate in correct direction
    )

    # Hint B: joint1 toward +2.70 rad (behind robot, toward final blue target)
    joint1_hint_target = RewTerm(
        func=mdp.joint1_transport_hint,
        params={
            "target_joint1_rad": +2.70,   # Positive direction -- arm faces behind robot for final target
            "min_height": 0.35,
            "gripper_threshold": 0.02,
            "std": 0.6,
        },
        weight=3.5,   # Strong hint
    )

    # Keep cup upright -- strictly penalise tilt beyond 30°
    object_tilt = RewTerm(
        func=mdp.object_tilt_penalty,
        params={"tilt_angle_limit": 30.0},
        weight=-5.0,
    )

    # Regularisation -- RIGHT arm only
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint.*", "openarm_right_finger_joint.*"])},
    )

    # Keep LEFT arm still
    left_arm_stable = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_left_joint.*", "openarm_left_finger_joint.*"])},
    )


##
# Termination configuration
##


@configclass
class CupLiftTerminationsCfg:
    """Terminations for Stage 3."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Success: cup is within 0.15 m of the final target AND above 0.35 m
    cup_at_target = DoneTerm(
        func=mdp.cup_reached_target,
        params={
            "target_pos": _TARGET_POS,
            "threshold": 0.15,
            "min_height": 0.35,
        },
    )



##
# Environment configuration
##


@configclass
class CupLiftEnvCfg(OpenArmCubeLiftEnvCfg):
    """Stage 3: combined Grasp + Transport task.

    The policy learns to:
      1. Close the gripper on the cup (Phase A, same as Stage 2).
      2. Once gripped, carry the cup to the right-side target (Phase B).

    Because cup_at_target only activates when the cup is above 10 cm, the
    policy MUST learn Phase A before Phase B rewards become available.
    This creates a natural 2-phase curriculum within a single PPO run.
    """

    def __post_init__(self):
        super().__post_init__()

        self.rewards = CupLiftRewardsCfg()
        self.terminations = CupLiftTerminationsCfg()

        # 20-second episodes: extended to cover 3-phase journey
        #   Phase A (grasp): ~4s, Phase B (lift): ~4s, Phase C (arc transport): ~12s
        self.episode_length_s = 20.0

        # Arm scale = 0.2 (compromise):
        #   - Allows joint deviations of ±0.8 rad (enough to reach via-point)
        #   - Small enough to avoid knocking the cup at episode start
        #   - Still 2x larger than original 0.1 for transport reach
        self.actions.arm_action.scale = 0.1




        # Gripper init = OPEN (0.044), same as Stage 2.
        # The policy must learn to close it (Phase A) before transport rewards activate.
        self.scene.robot.init_state.joint_pos = {
            "openarm_left_joint1": -1.57,
            "openarm_left_joint2": 0.0,
            "openarm_left_joint3": -1.57,
            "openarm_left_joint4": 1.57,
            "openarm_left_joint[5-7]": 0.0,
            "openarm_right_joint1": 0.1893,  # Stage 1 converged (Step 1400)
            "openarm_right_joint2": -0.0157,
            "openarm_right_joint3": -1.5705,  # Validator patched to warn-only
            "openarm_right_joint4": 0.3572,
            "openarm_right_joint5": 1.1688,
            "openarm_right_joint6": -0.2474,
            "openarm_right_joint7": 1.1162,
            "openarm_left_finger_joint.*": 0.044,
            "openarm_right_finger_joint.*": 0.044,  # OPEN -- policy must learn to close (Phase A)
        }

        # Gripper action is BINARY and NOT locked:
        #   policy outputs > 0 -> open (0.044)
        #   policy outputs < 0 -> close (0.0)
        # (No override of open_command_expr -- leave at default 0.044)

        # High gripper stiffness/damping: stable grip under gravity + lateral motion
        self.scene.robot.actuators["openarm_gripper"].stiffness = 5e3
        self.scene.robot.actuators["openarm_gripper"].damping = 3e2

        # Joint limits: relaxed to 1.5 to allow large transport swings.
        # factor=1.5 gives enough range for the arm to swing ~90° to reach behind
        # the robot, while still providing soft-limit protection.
        self.scene.robot.soft_joint_pos_limit_factor = 2.0


        # Self-collision: DISABLED by default (same as OPENARM_BI_CFG default).
        # The base pillar penetration in GRASP was a collision mesh geometry issue in the USD,
        # NOT a self-collision issue. Enabling self-collision only affects arm link vs arm link
        # (e.g., link4 vs link1), not arm vs base pillar (which is environment collision).
        # self.scene.robot.spawn.articulation_props.enabled_self_collisions = False

        # Increase solver iterations to reduce high-speed penetration artifacts.
        # Default is 8 position / 0 velocity; bumping to 16/1 gives PhysX more
        # correction passes per step, which noticeably reduces clipping at high velocities.
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 16
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 1

        # Increase max_depenetration_velocity so PhysX can push overlapping bodies apart faster.
        # Default 5.0 m/s is sometimes too slow to recover when the arm moves quickly.
        self.scene.robot.spawn.rigid_props.max_depenetration_velocity = 10.0

        # ── Visual position markers (purely decorative, AssetBaseCfg = zero PhysX cost) ──
        # Red disk  : cup start position [0.25, 0.00]  on table
        # Green disk: via-point         [0.15, -0.35]  on table
        # Blue disk : final target      [-0.20, 0.00]  on table
        # NOTE: AssetBaseCfg creates a static USD prim with NO rigid body / NO PhysX object.
        # Previously used RigidObjectCfg (kinematic) which created 4096×3=12,288 PhysX actors
        # and caused 3+ minute startup delays. Static prims have zero PhysX overhead.
        self.scene.marker_via = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/MarkerVia",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.15, -0.35, 0.002)),
            spawn=CylinderCfg(
                radius=0.04,
                height=0.003,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 1.0, 0.15)),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
        )
        self.scene.marker_target = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/MarkerTarget",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.20, -0.15, 0.002)),
            spawn=CylinderCfg(
                radius=0.04,
                height=0.003,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 1.0)),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
        )


@configclass
class CupLiftEnvCfg_PLAY(CupLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

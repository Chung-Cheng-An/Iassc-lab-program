"""
test_arm_back_spin.py
Test script that smoothly rotates OpenArm Shoulder Joint 1 (openarm_right_joint1) 180 degrees to the back.

Run with:
    .\\isaaclab.bat -p scripts\\test_arm_back_spin.py
"""

import argparse
import math
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test OpenArm Shoulder Joint 1 180-Degree Rotation")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.lift.config.openarm.cup_lift_back_env_cfg import CupLiftBackEnvCfg_PLAY


def main():
    cfg = CupLiftBackEnvCfg_PLAY()
    cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg)

    robot = env.scene["robot"]
    obj = env.scene["object"]

    print("\n" + "=" * 60)
    print("OpenArm Shoulder Joint 1 (180-Degree Rotation) Test Initialized!")
    print(f"Robot Root Pos (World): {robot.data.root_pos_w[0].tolist()}")
    print(f"End-Effector Pos       : {env.scene['ee_frame'].data.target_pos_w[0, 0].tolist()}")
    print(f"Cup Root Pos (World)  : {obj.data.root_pos_w[0].tolist()}")
    print("=" * 60 + "\n")

    step_count = 0
    obs, _ = env.reset()

    # Action dimension: 8 (7 right arm joints + 1 gripper)
    action = torch.zeros((1, 8), device=env.device)

    while simulation_app.is_running():
        # Smoothly ramp Joint 1 target angle from 0 to 180 degrees (pi rad) over 300 steps, hold, then return
        cycle = (step_count % 600) / 600.0  # 0.0 -> 1.0 -> 0.0
        if cycle < 0.5:
            # Ramping from 0 to pi (180 deg)
            target_angle = (1.0 - math.cos(cycle * 2.0 * math.pi)) * 0.5 * math.pi
        else:
            # Returning from pi to 0
            target_angle = (1.0 - math.cos((1.0 - cycle) * 2.0 * math.pi)) * 0.5 * math.pi

        # Action scale for arm_action is 0.2, so command = target_angle / 0.2
        action[0, 0] = target_angle / 0.2

        obs, reward, terminated, truncated, _ = env.step(action)
        step_count += 1

        if step_count % 30 == 0:
            j1_pos = robot.data.joint_pos[0, robot.find_joints("openarm_right_joint1")[0][0]].item()
            ee_z = env.scene['ee_frame'].data.target_pos_w[0, 0, 2].item()
            cup_z = obj.data.root_pos_w[0, 2].item()
            print(
                f"Step {step_count:4d} | Target J1 Angle: {math.degrees(target_angle):.1f}° | "
                f"Actual J1 Angle: {math.degrees(j1_pos):.1f}° | "
                f"EE Height: {ee_z:.3f} m | Cup Height: {cup_z:.3f} m"
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

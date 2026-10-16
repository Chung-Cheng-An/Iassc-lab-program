"""
debug_camera_pose.py
===================
深入排查相機鏡頭拍攝到的真實物體與世界座標。
"""

import argparse
import sys
import os

if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug camera pose and render output")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
carb.settings.get_settings().set("/isaaclab/cameras_enabled", True)

import numpy as np
import torch
import gymnasium as gym
from PIL import Image

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

IMG_DIR = r"C:\Users\user\Desktop\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation\lift\config\openarm"


def main():
    task_name = "Isaac-Cup-Grasp-OpenArm-Play-v0"

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1

    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    obj   = env.unwrapped.scene["object"]
    cam   = env.unwrapped.scene["head_cam"]

    print("\n" + "=" * 70)
    print("  [CAMERA DEBUG INFO]")
    print(f"  Camera Prim Path   : {cam.cfg.prim_path}")
    print(f"  Camera Data Types  : {cam.cfg.data_types}")
    print("=" * 70 + "\n")

    for step in range(50):
        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(action)

        cam_pos_w  = cam.data.pos_w[0].cpu().tolist()
        cam_quat_w = cam.data.quat_w_world[0].cpu().tolist()
        obj_pos_w  = obj.data.root_pos_w[0].cpu().tolist()
        robot_pos_w= robot.data.root_pos_w[0].cpu().tolist()

        if step == 25:
            print(f"[Step {step}] Camera World Pos  : {cam_pos_w}")
            print(f"[Step {step}] Camera World Quat : {cam_quat_w}")
            print(f"[Step {step}] Robot World Pos   : {robot_pos_w}")
            print(f"[Step {step}] Object World Pos  : {obj_pos_w}")

            if "rgb" in cam.data.output:
                rgb_np = cam.data.output["rgb"][0].cpu().numpy()
                img = Image.fromarray(rgb_np[..., :3].astype(np.uint8))
                out_path = os.path.join(IMG_DIR, "debug_cam_step25.png")
                img.save(out_path)
                print(f"\n[SUCCESS] 已寫入除錯圖片: {out_path}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

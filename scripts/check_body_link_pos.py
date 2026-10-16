"""
check_body_link_pos.py
找出 openarm_body_link 在世界座標中的實際位置
執行: .\isaaclab.bat -p scripts\check_body_link_pos.py
"""
import argparse
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

def main():
    task_name = "Isaac-Cup-Lift-OpenArm-v0"
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5

    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    robot = env.unwrapped.scene["robot"]

    body_names = robot.data.body_names
    print("\n" + "=" * 60)
    print("  Robot Body Links & World Positions")
    print("=" * 60)
    for i, name in enumerate(body_names):
        pos_w = robot.data.body_pos_w[0, i].cpu().tolist()
        print(f"  [{i:2d}] {name:<40s} world=({pos_w[0]:6.3f}, {pos_w[1]:6.3f}, {pos_w[2]:6.3f})")
    print("=" * 60)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()

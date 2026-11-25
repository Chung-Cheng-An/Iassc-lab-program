import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg

def main():
    env_cfg = parse_env_cfg("Isaac-Cup-PickPlace-OpenArm-Play-v0", device="cuda:0", num_envs=1)
    env_cfg.scene.head_cam = None
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    env = gym.make("Isaac-Cup-PickPlace-OpenArm-Play-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    obs = env.get_observations()

    ckpt_path = r"logs\rsl_rl\openarm_lift\2026-11-09_12-01-00\model_11097.pt"
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    runner_cfg = load_cfg_from_registry("Isaac-Cup-PickPlace-OpenArm-Play-v0", "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["object"]

    target_xy = np.array([0.16, -0.28])
    has_released = False
    release_arm_action = None

    for step in range(200):
        cup_w = obj.data.root_pos_w[0, :3].cpu().numpy()
        robot_root = robot.data.root_pos_w[0, :3].cpu().numpy()
        cup_local_xy = cup_w[:2] - robot_root[:2]
        dist_to_plat = np.linalg.norm(cup_local_xy - target_xy)

        qx = obj.data.root_quat_w[0, 1].item()
        qy = obj.data.root_quat_w[0, 2].item()
        z_proj = np.clip(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0)
        tilt_deg = float(np.degrees(np.arccos(z_proj)))

        with torch.inference_mode():
            action = policy(obs).clone()

            # 🎯 解決夾杯底問題: 在靠近與夾取階段 (step 10~45)，肩部抬高 -0.15，手肘微提，直擊杯腰 (高度 +4cm)
            if 10 <= step < 45:
                action[0, 1] -= 0.15  # 抬高肩關節，避開杯底錐形
                action[0, 3] -= 0.10  # 協同微調前臂

            # 搬運階段手腕水準鎖定
            if 45 <= step < 70:
                action[0, 4] += np.clip(1.8 * qx, -0.4, 0.4)
                action[0, 5] -= np.clip(1.8 * qy, -0.4, 0.4)

            # 抵達杯墊正上方 (距離 < 6cm，高度 < 2.5cm，step > 65)，鎖定放手！
            if (step > 65 and dist_to_plat < 0.06 and cup_w[2] < 0.025) or has_released:
                if not has_released:
                    has_released = True
                    release_arm_action = action[:, :7].clone()

                action[:, :7] = release_arm_action.clone()  # 絕對凍結手臂！
                action[0, 7] = 1.0  # 永遠維持全開夾爪！

        if step % 10 == 0 or (step >= 20 and step <= 50):
            grip_cmd = action[0, 7].item()
            print(f"Step {step:3d} | Z: {cup_w[2]*100:4.1f}cm | Dist: {dist_to_plat*100:4.1f}cm | Tilt: {tilt_deg:4.1f} deg | Grip: {grip_cmd:.1f}", flush=True)

        obs, _, dones, _ = env.step(action)

        if step % 10 == 0 or (step >= 70 and step <= 110):
            grip_cmd = action[0, 7].item()
            print(f"Step {step:3d} | Z: {cup_w[2]*100:4.1f}cm | Dist: {dist_to_plat*100:4.1f}cm | Tilt: {tilt_deg:4.1f} deg | Grip: {grip_cmd:.1f}", flush=True)

        obs, _, dones, _ = env.step(action)

    simulation_app.close()

if __name__ == "__main__":
    main()

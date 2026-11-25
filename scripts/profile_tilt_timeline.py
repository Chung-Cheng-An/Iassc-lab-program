import os, sys, glob, torch
import gymnasium as gym
from isaaclab.app import AppLauncher
app = AppLauncher({"headless": True}).app

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

task_name = "Isaac-Cup-PickPlace-OpenArm-Play-v0"
env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
env_cfg.scene.num_envs = 16
env = gym.make(task_name, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)
obs = env.get_observations()

runner = OnPolicyRunner(env, load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point").to_dict(), log_dir=None, device="cuda:0")
runner.load("logs/rsl_rl/openarm_lift/2026-11-09_12-01-00/model_11097.pt")
policy = runner.get_inference_policy(device="cuda:0")

robot = env.unwrapped.scene["robot"]
obj = env.unwrapped.scene["object"]

target_xy = torch.tensor([0.16, -0.28], device="cuda:0")

check_steps = [30, 60, 90, 130, 180, 220, 260, 295]
tilt_history = {s: [] for s in check_steps}
z_history = {s: [] for s in check_steps}
dist_history = {s: [] for s in check_steps}

for step in range(300):
    with torch.inference_mode():
        actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    
    pos = obj.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    quat = obj.data.root_quat_w
    qx, qy = quat[:, 1], quat[:, 2]
    tilt = torch.rad2deg(torch.acos(torch.clamp(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0)))
    dist = torch.norm(pos[:, :2] - target_xy, dim=1) * 100.0
    z = pos[:, 2] * 100.0

    if step in check_steps:
        tilt_history[step] = tilt.cpu().tolist()
        z_history[step] = z.cpu().tolist()
        dist_history[step] = dist.cpu().tolist()

print("\n" + "=" * 70, flush=True)
print("  📍 【水杯傾角與高度隨時間全週期演變 (16 個環境平均值)】", flush=True)
print("=" * 70, flush=True)
for s in check_steps:
    mean_t = sum(tilt_history[s]) / len(tilt_history[s])
    mean_z = sum(z_history[s]) / len(z_history[s])
    mean_d = sum(dist_history[s]) / len(dist_history[s])
    toppled_count = sum(1 for t in tilt_history[s] if t > 25.0)
    print(f"  Step {s:3d}: 傾角均值={mean_t:4.1f}° | 高度均值={mean_z:4.1f}cm | 距目標={mean_d:4.1f}cm | 已側傾超過25°的環境數: {toppled_count}/16", flush=True)
print("=" * 70 + "\n", flush=True)

env.close()
app.close()


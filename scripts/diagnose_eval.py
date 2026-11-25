import os, sys, glob, torch
import gymnasium as gym
from isaaclab.app import AppLauncher
app = AppLauncher({"headless": True}).app

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

task_name = "Isaac-Cup-PickPlace-OpenArm-Play-v0"
env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
env_cfg.scene.num_envs = 4
env = gym.make(task_name, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)
obs = env.get_observations()

runner = OnPolicyRunner(env, load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point").to_dict(), log_dir=None, device="cuda:0")
runner.load("logs/rsl_rl/openarm_lift/2026-11-09_12-01-00/model_11097.pt")
policy = runner.get_inference_policy(device="cuda:0")

robot = env.unwrapped.scene["robot"]
obj = env.unwrapped.scene["object"]

target_xy = torch.tensor([0.16, -0.28], device="cuda:0")

for step in range(300):
    with torch.inference_mode():
        actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    pos = obj.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    quat = obj.data.root_quat_w
    qx, qy = quat[:, 1], quat[:, 2]
    tilt_deg = torch.rad2deg(torch.acos(torch.clamp(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0)))
    dist_cm = torch.norm(pos[:, :2] - target_xy, dim=1) * 100.0

    if step in [0, 40, 75, 120, 180, 230, 270, 299]:
        print(f"Step {step:3d} | Env 0: Z={pos[0, 2]*100:.1f}cm, Tilt={tilt_deg[0]:.1f}deg, Dist={dist_cm[0]:.1f}cm, Grip={actions[0, 7]:.2f}")

print("\nFinal Results for 4 envs:")
for i in range(4):
    print(f"Env {i}: Pos=[{pos[i, 0]:.3f}, {pos[i, 1]:.3f}, {pos[i, 2]:.3f}], Tilt={tilt_deg[i]:.1f}deg, Dist={dist_cm[i]:.1f}cm, Grip={actions[i, 7]:.2f}")

env.close()
app.close()


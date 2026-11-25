import glob, torch, os, sys
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

ckpts = [
    r"logs\rsl_rl\openarm_lift\2026-11-09_20-24-23\model_12150.pt",
    r"logs\rsl_rl\openarm_lift\2026-11-09_20-24-23\model_12250.pt",
    r"logs\rsl_rl\openarm_lift\2026-11-09_20-24-23\model_12400.pt",
    r"logs\rsl_rl\openarm_lift\2026-11-09_20-24-23\model_12500.pt",
    r"logs\rsl_rl\openarm_lift\2026-11-09_12-01-00\model_11097.pt",
]

robot = env.unwrapped.scene["robot"]
obj = env.unwrapped.scene["object"]
target_xy = torch.tensor([0.16, -0.28], device="cuda:0")

print("\n" + "="*75, flush=True)
print("🔥 CHECKPOINT BENCHMARK (16 Envs, 120 steps)", flush=True)
print("="*75, flush=True)

for ckpt in ckpts:
    if not os.path.exists(ckpt):
        print(f"Skipping {ckpt} (not found)", flush=True)
        continue
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(ckpt)
    policy = runner.get_inference_policy(device="cuda:0")
    
    obs, _ = env.reset()
    
    for step in range(120):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        
    pos = obj.data.root_pos_w[:, :3] - robot.data.root_pos_w[:, :3]
    quat = obj.data.root_quat_w
    qx, qy = quat[:, 1], quat[:, 2]
    tilt_deg = torch.rad2deg(torch.acos(torch.clamp(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0)))
    dist_cm = torch.norm(pos[:, :2] - target_xy, dim=1) * 100.0
    
    succ = ((dist_cm <= 12.0) & (pos[:, 2] <= 0.035) & (tilt_deg <= 20.0)).sum().item()
    rate = succ / 16.0 * 100.0
    print(f"  • {os.path.basename(ckpt):15s} | 成功率: {succ:2d}/16 ({rate:5.1f}%) | 距目標: {dist_cm.mean().item():4.1f}cm | 傾角: {tilt_deg.mean().item():4.1f}° | 高度: {pos[:, 2].mean().item()*100:4.1f}cm", flush=True)

print("="*75 + "\n", flush=True)
env.close()
app.close()


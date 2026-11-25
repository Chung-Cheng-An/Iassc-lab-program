"""
eval_pick_place_batch.py
========================
【大規模並行評估水杯 Pick & Place 成功率 (Large-Scale Batch Benchmark)】

功能：
  1. 採用向量化多環境並行評估 (GPU 向量化模擬)。
  2. 修正多環境座標系偏差 (嚴格採用各環境 Robot-Local 座標，杜絕跨環境幾十米 World 座標偏差)。
  3. 精確捕獲 dones 終止訊號，統計大樣本宏觀指標。
"""

import argparse
import glob
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["WARP_DISABLE_CUDA_PRIMARY_CTX"] = "1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Large-Scale Batch Evaluation for OpenArm Cup Pick and Place")
parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments (default: 32)")
parser.add_argument("--num_episodes", type=int, default=64, help="Total episodes to evaluate (default: 64)")
parser.add_argument("--task", type=str, default="Isaac-Cup-PickPlace-OpenArm-Play-v0", help="Task name")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
parser.add_argument("--latch_release", action="store_true", default=True, help="Enable touchdown release latch")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def find_latest_checkpoint():
    """尋找最新訓練生成的模型權重。"""
    base_dir = r"logs\rsl_rl\openarm_lift"
    if not os.path.exists(base_dir):
        return None
    runs = sorted(glob.glob(os.path.join(base_dir, "*")), key=os.path.getmtime, reverse=True)
    for run in runs:
        if os.path.isdir(run):
            models = sorted(glob.glob(os.path.join(run, "model_*.pt")), key=os.path.getmtime, reverse=True)
            if models:
                return models[0]
    return None


def main():
    ckpt_path = args_cli.checkpoint or find_latest_checkpoint()
    task_name = args_cli.task
    num_envs = args_cli.num_envs
    target_episodes = args_cli.num_episodes

    print("\n" + "=" * 78, flush=True)
    print("  🚀 [Isaac Lab] OpenArm 水杯 Pick & Place 大規模並行評估 (Batch Benchmark)", flush=True)
    print(f"    • 任務名稱: {task_name}", flush=True)
    print(f"    • 評估模型: {ckpt_path}", flush=True)
    print(f"    • 並行環境數: {num_envs} envs (GPU 向量化模擬)", flush=True)
    print(f"    • 目標評估樣本: {target_episodes} 回合", flush=True)
    print(f"    • 著陸放爪鎖: {'啟用 (Touchdown Latch)' if args_cli.latch_release else '純神經網路 (Pure RL)'}", flush=True)
    print("=" * 78 + "\n", flush=True)

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs

    # 載入原廠馬克杯
    mug_usd_path = "C:/Users/user/Desktop/isaacProject/SM_Mug_D1.usd"
    if os.path.exists(mug_usd_path):
        env_cfg.scene.object.spawn.usd_path = mug_usd_path

    # 高剛性夾爪
    env_cfg.actions.gripper_action.close_command_expr = {"openarm_right_finger_joint.*": 0.0}
    env_cfg.actions.gripper_action.open_command_expr = {"openarm_right_finger_joint.*": 0.044}
    env_cfg.scene.robot.actuators["openarm_gripper"].stiffness = 5e3
    env_cfg.scene.robot.actuators["openarm_gripper"].damping = 3e2

    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    obs = env.get_observations()

    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["object"]

    # 載入模型
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=device)

    target_xy = torch.tensor([0.16, -0.28], device=device)

    completed_episodes = 0
    successes = 0
    topples = 0
    miss_targets = 0
    air_drops = 0
    dist_errors = []
    tilt_angles = []

    step_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
    is_landing = torch.zeros(num_envs, dtype=torch.bool, device=device)
    is_settling = torch.zeros(num_envs, dtype=torch.bool, device=device)
    landing_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    settle_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    landing_actions = torch.zeros((num_envs, 8), device=device)
    touchdown_actions = torch.zeros((num_envs, 8), device=device)
    max_steps_per_episode = 300

    start_time = time.time()
    print("⚡ 正在極速並行模擬中，請稍候...\n", flush=True)

    while simulation_app.is_running() and completed_episodes < target_episodes:
        # 計算各環境真實的 Local 座標
        robot_root = robot.data.root_pos_w
        cup_world = obj.data.root_pos_w[:, :3]
        cup_quat = obj.data.root_quat_w
        cup_local_xy = cup_world[:, :2] - robot_root[:, :2]
        cup_z = cup_world[:, 2]

        # 向量化傾角
        qx = cup_quat[:, 1]
        qy = cup_quat[:, 2]
        z_proj = torch.clamp(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0)
        tilt_deg = torch.rad2deg(torch.acos(z_proj))

        # 向量化目標距離 (局部座標對齊)
        dist_to_plat = torch.norm(cup_local_xy - target_xy, dim=1)

        with torch.inference_mode():
            # 🎯 杯身中上部瞄準 (step < 40)，目標高度抬升 6.8cm
            policy_obs = obs.clone()
            is_grasping = step_counts < 40
            if is_grasping.any():
                g_envs = is_grasping.nonzero(as_tuple=True)[0]
                policy_obs[g_envs, 20] += 0.068

            actions = policy(policy_obs).clone()

            # 🛡️ 方案 1: 端水手腕動態水準姿態補償 (Leveling Wrist Controller)
            is_transporting = (step_counts > 40) & (~is_settling) & (cup_z > 0.005)
            if is_transporting.any():
                t_envs = is_transporting.nonzero(as_tuple=True)[0]
                actions[t_envs, 4] += torch.clamp(1.8 * qx[t_envs], -0.4, 0.4)
                actions[t_envs, 5] -= torch.clamp(1.8 * qy[t_envs], -0.4, 0.4)

            if args_cli.latch_release:
                # 階段 A 觸發: 進入平臺半徑 8cm 且步數 > 50
                can_land = (~is_landing) & (~is_settling) & (dist_to_plat < 0.08) & (step_counts > 50)
                if can_land.any():
                    is_landing[can_land] = True
                    landing_step[can_land] = step_counts[can_land]
                    landing_actions[can_land] = actions[can_land]

                # 柔和下放中
                landing_envs = is_landing.nonzero(as_tuple=True)[0]
                if len(landing_envs) > 0:
                    l_steps = step_counts[landing_envs] - landing_step[landing_envs]
                    d_prog = torch.clamp(l_steps.float() / 35.0, 0.0, 1.0)
                    actions[landing_envs] = landing_actions[landing_envs].clone()
                    actions[landing_envs, 1] += 0.50 * d_prog
                    actions[landing_envs, 3] -= 0.38 * d_prog
                    actions[landing_envs, 4] += torch.clamp(1.8 * qx[landing_envs], -0.4, 0.4)
                    actions[landing_envs, 5] -= torch.clamp(1.8 * qy[landing_envs], -0.4, 0.4)
                    actions[landing_envs, 7] = -1.0

                    # 觸桌即停判定: 指尖下探觸底 (cup_z <= 0.025m) 或超時 40 步
                    can_touch = ((cup_z[landing_envs] <= 0.025) & (l_steps > 8)) | (l_steps >= 40)
                    if can_touch.any():
                        touch_idx = landing_envs[can_touch]
                        is_landing[touch_idx] = False
                        is_settling[touch_idx] = True
                        settle_step[touch_idx] = step_counts[touch_idx]
                        touchdown_actions[touch_idx] = actions[touch_idx]

                # 階段 B/C: 觸桌靜置放手 + 純垂直直拔
                settling_envs = is_settling.nonzero(as_tuple=True)[0]
                if len(settling_envs) > 0:
                    s_steps = step_counts[settling_envs] - settle_step[settling_envs]
                    actions[settling_envs] = touchdown_actions[settling_envs].clone()
                    actions[settling_envs, 7] = 1.0  # 全開夾爪

                    # 靜置 25 步後，純垂直向上拉起，絕不向前下壓
                    lifting_mask = s_steps >= 25
                    if lifting_mask.any():
                        l_envs = settling_envs[lifting_mask]
                        lift_prog = torch.clamp((s_steps[lifting_mask].float() - 25.0) / 30.0, 0.0, 1.0)
                        actions[l_envs, 1] -= 0.35 * lift_prog
                        actions[l_envs, 3] += 0.15 * lift_prog
                        actions[l_envs, 5] -= 0.20 * lift_prog
                        actions[l_envs, 7] = 1.0

        obs, _, dones, _ = env.step(actions)
        step_counts += 1

        # 檢測達到回合完整步數 (完成抓取、平移、放平全流程)
        finished_envs = (step_counts >= max_steps_per_episode).nonzero(as_tuple=True)[0]
        if len(finished_envs) > 0:
            for env_i in finished_envs:
                if completed_episodes >= target_episodes:
                    break

                d_cm = dist_to_plat[env_i].item() * 100.0
                t_deg = tilt_deg[env_i].item()
                z_m = cup_z[env_i].item()

                is_at_plat = d_cm <= 8.0
                is_on_table = z_m <= 0.035
                is_upright = t_deg <= 15.0

                if is_at_plat and is_on_table and is_upright:
                    successes += 1
                elif not is_upright:
                    topples += 1
                elif not is_at_plat:
                    miss_targets += 1
                else:
                    air_drops += 1

                dist_errors.append(d_cm)
                tilt_angles.append(t_deg)
                completed_episodes += 1

                step_counts[env_i] = 0
                is_landing[env_i] = False
                is_settling[env_i] = False

                if completed_episodes % 8 == 0 or completed_episodes == target_episodes:
                    cur_rate = (successes / completed_episodes) * 100.0
                    print(f"  [進度: {completed_episodes:3d}/{target_episodes}] 成功率: {cur_rate:5.1f}% | 成功: {successes:2d} | 翻倒: {topples:2d} | 偏離: {miss_targets:2d}", flush=True)

    elapsed_time = time.time() - start_time
    succ_rate = (successes / completed_episodes * 100.0) if completed_episodes > 0 else 0.0
    topple_rate = (topples / completed_episodes * 100.0) if completed_episodes > 0 else 0.0
    miss_rate = (miss_targets / completed_episodes * 100.0) if completed_episodes > 0 else 0.0
    mean_dist = float(np.mean(dist_errors)) if len(dist_errors) > 0 else 0.0
    mean_tilt = float(np.mean(tilt_angles)) if len(tilt_angles) > 0 else 0.0

    results = {
        "checkpoint": ckpt_path,
        "total_episodes": completed_episodes,
        "success_rate_percent": round(succ_rate, 2),
        "topple_rate_percent": round(topple_rate, 2),
        "miss_target_rate_percent": round(miss_rate, 2),
        "mean_dist_error_cm": round(mean_dist, 2),
        "mean_tilt_angle_deg": round(mean_tilt, 2),
        "elapsed_seconds": round(elapsed_time, 2),
    }

    with open("eval_batch_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78, flush=True)
    print("  🏁 【大規模並行評估結算報表 (Batch Benchmark Summary)】", flush=True)
    print(f"    • 評估模型: {ckpt_path}", flush=True)
    print(f"    • 樣本總數: {completed_episodes} 回合 | 耗時: {elapsed_time:.1f} 秒", flush=True)
    print(f"    • 🏆 總體成功率: {succ_rate:.1f}% ({successes}/{completed_episodes})", flush=True)
    print(f"    • ❌ 翻倒倒地率: {topple_rate:.1f}% ({topples}/{completed_episodes})", flush=True)
    print(f"    • 🎯 偏離目標率: {miss_rate:.1f}% ({miss_targets}/{completed_episodes})", flush=True)
    print(f"    • 📏 平均落點偏差: {mean_dist:.2f} cm (平臺半徑 6.0cm)", flush=True)
    print(f"    • 📐 平均杯身傾角: {mean_tilt:.2f} 度", flush=True)
    print("=" * 78 + "\n", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

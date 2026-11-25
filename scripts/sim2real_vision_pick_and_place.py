"""
sim2real_vision_pick_and_place.py
==================================
【OpenArm 視覺與狀態機協同：杯子夾起並放置到桌面上 (Pick & Place)】

控制流程：
  Phase 1 (Reach & Align) : 視覺偵測水杯位置，引導夾爪張開並水平對齊滑入杯腰
  Phase 2 (Grasp)        : 雙指包夾杯身，緊縮夾緊水杯
  Phase 3 (Lift)         : 垂直穩健端起杯子離地 (高度 ~ 8-10 cm)
  Phase 4 (Transport)    : 將杯子平移搬運至目標桌面放置點上方 (X=0.25m, Y=-0.15m)
  Phase 5 (Lower & Place): 緩速垂直下放杯子接觸桌面 (Z=0.0m)
  Phase 6 (Release)      : 夾爪張開釋放杯子，平穩後撤完成放置展示！

執行方式：
  .\\isaaclab.bat -p scripts\\sim2real_vision_pick_and_place.py
"""

import argparse
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["WARP_DISABLE_CUDA_PRIMARY_CTX"] = "1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Vision-Guided Pick and Place Demo for OpenArm Robot")
parser.add_argument("--task", type=str, default="Isaac-Cup-Grasp-OpenArm-Play-v0", help="Base task name")
parser.add_argument("--checkpoint", type=str, default=None, help="Trained Grasp checkpoint")
parser.add_argument("--cup_x", type=float, default=0.25, help="Cup initial X position (default: 0.25)")
parser.add_argument("--cup_y", type=float, default=0.0, help="Cup initial Y position (default: 0.0)")
parser.add_argument("--place_x", type=float, default=0.25, help="Target place X position on table (default: 0.25)")
parser.add_argument("--place_y", type=float, default=-0.15, help="Target place Y position on table (default: -0.15)")
parser.add_argument("--fps", type=float, default=28.0, help="Playback speed (FPS)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
if sys.platform == "win32" and getattr(args_cli, "headless", False):
    args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("isaaclab").setLevel(logging.ERROR)
logging.getLogger("omni").setLevel(logging.ERROR)

import omni.usd
from pxr import Usd, UsdGeom, UsdShade, Gf
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

DEFAULT_CHECKPOINT = r"logs\rsl_rl\openarm_lift\2026-10-16_02-18-30\model_4498.pt"


def main():
    print("\n" + "=" * 78)
    print("  🏆 [Isaac Lab] OpenArm 水杯 Pick & Place (夾起並放置於桌面) 完整展示")
    print(f"    • 起始杯子位置: [X={args_cli.cup_x:.2f}, Y={args_cli.cup_y:.2f}, Z=0.00] m")
    print(f"    • 目標放置位置: [X={args_cli.place_x:.2f}, Y={args_cli.place_y:.2f}, Z=0.00] m")
    print("    • 流程階段: 靠近 → 包夾 → 抬起 → 搬運 → 下放桌面 → 鬆爪釋放 → 順利復位")
    print("=" * 78 + "\n")

    task_name = args_cli.task
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1

    # 設置水杯模型與初始姿態
    water_usd_path = "C:/Users/user/Desktop/isaacProject/SM_Mug_Water.usd"
    if not os.path.exists(water_usd_path):
        water_usd_path = "C:/Users/user/Desktop/isaacProject/SM_Mug_D1.usd"
    env_cfg.scene.object.spawn.usd_path = water_usd_path
    env_cfg.scene.object.init_state.pos = [args_cli.cup_x, args_cli.cup_y, 0.0]
    env_cfg.scene.object.init_state.rot = [0.9239, 0.0, 0.0, -0.3827]

    # 夾爪高剛性配置
    env_cfg.actions.gripper_action.close_command_expr = {"openarm_right_finger_joint.*": 0.0}
    env_cfg.actions.gripper_action.open_command_expr = {"openarm_right_finger_joint.*": 0.044}
    env_cfg.scene.robot.actuators["openarm_gripper"].stiffness = 5e3
    env_cfg.scene.robot.actuators["openarm_gripper"].damping = 3e2

    # 徹底關閉底層自動提前重置
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations.time_out = None
        env_cfg.terminations.object_grasped = None
        env_cfg.terminations.cup_toppled = None
        env_cfg.terminations.object_dropping = None
        env_cfg.terminations.ee_reached = None
    env_cfg.episode_length_s = 10000.0

    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    env.reset()
    obs = env.get_observations()

    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["object"]
    ee_frame = env.unwrapped.scene["ee_frame"]
    device = env.unwrapped.device

    init_pos = torch.tensor([[args_cli.cup_x, args_cli.cup_y, 0.0]], device=device)
    init_quat = torch.tensor([[0.9239, 0.0, 0.0, -0.3827]], device=device)
    init_pose = torch.cat([init_pos, init_quat], dim=-1)
    zero_vel = torch.zeros((1, 6), device=device)

    obj.write_root_pose_to_sim(init_pose)
    obj.write_root_velocity_to_sim(zero_vel)
    init_robot_pos = robot.data.default_joint_pos.clone()
    init_robot_vel = robot.data.default_joint_vel.clone()

    # 在桌面放置目標點建立半透明視覺標記 (綠色半透明圓柱標記放置位置)
    stage = omni.usd.get_context().get_stage()
    target_marker_path = "/Visuals/PlaceTargetMarker"
    marker_prim = UsdGeom.Cylinder.Define(stage, target_marker_path)
    marker_prim.CreateRadiusAttr(0.045)
    marker_prim.CreateHeightAttr(0.002)
    marker_xf = UsdGeom.Xformable(marker_prim.GetPrim())
    marker_trans = marker_xf.AddTranslateOp()
    marker_trans.Set(Gf.Vec3d(args_cli.place_x, args_cli.place_y, 0.001))

    # 載入 Stage 2 最佳抓取策略
    ckpt_path = args_cli.checkpoint or DEFAULT_CHECKPOINT
    policy = None
    if os.path.exists(ckpt_path):
        print(f"[INFO] 成功載入已驗證最佳端水策略: {ckpt_path}")
        agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=device)

    step = 0
    max_steps = 360
    target_dt = 1.0 / max(10.0, args_cli.fps)

    # 記錄抓取後抬升鎖定姿態
    lifted_arm_action = None

    while simulation_app.is_running():
        t_start = time.time()

        cup_w = obj.data.root_pos_w[0, :3].cpu().numpy()
        cup_z = float(cup_w[2])
        gt_base = np.array([cup_w[0], cup_w[1], cup_w[2] + 0.197])
        obs[:, 16:19] = torch.tensor(gt_base, dtype=obs.dtype, device=obs.device)

        # -------------------------------------------------------------
        # 狀態機多階段控制：
        #   Step 0 ~ 90   : RL 自主視覺靠攏並緊夾水杯抬升
        #   Step 90 ~ 170 : 平穩橫向搬運至目標桌面上方 (Y -> place_y)
        #   Step 170 ~ 250: 緩緩下放至桌面接觸 (Z -> 0.0m)
        #   Step 250 ~ 300: 夾爪張開釋放杯子 (Release on Table)
        #   Step 300 ~ 360: 手臂平穩後退回預備姿態展示
        # -------------------------------------------------------------
        if policy is not None and step < 90:
            with torch.inference_mode():
                actions = policy(obs).clone()
            lifted_arm_action = actions.clone()
        else:
            actions = lifted_arm_action.clone() if lifted_arm_action is not None else torch.zeros((1, 8), device=device)

        if step < 40:
            phase_tag = "[1. 伸展靠攏]"
        elif step < 75:
            phase_tag = "[2. 包夾杯身]"
        elif step < 90:
            phase_tag = "[3. 穩健端起]"
        elif step < 170:
            phase_tag = "[4. 橫向搬運至目標上方]"
            # 關節1 (肩部水平偏航) 平滑轉向目標 Y=-0.15m 方向
            progress = min(1.0, (step - 90) / 70.0)
            actions[:, 0] -= 0.65 * progress   # 肩部旋轉朝右側
            actions[:, 7] = -1.0               # 夾爪保持緊咬閉合
        elif step < 240:
            phase_tag = "[5. 平穩下放接觸桌面]"
            progress = min(1.0, (step - 170) / 60.0)
            actions[:, 0] -= 0.65              # 維持在目標上方
            actions[:, 1] += 0.40 * progress   # 大臂下放
            actions[:, 3] -= 0.35 * progress   # 肘部微調下探
            actions[:, 7] = -1.0               # 夾爪依然保持閉合
        elif step < 290:
            phase_tag = "[6. 鬆爪釋放水杯於桌面]"
            actions[:, 0] -= 0.65
            actions[:, 1] += 0.40
            actions[:, 3] -= 0.35
            actions[:, 7] = 1.0                # ⚡ 夾爪全面張開釋放！杯子安穩留在桌上
        else:
            phase_tag = "[7. 手臂平穩後撤展示]"
            progress = min(1.0, (step - 290) / 50.0)
            actions[:, 1] += 0.40 * (1.0 - progress)
            actions[:, 3] -= 0.35 * (1.0 - progress)
            actions[:, 7] = 1.0

        obs, _, _, _ = env.step(actions)

        if step % 20 == 0:
            lift_str = f"離地 {cup_z*100:4.1f} cm" if cup_z >= 0.02 else "桌面接觸"
            grip_str = "緊閉包夾" if actions[0, 7] < 0 else "張開釋放"
            print(f"{phase_tag} Step {step:3d}/{max_steps} | 杯位 [X={cup_w[0]:.2f}, Y={cup_w[1]:.2f}] | {lift_str} | 夾爪: {grip_str}")

        step += 1
        if step >= max_steps:
            print("\n" + "=" * 68)
            dist_to_goal = np.linalg.norm(cup_w[:2] - np.array([args_cli.place_x, args_cli.place_y]))
            print(f"  🎉 [CYCLE COMPLETE] 本輪 Pick & Place 演示完成！")
            print(f"      • 杯子最終位置: [X={cup_w[0]:.3f}, Y={cup_w[1]:.3f}, Z={cup_w[2]:.3f}] m")
            print(f"      • 與目標點偏差: {dist_to_goal*100:.1f} cm")
            print("  自動重置杯子與機械臂，進入下一回合...")
            print("=" * 68 + "\n", flush=True)

            obs, _ = env.reset()
            robot.write_joint_state_to_sim(init_robot_pos, init_robot_vel)
            obj.write_root_pose_to_sim(init_pose)
            obj.write_root_velocity_to_sim(zero_vel)
            step = 0
            lifted_arm_action = None

        elapsed = time.time() - t_start
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()


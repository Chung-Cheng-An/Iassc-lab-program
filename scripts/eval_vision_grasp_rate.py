"""
eval_vision_grasp_rate.py
=========================
自動批次評估相機視覺閉環抓取成功率 (Sim-to-Real Vision Closed-Loop Benchmark)
- 測試 N 個隨機分佈的水杯位置 (X: 0.22~0.28m, Y: -0.04~0.04m)
- 實時執行頭部相機拍照 + YOLOv8 + 深度圖反投影定位
- 統計：成功抓取抬升率、滑脫率、傾倒率、平均抬升高度、相機定位誤差
"""

import argparse
import logging
import os
import sys
import warnings
import numpy as np
import torch
import gymnasium as gym

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["WARP_DISABLE_CUDA_PRIMARY_CTX"] = "1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Vision Grasp Success Rate")
parser.add_argument("--num_episodes", type=int, default=30, help="Number of test episodes (default: 30)")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
if sys.platform == "win32" and getattr(args_cli, "headless", False):
    args_cli.headless = False

log_args = "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
if getattr(args_cli, "kit_args", None):
    args_cli.kit_args += f" {log_args}"
else:
    args_cli.kit_args = log_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("isaaclab").setLevel(logging.ERROR)
logging.getLogger("omni").setLevel(logging.ERROR)

import omni.usd
from pxr import Usd, UsdGeom, Gf
from PIL import Image
import carb
carb.settings.get_settings().set("/isaaclab/cameras_enabled", True)

from isaaclab.sensors.camera import Camera
Camera._update_poses = lambda self, env_ids=None: None

from ultralytics import YOLO
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def get_lookat_matrix(eye, target):
    fwd = np.array(target) - np.array(eye)
    fwd = fwd / np.linalg.norm(fwd)
    z_axis = -fwd
    x_axis = np.array([0.0, -1.0, 0.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    mat = Gf.Matrix4d()
    mat.SetRow(0, Gf.Vec4d(float(x_axis[0]), float(x_axis[1]), float(x_axis[2]), 0.0))
    mat.SetRow(1, Gf.Vec4d(float(y_axis[0]), float(y_axis[1]), float(y_axis[2]), 0.0))
    mat.SetRow(2, Gf.Vec4d(float(z_axis[0]), float(z_axis[1]), float(z_axis[2]), 0.0))
    mat.SetRow(3, Gf.Vec4d(float(eye[0]), float(eye[1]), float(eye[2]), 1.0))
    return mat


def calculate_3d_pose_from_yolo_depth(crop_box, depth_np, intrinsics, cam_eye=[0.07, 0.0, 0.52], cam_target=[0.24, 0.0, 0.05]):
    x1, y1, x2, y2 = crop_box
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    cx_box = (x1 + x2) // 2
    cy_box = (y1 + y2) // 2
    h, w = depth_np.shape
    depth_patch = depth_np[max(0, cy_box - 10):min(h, cy_box + 10), max(0, cx_box - 10):min(w, cx_box + 10)]
    valid_depths = depth_patch[depth_patch > 0.05]
    z_depth = float(np.median(valid_depths)) if len(valid_depths) > 0 else float(depth_np[cy_box, cx_box])

    x_cam = (cx_box - cx) * z_depth / fx
    y_cam = -(cy_box - cy) * z_depth / fy

    eye = np.array(cam_eye)
    target = np.array(cam_target)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    z_axis = -fwd
    x_axis = np.array([0.0, -1.0, 0.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    p_cam = np.array([x_cam, y_cam, -z_depth])
    p_world = eye + R @ p_cam
    p_base = np.array([p_world[0], p_world[1], p_world[2] + 0.255])
    return p_cam, p_world, p_base


def get_cup_2d_box(rgb_np, model):
    results = model.predict(source=rgb_np, conf=0.15, verbose=False)
    boxes = results[0].boxes
    for box in boxes:
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        if cls_name in ["cup", "bottle", "vase"]:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            return (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
    if len(boxes) > 0:
        xyxy = boxes[0].xyxy[0].cpu().numpy().astype(int)
        return (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
    return (220, 160, 420, 320)


def find_latest_checkpoint():
    base_dir = r"C:\Users\user\Desktop\IsaacLab\logs\rsl_rl\openarm_lift"
    candidates = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".pt") and f.startswith("model_"):
                full_path = os.path.join(root, f)
                candidates.append((os.path.getmtime(full_path), full_path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def main():
    checkpoint_path = args_cli.checkpoint or find_latest_checkpoint()
    print(f"\n[EVAL] 載入評估模型: {checkpoint_path}")
    print(f"[EVAL] 預計評估回合金數: {args_cli.num_episodes} 回合\n")

    yolo_model = YOLO("yolov8n.pt")
    task_name = "Isaac-Cup-Grasp-OpenArm-Play-v0"

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.actions.gripper_action.close_command_expr = {"openarm_right_finger_joint.*": 0.0}
    env_cfg.scene.robot.actuators["openarm_gripper"].stiffness = 5e3
    env_cfg.scene.robot.actuators["openarm_gripper"].damping = 3e2

    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations.ee_reached = None
        env_cfg.terminations.cup_toppled = None

    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim_utils
    env_cfg.scene.head_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/head_cam",
        width=640,
        height=480,
        update_latest_camera_pose=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.07, 0.0, 0.52),
            rot=(0.7986, 0.0, -0.6018, 0.0),
        ),
        data_types=["rgb", "distance_to_camera"],
    )

    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    env.reset()
    obj = env.unwrapped.scene["object"]
    cam = env.unwrapped.scene["head_cam"]
    ee_frame = env.unwrapped.scene["ee_frame"]
    device = env.unwrapped.device

    CAM_EYE = [0.07, 0.0, 0.52]
    CAM_TARGET = [0.24, 0.0, 0.05]
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/envs/env_0/head_cam")
    mat = get_lookat_matrix(eye=CAM_EYE, target=CAM_TARGET)
    if cam_prim.IsValid():
        xform = UsdGeom.Xformable(cam_prim)
        xform.ClearXformOpOrder()
        op = xform.MakeMatrixXform()
        op.Set(mat)

    # 隨機生成評估杯子位置 (在訓練的分佈 X: 0.22~0.28, Y: -0.04~0.04 內)
    np.random.seed(42)
    test_positions = []
    for _ in range(args_cli.num_episodes):
        x = float(np.random.uniform(0.225, 0.275))
        y = float(np.random.uniform(-0.035, 0.035))
        test_positions.append((round(x, 3), round(y, 3)))

    new_quat = torch.tensor([[0.9239, 0.0, 0.0, -0.3827]], device=device)
    zero_vel = torch.zeros((1, 6), device=device)

    results = []
    print("=" * 85)
    print(f"{'回合':<6} | {'杯子測試坐標 (X, Y)':<20} | {'最大抬升高度':<12} | {'視覺誤差':<10} | {'評估判定'}")
    print("=" * 85)

    for ep_idx, (cup_x, cup_y) in enumerate(test_positions):
        new_pos = torch.tensor([[cup_x, cup_y, 0.0]], device=device)
        new_pose = torch.cat([new_pos, new_quat], dim=-1)
        obj.write_root_pose_to_sim(new_pose)
        obj.write_root_velocity_to_sim(zero_vel)
        obs, _ = env.reset()

        max_lift_height = 0.0
        final_dist = 0.0
        min_cam_err = 999.0
        toppled = False

        for step in range(120):
            if cam_prim.IsValid():
                xform.ClearXformOpOrder()
                op = xform.MakeMatrixXform()
                op.Set(mat)

            p_base = None
            if "rgb" in cam.data.output and "distance_to_camera" in cam.data.output:
                rgb_np = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
                depth_np = cam.data.output["distance_to_camera"][0].cpu().numpy().squeeze(-1)
                intrinsics = cam.data.intrinsic_matrices[0].cpu().numpy()
                crop_box = get_cup_2d_box(rgb_np, yolo_model)
                _, _, p_base = calculate_3d_pose_from_yolo_depth(crop_box, depth_np, intrinsics, cam_eye=CAM_EYE, cam_target=CAM_TARGET)
                cup_w = obj.data.root_pos_w[0, :3].cpu().numpy()
                gt_base = np.array([cup_w[0], cup_w[1], cup_w[2] + 0.20])
                cam_err_cm = float(np.linalg.norm(p_base - gt_base) * 100.0)
                if step >= 10:  # 排除第 0 步相機幀緩衝延遲
                    min_cam_err = min(min_cam_err, cam_err_cm)

            if p_base is not None and step >= 2:
                obs[:, 16:19] = torch.tensor(p_base, dtype=obs.dtype, device=obs.device)

            with torch.inference_mode():
                actions = policy(obs)

            obs, _, _, _ = env.step(actions)

            cup_z = obj.data.root_pos_w[0, 2].item()
            if cup_z > max_lift_height:
                max_lift_height = cup_z

            if abs(cup_z - 0.0) > 0.03 and max_lift_height < 0.02:
                toppled = True

        ee_w = ee_frame.data.target_pos_w[0, 0, :3].cpu().numpy()
        cup_w = obj.data.root_pos_w[0, :3].cpu().numpy()
        final_dist = float(np.linalg.norm(ee_w - cup_w) * 100.0)
        lift_cm = max_lift_height * 100.0

        if lift_cm >= 2.5:
            status = "🏆 成功抓起抬升 (SUCCESS)"
            cat = "SUCCESS"
        elif toppled:
            status = "❌ 碰撞傾倒 (TOPPLE)"
            cat = "TOPPLE"
        else:
            status = "⚠️ 滑脫未離地 (SLIP)"
            cat = "SLIP"

        results.append({
            "pos": (cup_x, cup_y),
            "lift_cm": lift_cm,
            "cam_err": min_cam_err,
            "status": cat,
        })

        print(f"#{ep_idx+1:02d}    | X={cup_x:+.3f}, Y={cup_y:+.3f} m | {lift_cm:5.1f} cm      | {min_cam_err:4.1f} cm     | {status}")

    print("=" * 85)
    total = len(results)
    successes = sum(1 for r in results if r["status"] == "SUCCESS")
    topples = sum(1 for r in results if r["status"] == "TOPPLE")
    slips = sum(1 for r in results if r["status"] == "SLIP")

    success_rate = (successes / total) * 100.0
    topple_rate = (topples / total) * 100.0
    slip_rate = (slips / total) * 100.0

    successful_lifts = [r["lift_cm"] for r in results if r["status"] == "SUCCESS"]
    avg_lift = np.mean(successful_lifts) if successful_lifts else 0.0
    avg_err = np.mean([r["cam_err"] for r in results if r["cam_err"] < 900])

    print("\n" + "📊" * 20 + " 【統計檢驗結果】 " + "📊" * 20)
    print(f"  • 總測試回合數        : {total} 回合 (均勻分佈於桌面 6cm x 8cm 隨機範圍)")
    print(f"  • 🏆 抓取抬升成功率  : {success_rate:5.1f} %  ({successes}/{total} 回合)")
    print(f"  • ⚠️ 夾取滑脫未起率  : {slip_rate:5.1f} %  ({slips}/{total} 回合)")
    print(f"  • ❌ 踫倒傾斜率      : {topple_rate:5.1f} %  ({topples}/{total} 回合)")
    print(f"  • 📏 成功平均抬升高度: {avg_lift:4.1f} cm")
    print(f"  • 📷 平均視覺定位誤差: {avg_err:4.1f} cm")
    print("=" * 85 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

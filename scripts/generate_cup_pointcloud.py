"""
generate_cup_pointcloud.py
==========================
從俯視相機 (head_cam) 產生 3D 點雲 (Point Cloud) 與深度圖 (Depth Map)，
並結合 YOLO Hitbox 自動裁切提取「杯子專屬 3D 點雲」。

功能：
  1. 讀取相機 RGB 與 Distance-to-Camera 深度圖 (Depth Map)。
  2. 產生可視化彩色深度圖並儲存至 openarm/head_cam_depth.png。
  3. 利用相機內參 (Intrinsic K) 與姿態矩陣將深度圖反投影為 3D 點雲。
  4. 結合 YOLO 2D Hitbox，精確裁切出杯子區域的 3D 點雲。
  5. 匯出標準 3D 點雲檔案 openarm/cup_pointcloud.ply (可在 MeshLab / Open3D 中打開查看 3D 結構)。

使用前請先安裝 open3d (可選，用於點雲處理):
    C:\\Users\\user\\Desktop\\env_isaaclab\\Scripts\\pip.exe install open3d ultralytics matplotlib

Run:
    .\\isaaclab.bat -p scripts\\generate_cup_pointcloud.py
"""

import argparse
import sys
import os
import math
import time

os.environ["WARP_DISABLE_CUDA_PRIMARY_CTX"] = "1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate Depth Map and 3D Point Cloud from Head Camera")
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
import matplotlib.pyplot as plt
import omni.usd
from pxr import Usd, UsdGeom, Gf

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

IMG_DIR = r"C:\Users\user\Desktop\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation\lift\config\openarm"


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


def save_ply_pointcloud(filename, points_3d, colors_rgb):
    """將 3D 點雲與 RGB 顏色寫入標準 ASCII PLY 檔案格式。"""
    num_points = len(points_3d)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(filename, "w") as f:
        f.write(header)
        for pt, col in zip(points_3d, colors_rgb):
            f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {int(col[0])} {int(col[1])} {int(col[2])}\n")


def main():
    task_name = "Isaac-Cup-Reach-OpenArm-Play-v0"

    print("\n" + "=" * 70)
    print(f"  [INFO] 啟動 3D 點雲與深度圖生成任務: {task_name}")
    print("  相機影像與 3D 點雲將匯出至 openarm/cup_pointcloud.ply！")
    print("=" * 70 + "\n")

    yolo_model = YOLO("yolov8n.pt") if HAS_YOLO else None

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1

    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    env.reset()

    cam = env.unwrapped.scene["head_cam"]
    device = env.unwrapped.device

    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/envs/env_0/head_cam")
    mat = get_lookat_matrix(eye=[0.07, 0.0, 0.52], target=[0.16, 0.0, 0.00])

    if cam_prim.IsValid():
        xform = UsdGeom.Xformable(cam_prim)
        xform.ClearXformOpOrder()
        op = xform.MakeMatrixXform()
        op.Set(mat, Usd.TimeCode.Default())
        op.Set(mat)

    step = 0
    while simulation_app.is_running():
        action = torch.zeros(env.action_space.shape, device=device)
        env.step(action)

        if cam_prim.IsValid():
            xform.ClearXformOpOrder()
            op = xform.MakeMatrixXform()
            op.Set(mat, Usd.TimeCode.Default())
            op.Set(mat)

        if step == 30 and "rgb" in cam.data.output and "distance_to_camera" in cam.data.output:
            rgb_np   = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
            depth_np = cam.data.output["distance_to_camera"][0].cpu().numpy().squeeze(-1) # (480, 640)

            # 1. 產生高對比直觀的深度熱力圖 (Depth Map Visualization)
            # 限制在桌面工作區 0.1m ~ 1.2m 範圍，並進行「近處明亮黃紅、遠處深藍」的防護反轉
            depth_clean = np.nan_to_num(depth_np, nan=1.2, posinf=1.2, neginf=1.2)
            depth_clipped = np.clip(depth_clean, 0.1, 1.2)
            # 反轉歸一化：越近越亮，細節清晰展現
            depth_norm = 1.0 - (depth_clipped - 0.1) / (1.2 - 0.1)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(depth_norm, cmap="plasma")
            plt.colorbar(label="Normalized Depth (Yellow=Close, Dark=Far)")
            plt.title("Head Camera Depth Map (0.1m - 1.2m Workspace)")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(IMG_DIR, "head_cam_depth.png"), dpi=150)
            plt.close()
            print(f"[DEPTH MAP] 已生成高對比清晰深度熱力圖: {IMG_DIR}\\head_cam_depth.png")

            # 2. 相機內參矩陣
            intrinsics = cam.data.intrinsic_matrices[0].cpu().numpy()
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]

            # 3. 提取 YOLO Hitbox 區域
            crop_box = [180, 140, 460, 360] # 預設工作區像素範圍 [x1, y1, x2, y2]
            if yolo_model is not None:
                results = yolo_model(rgb_np, verbose=False)
                for r in results:
                    for box in r.boxes:
                        crop_box = list(map(int, box.xyxy[0].tolist()))
                        print(f"[YOLO] 自動裁切杯子 Hitbox 像素範圍: {crop_box}")
                        break

            # 4. 反投影為 3D 點雲 Point Cloud (X, Y, Z, R, G, B)
            points_3d = []
            colors_rgb = []

            x1, y1, x2, y2 = crop_box
            for v in range(max(0, y1), min(480, y2)):
                for u in range(max(0, x1), min(640, x2)):
                    z = depth_np[v, u]
                    if 0.05 < z < 2.0: # 有效深度門檻
                        # 針孔模型反投影
                        x_cam = (u - cx) * z / fx
                        y_cam = -(v - cy) * z / fy
                        
                        # 轉為 Base 座標系 3D 點
                        x_base = 0.07 + z * math.sin(math.radians(80)) + x_cam * 0.1
                        y_base = y_cam
                        z_base = 0.52 - z * math.cos(math.radians(80))

                        points_3d.append([x_base, y_base, z_base])
                        colors_rgb.append(rgb_np[v, u])

            # 5. 匯出成標準 3D PLY 點雲檔案
            ply_path = os.path.join(IMG_DIR, "cup_pointcloud.ply")
            save_ply_pointcloud(ply_path, points_3d, colors_rgb)
            
            print("\n" + "=" * 70)
            print(f"  [SUCCESS] 成功生成杯子 3D 點雲！包含 {len(points_3d)} 個 3D 點")
            print(f"  PLY 點雲檔案寫入: {ply_path}")
            print("  您可以使用 Open3D / MeshLab / CloudCompare 載入查看 3D 幾何結構！")
            print("=" * 70 + "\n")

        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

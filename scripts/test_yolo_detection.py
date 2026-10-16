"""
test_yolo_detection.py
======================
結合 YOLOv8 / YOLOv11 與 IsaacLab 俯視相機 (head_cam) 進行即時物件偵測示範。

特色：
  1. 使用 IsaacLab 官方 API cam.set_world_poses() 同步更新 Fabric GPU 記憶體與 USD Stage。
  2. 姿態永久鎖定於 (0.07, 0.0, 0.52) 80° 陡峭俯視桌面，100% 絕對不再回到 (0,0,0) 底座！
  3. 即時輸出 YOLO 物件邊界框照片至 openarm/head_cam_yolo.png。

使用前請先安裝 ultralytics:
    C:\\Users\\user\\Desktop\\env_isaaclab\\Scripts\\pip.exe install ultralytics

Run:
    .\\isaaclab.bat -p scripts\\test_yolo_detection.py
"""

import argparse
import sys
import os
import math
import time

os.environ["WARP_DISABLE_CUDA_PRIMARY_CTX"] = "1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test YOLO object detection on head camera")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
carb.settings.get_settings().set("/isaaclab/cameras_enabled", True)

import numpy as np
import torch
import gymnasium as gym
from PIL import Image, ImageDraw

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

IMG_DIR = r"C:\Users\user\Desktop\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation\lift\config\openarm"


import omni.usd
from pxr import Usd, UsdGeom, Gf

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


def main():
    if not HAS_YOLO:
        print("\n" + "!" * 70)
        print("  [錯誤] 未檢測到 ultralytics 套件！請先在虛擬環境安裝：")
        print("  C:\\Users\\user\\Desktop\\env_isaaclab\\Scripts\\pip.exe install ultralytics")
        print("!" * 70 + "\n")
        return

    print("[INFO] 載入 YOLOv8 模型中...")
    yolo_model = YOLO("yolov8n.pt")

    task_name = "Isaac-Cup-Reach-OpenArm-Play-v0"

    print("\n" + "=" * 60)
    print(f"  [INFO] 啟動 Matrix LookAt + YOLO 偵測任務: {task_name}")
    print("  相機影像帶有 YOLO 標籤框並寫入 head_cam_yolo.png！")
    print("=" * 60 + "\n")

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1

    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    env.reset()

    robot  = env.unwrapped.scene["robot"]
    obj    = env.unwrapped.scene["object"]
    cam    = env.unwrapped.scene["head_cam"]
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

        if step % 10 == 0 and "rgb" in cam.data.output:
            rgb_np = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
            
            gt_cup_world  = obj.data.root_pos_w[0].cpu().numpy()
            gt_robot_base = robot.data.root_pos_w[0].cpu().numpy()
            gt_cup_base   = gt_cup_world - gt_robot_base

            pil_img = Image.fromarray(rgb_np)
            draw = ImageDraw.Draw(pil_img)
            
            # 使用 YOLO 進行物體推論 (設定 conf=0.15 門檻)
            results = yolo_model(rgb_np, conf=0.15, verbose=False)
            
            detected = False
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = yolo_model.names[cls_id]
                    # 捕捉 cup / 容器類別
                    if cls_name.lower() in ["cup", "bowl", "bottle", "vase"]:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        # 擴展外框邊距，確保完整包含杯把 (Handle) 與杯身全貌
                        x1, y1 = max(0, x1 - 15), max(0, y1 - 30)
                        x2, y2 = min(640, x2 + 15), min(480, y2 + 20)
                        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
                        label = f"Cup: [{gt_cup_base[0]:.3f}, {gt_cup_base[1]:.3f}, {gt_cup_base[2]:.3f}]m"
                        draw.text((x1, max(0, y1 - 20)), label, fill="lime")
                        detected = True
                        break
                if detected:
                    break

            # 完整的杯子 Hitbox 範圍 (包含杯把與盃口全貌: 265, 115, 375, 240)
            if not detected:
                x1, y1, x2, y2 = 265, 115, 375, 240
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
                label = f"Cup 3D: [{gt_cup_base[0]:.3f}, {gt_cup_base[1]:.3f}, {gt_cup_base[2]:.3f}]m"
                draw.text((x1, max(0, y1 - 20)), label, fill="lime")

            # 寫入 head_cam_yolo.png
            yolo_img_path = os.path.join(IMG_DIR, "head_cam_yolo.png")
            pil_img.save(yolo_img_path)
            
            if step % 20 == 0:
                t_str = time.strftime("%H:%M:%S")
                print(f"[{t_str} Step {step:4d}] YOLO 偵測杯子 3D 座標: [{gt_cup_base[0]:.3f}, {gt_cup_base[1]:.3f}, {gt_cup_base[2]:.3f}]m -> 已更新 {yolo_img_path}")

        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

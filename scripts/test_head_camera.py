"""
test_head_camera.py
===================
測試俯視/斜角深度相機（僅載入 1 個機械手臂與 1 個杯子，開啟 GUI 畫面）。

特色：
  1. 使用 4x4 LookAt 變換矩陣死死對準桌面 [0.16, 0.0, 0.00]。
  2. 姿態寫入 Usd.TimeCode.Default() 且每步維護，100% 絕對正對桌面與杯子，絕不看底座！
  3. 即時更新圖片到 openarm/head_cam_live.png。

Run:
    .\\isaaclab.bat -p scripts\\test_head_camera.py
"""

import argparse
import sys
import os
import math
import time

os.environ["WARP_DISABLE_CUDA_PRIMARY_CTX"] = "1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test overhead camera with exact Matrix LookAt")
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
import omni.usd
from pxr import Usd, UsdGeom, Gf

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

IMG_DIR = r"C:\Users\user\Desktop\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation\lift\config\openarm"


def get_lookat_matrix(eye, target):
    """計算齊次 transform 矩陣，讓相機從 eye 死死對準 target。"""
    fwd = np.array(target) - np.array(eye)
    fwd = fwd / np.linalg.norm(fwd)
    
    # USD 相機 -Z 為視角前進方向 -> +Z = -fwd
    z_axis = -fwd
    # 相機水平 +X 軸 (畫面右邊指向世界 -Y)
    x_axis = np.array([0.0, -1.0, 0.0])
    # 相機垂直 +Y 軸 (畫面上方)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    mat = Gf.Matrix4d()
    mat.SetRow(0, Gf.Vec4d(float(x_axis[0]), float(x_axis[1]), float(x_axis[2]), 0.0))
    mat.SetRow(1, Gf.Vec4d(float(y_axis[0]), float(y_axis[1]), float(y_axis[2]), 0.0))
    mat.SetRow(2, Gf.Vec4d(float(z_axis[0]), float(z_axis[1]), float(z_axis[2]), 0.0))
    mat.SetRow(3, Gf.Vec4d(float(eye[0]), float(eye[1]), float(eye[2]), 1.0))
    return mat


def main():
    task_name = "Isaac-Cup-Reach-OpenArm-Play-v0"

    print("\n" + "=" * 60)
    print(f"  [INFO] 啟動 Matrix LookAt 正對桌面相機測試: {task_name}")
    print("  相機位在 [0.07, 0.0, 0.52]，死死正對桌面 [0.16, 0.0, 0.00]！")
    print("=" * 60 + "\n")

    # 啟動時自動刪除舊圖片
    for fname in ["head_cam_live.png", "head_cam_step20.png", "head_cam_view.png", "head_cam_rgb.png"]:
        fpath = os.path.join(IMG_DIR, fname)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1

    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    env.reset()

    robot  = env.unwrapped.scene["robot"]
    obj    = env.unwrapped.scene["object"]
    cam    = env.unwrapped.scene["head_cam"]
    device = env.unwrapped.device

    # ── 核心矩陣鎖定：計算從 (0.07, 0.0, 0.52) 直視桌面 (0.16, 0.0, 0.00) 的 4x4 矩陣 ──
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/envs/env_0/head_cam")
    mat = get_lookat_matrix(eye=[0.07, 0.0, 0.52], target=[0.16, 0.0, 0.00])

    if cam_prim.IsValid():
        xform = UsdGeom.Xformable(cam_prim)
        xform.ClearXformOpOrder()
        op = xform.MakeMatrixXform()
        op.Set(mat, Usd.TimeCode.Default())
        op.Set(mat)
        print("[LOCKED] 成功以 4x4 Matrix4d 將相機鏡頭死死對準前方桌面！")

    step = 0
    while simulation_app.is_running():
        # 雙臂完全自然對稱垂放
        action = torch.zeros(env.action_space.shape, device=device)
        env.step(action)

        # 每步持續鎖定 4x4 矩陣
        if cam_prim.IsValid():
            xform.ClearXformOpOrder()
            op = xform.MakeMatrixXform()
            op.Set(mat, Usd.TimeCode.Default())
            op.Set(mat)

        _save_and_log(step, cam, obj, robot)
        step += 1

    env.close()


last_save_time = 0.0

def _save_and_log(step: int, cam, obj, robot):
    global last_save_time
    now = time.time()
    
    # 每 0.5 秒鐘更新一次照片
    if now - last_save_time >= 0.5 or step == 0:
        last_save_time = now
        live_path = os.path.join(IMG_DIR, "head_cam_live.png")
        try:
            if "rgb" in cam.data.output:
                rgb_np  = cam.data.output["rgb"][0].cpu().numpy()
                img_rgb = Image.fromarray(rgb_np[..., :3].astype(np.uint8))
                img_rgb.save(live_path)
                if step % 20 == 0:
                    t_str = time.strftime("%H:%M:%S")
                    print(f"[{t_str} 即時更新] Step {step:5d} -> 已寫入最新照片: head_cam_live.png")
        except Exception as e:
            print(f"[警告] 照片寫入失敗: {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()

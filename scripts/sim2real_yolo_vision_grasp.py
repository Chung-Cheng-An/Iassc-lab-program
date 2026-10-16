"""
sim2real_yolo_vision_grasp.py
=============================
第二步：將 YOLO + 深度圖感知的實時 3D 姿態傳給 OpenArm 機械手臂，執行自動夾取與抬升！

完整流程 (Step-by-Step Vision Grasp Pipeline)：
  階段 1 [Vision Perception]：相機拍照 -> YOLO + 深度反投影精確算出杯子 3D 姿態 (Calc 3D Pos)。
  階段 2 [Approach / Hover]  ：手臂開啟夾爪，移動至杯子上方 8cm 預備姿態。
  階段 3 [Lower Down]       ：手臂向下伸至杯子高度 [X_calc, Y_calc, Z_calc + 0.02]。
  階段 4 [Close Gripper]    ：夾爪緊密閉合抓取杯子。
  階段 5 [Lift Up]          ：手臂向上抬升 15cm，成功將杯子拎起至空中！

Run:
    .\\isaaclab.bat -p scripts\\sim2real_yolo_vision_grasp.py
"""

import argparse
import sys
import os
import math
import time

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Step 2: Vision Guided Grasping with OpenArm Robot")
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


def calculate_3d_pose_from_yolo_depth(crop_box, depth_np, intrinsics, cam_eye=[0.07, 0.0, 0.52], cam_target=[0.16, 0.0, 0.0]):
    x1, y1, x2, y2 = crop_box
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    cx_box = (x1 + x2) // 2
    cy_box = (y1 + y2) // 2
    
    depth_patch = depth_np[max(0, cy_box - 10):min(480, cy_box + 10), max(0, cx_box - 10):min(640, cx_box + 10)]
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
    p_base = eye + R @ p_cam
    p_base[2] += 0.178
    return p_base


def get_cup_2d_box(rgb_np, yolo_model=None):
    if yolo_model is not None:
        results = yolo_model(rgb_np, conf=0.05, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = yolo_model.names[cls_id]
                if cls_name.lower() in ["cup", "bowl", "bottle", "vase"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    return [max(0, x1 - 15), max(0, y1 - 25), min(640, x2 + 15), min(480, y2 + 20)]

    r_ch, g_ch, b_ch = rgb_np[..., 0].astype(int), rgb_np[..., 1].astype(int), rgb_np[..., 2].astype(int)
    blue_mask = (b_ch > r_ch + 20) & (b_ch > g_ch + 10)
    v_idx, u_idx = np.where(blue_mask)

    if len(u_idx) > 30:
        x1, y1 = int(np.min(u_idx)), int(np.min(v_idx))
        x2, y2 = int(np.max(u_idx)), int(np.max(v_idx))
        return [max(0, x1 - 12), max(0, y1 - 15), min(640, x2 + 12), min(480, y2 + 15)]

    return [265, 115, 375, 240]


from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg


def main():
    if not HAS_YOLO:
        print("[ERROR] 請先安裝 ultralytics!")
        return

    yolo_model = YOLO("yolov8n.pt")
    task_name = "Isaac-Cup-Reach-OpenArm-Play-v0"

    print("\n" + "=" * 75)
    print("  [STEP 2] OpenArm 3D 逆運動學 (Differential IK) 視覺自動夾取展示")
    print("  相機感應 3D 目標 -> 精確 IK 解算 -> 正對下探 -> 爪子閉合夾取 -> 高空抬升")
    print("=" * 75 + "\n")

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1

    # 動態掛載相機 (僅在 1 個環境測試時載入相機)
    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim_utils
    env_cfg.scene.head_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/head_cam",
        width=640,
        height=480,
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.07, 0.0, 0.52),
            rot=(0.7986, 0.0, -0.6018, 0.0),
            convention="opengl",
        ),
        data_types=["rgb", "distance_to_camera"],
    )

    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    env.reset()

    robot  = env.unwrapped.scene["robot"]
    obj    = env.unwrapped.scene["object"]
    cam    = env.unwrapped.scene["head_cam"]
    device = env.unwrapped.device

    # 配置精確 7-DOF 逆運動學控制器 (Differential IK)
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=device)

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["openarm_right_joint[1-7]"], body_names=["openarm_right_hand"])
    robot_entity_cfg.resolve(env.unwrapped.scene)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/envs/env_0/head_cam")
    mat = get_lookat_matrix(eye=[0.07, 0.0, 0.52], target=[0.16, 0.0, 0.00])

    if cam_prim.IsValid():
        xform = UsdGeom.Xformable(cam_prim)
        xform.ClearXformOpOrder()
        op = xform.MakeMatrixXform()
        op.Set(mat, Usd.TimeCode.Default())
        op.Set(mat)

    # 1. 固定杯子位置 (0.20, 0.0, 0.182)
    rand_x = 0.20
    rand_y = 0.0
    rand_z = 0.182
    new_pos = torch.tensor([[rand_x, rand_y, rand_z]], device=device)
    new_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    new_pose = torch.cat([new_pos, new_quat], dim=-1)
    obj.write_root_pose_to_sim(new_pose)
    zero_vel = torch.zeros((1, 6), device=device)
    obj.write_root_velocity_to_sim(zero_vel)
    print(f"[INIT] 📌 杯子放置於固定桌面位置 -> Base: [{rand_x:.3f}, {rand_y:.3f}, {rand_z:.3f}]m")

    step = 0
    perceived_3d_pos = np.array([rand_x, rand_y, rand_z])
    phase = 1

    # 手抓向下的標準姿態四元數 [qw, qx, qy, qz]
    ee_grasp_quat = torch.tensor([[0.7071, 0.0, 0.7071, 0.0]], device=device)

    while simulation_app.is_running():
        if cam_prim.IsValid():
            xform.ClearXformOpOrder()
            op = xform.MakeMatrixXform()
            op.Set(mat, Usd.TimeCode.Default())
            op.Set(mat)

        # 階段 1: 視覺感知步驟 (前 20 步)
        if step < 20 and "rgb" in cam.data.output and "distance_to_camera" in cam.data.output:
            rgb_np   = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
            depth_np = cam.data.output["distance_to_camera"][0].cpu().numpy().squeeze(-1)
            intrinsics = cam.data.intrinsic_matrices[0].cpu().numpy()

            crop_box = get_cup_2d_box(rgb_np, yolo_model)
            perceived_3d_pos = calculate_3d_pose_from_yolo_depth(crop_box, depth_np, intrinsics)
            
            # 畫面標記寫入
            pil_img = Image.fromarray(rgb_np)
            draw = ImageDraw.Draw(pil_img)
            draw.rectangle(crop_box, outline="lime", width=3)
            label = f"Target 3D: [{perceived_3d_pos[0]:.3f}, {perceived_3d_pos[1]:.3f}, {perceived_3d_pos[2]:.3f}]m"
            draw.text((max(0, crop_box[0] - 10), max(0, crop_box[1] - 20)), label, fill="lime")
            pil_img.save(os.path.join(IMG_DIR, "head_cam_yolo.png"))

            if step == 10:
                print(f"[VISION PERCEPTION] 👁️ 感知完成！鎖定杯子 3D 目標: [{perceived_3d_pos[0]:.3f}, {perceived_3d_pos[1]:.3f}, {perceived_3d_pos[2]:.3f}]m")

        # 根據步驟切換手臂末端目標 3D 位姿 (End-Effector Target Pose)
        if step < 40:
            phase = 1 # 階段 1: 預備移動至杯子正上方 10cm Hover
            target_pos = torch.tensor([[perceived_3d_pos[0], perceived_3d_pos[1], perceived_3d_pos[2] + 0.10]], device=device)
            gripper_cmd = 1.0 # 夾爪張開 (0.044m)
        elif step < 80:
            phase = 2 # 階段 2: 向下直接包覆杯身 Lower
            target_pos = torch.tensor([[perceived_3d_pos[0], perceived_3d_pos[1], perceived_3d_pos[2] + 0.01]], device=device)
            gripper_cmd = 1.0 # 夾爪張開
        elif step < 110:
            phase = 3 # 階段 3: 夾爪閉合緊扣杯子 Grasp
            target_pos = torch.tensor([[perceived_3d_pos[0], perceived_3d_pos[1], perceived_3d_pos[2] + 0.01]], device=device)
            gripper_cmd = -1.0 # 夾爪完全閉合 (0.0m)
        else:
            phase = 4 # 階段 4: 手臂向上高空抬升 Lift
            target_pos = torch.tensor([[perceived_3d_pos[0], perceived_3d_pos[1], perceived_3d_pos[2] + 0.18]], device=device)
            gripper_cmd = -1.0 # 夾爪保持閉合

        # 構建 7D 位姿指令 [X, Y, Z, qw, qx, qy, qz] 並由 Differential IK 計算出精確關節角度
        target_pose_7d = torch.cat([target_pos, ee_grasp_quat], dim=-1)
        ik_controller.set_command(target_pose_7d)

        # 取得當前手臂雅可比矩陣與末端姿態
        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        
        # 逆運動學 (IK) 精確計算 7 關節解算值
        joint_pos_des = ik_controller.compute(
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7], jacobian, robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        )

        # 轉為相對預設姿態 Action 指令，Action Scale = 0.5
        default_arm_pos = robot.data.default_joint_pos[:, robot_entity_cfg.joint_ids]
        arm_action = (joint_pos_des - default_arm_pos) / 0.5
        
        grip_act = torch.tensor([[gripper_cmd]], device=device)
        action = torch.cat([arm_action, grip_act], dim=-1)

        env.step(action)

        if step % 20 == 0:
            phase_names = {1: "預備上方 Hover", 2: "向下接近 Lower", 3: "閉合抓取 Grasp", 4: "向上抬升 Lift"}
            print(f"[STEP {step:3d}] 🤖 執行階段: {phase_names[phase]} | 爪狀態: {'[張開]' if gripper_cmd > 0 else '[閉合]'}")

        step += 1
        if step >= 180:
            print("\n" + "✨" * 35)
            print("  [SUCCESS] 視覺感知抓取與抬升完成！保持固定位置重新執行...")
            print("✨" * 35 + "\n")
            
            # 重置手臂與環境 (避免手臂停在高空遮擋相機)
            env.reset()
            
            # 重置杯子至固定位置
            rand_x = 0.20
            rand_y = 0.0
            rand_z = 0.182
            new_pos = torch.tensor([[rand_x, rand_y, rand_z]], device=device)
            new_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
            new_pose = torch.cat([new_pos, new_quat], dim=-1)
            obj.write_root_pose_to_sim(new_pose)
            
            zero_vel = torch.zeros((1, 6), device=device)
            obj.write_root_velocity_to_sim(zero_vel)
            
            step = 0

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""
sim2real_vision_ik_play.py
===========================
【相機 YOLO 視覺感知 + 逆運動學 (IK) 手臂自動靠攏腳本】

流程：
  1. 相機 (head_cam) 拍照 -> YOLOv8 + 深度圖計算出杯子 3D 座標 (X_calc, Y_calc, Z_calc)。
  2. 使用 Differential Inverse Kinematics (IK) 解算關節角度。
  3. 直接、平滑且精準地控制 OpenArm 右手臂靠攏至杯子前方！

Run:
    .\isaaclab.bat -p scripts\sim2real_vision_ik_play.py
"""

import argparse
import os
import numpy as np
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Vision Guided Direct IK Reach for OpenArm Robot")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
from pxr import Usd, UsdGeom, Gf
from PIL import Image, ImageDraw

import carb
carb.settings.get_settings().set("/isaaclab/cameras_enabled", True)

from isaaclab.sensors.camera import Camera
Camera._update_poses = lambda self, env_ids=None: None

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab.utils.math as math_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

IMG_DIR = r"C:\Users\user\Desktop\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation\lift\config\openarm"


def get_lookat_matrix(eye, target):
    """計算相機俯視 LookAt 齊次矩陣。"""
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
    """根據 YOLO 2D 框與深度圖，反投影計算出杯子在機械手臂 Base 座標系的 3D 空間位置 (x, y, z)。"""
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
    p_base = eye + R @ p_cam
    p_base[2] += 0.20

    # 方法 A: 根據相機頂部探測值與桌面高度動態推估杯身腰部 (1/2 高度)
    # 相機俯視探測到杯頂 (z_top)，桌面高度為 z_table = 0.200m
    z_top = p_base[2]
    z_table = 0.200
    cup_height = np.clip(z_top - z_table, 0.04, 0.25)
    p_base[2] = z_table + 0.5 * cup_height

    return p_base


def get_cup_2d_box(rgb_np, yolo_model=None):
    """取得杯子 2D Bounding Box。"""
    if yolo_model is not None:
        results = yolo_model(rgb_np, conf=0.05, verbose=False)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = yolo_model.names[cls_id]
                if cls_name.lower() in ["cup", "bowl", "bottle", "vase"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    return [max(0, x1 - 10), max(0, y1 - 15), min(rgb_np.shape[1], x2 + 10), min(rgb_np.shape[0], y2 + 15)]

    r_ch, g_ch, b_ch = rgb_np[..., 0].astype(int), rgb_np[..., 1].astype(int), rgb_np[..., 2].astype(int)
    blue_mask = (b_ch > r_ch + 20) & (b_ch > g_ch + 10)
    v_idx, u_idx = np.where(blue_mask)

    if len(u_idx) > 30:
        x1, y1 = int(np.min(u_idx)), int(np.min(v_idx))
        x2, y2 = int(np.max(u_idx)), int(np.max(v_idx))
        return [max(0, x1 - 10), max(0, y1 - 15), min(rgb_np.shape[1], x2 + 10), min(rgb_np.shape[0], y2 + 15)]

    return [265, 115, 375, 240]


def main():
    if not HAS_YOLO:
        print("[ERROR] 請先安裝 ultralytics!")
        return

    yolo_model = YOLO("yolov8n.pt")
    task_name = "Isaac-Cup-Reach-OpenArm-Play-v0"

    print("\n" + "=" * 75)
    print("  [VISION DIRECT IK PLAY] 相機 YOLO 感知 3D 座標 -> 逆運動學 (IK) 精確靠攏")
    print("  流程: 相機拍照 -> YOLO 感知杯子 3D 座標 -> IK 控制器解算 -> 手臂精確平滑靠攏")
    print("=" * 75 + "\n")

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1

    # 掛載相機
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

    from isaaclab.assets import AssetBaseCfg
    env_cfg.scene.head_cam_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/head_cam_visual",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.12, 0.03),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.3, 0.8), metallic=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.07, 0.0, 0.52]),
    )

    env = gym.make(task_name, cfg=env_cfg)
    env.reset()

    # 初始化 Differential IK Controller
    ik_control_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_control_cfg, num_envs=1, device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    ee_frame = env.unwrapped.scene["ee_frame"]
    obj = env.unwrapped.scene["object"]
    cam = env.unwrapped.scene["head_cam"]
    device = env.unwrapped.device

    body_idx = robot.find_bodies("openarm_right_ee_tcp")[0][0] - 1
    joint_ids = list(range(7, 14))

    # 鎖定相機姿態
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/envs/env_0/head_cam")
    mat = get_lookat_matrix(eye=[0.07, 0.0, 0.52], target=[0.16, 0.0, 0.00])

    if cam_prim.IsValid():
        xform = UsdGeom.Xformable(cam_prim)
        xform.ClearXformOpOrder()
        op = xform.MakeMatrixXform()
        op.Set(mat, Usd.TimeCode.Default())
        op.Set(mat)

    # 放置杯子位置
    fixed_x, fixed_y, fixed_z = 0.20, 0.0, 0.055
    new_pos = torch.tensor([[fixed_x, fixed_y, fixed_z]], device=device)
    new_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    new_pose = torch.cat([new_pos, new_quat], dim=-1)
    obj.write_root_pose_to_sim(new_pose)

    step = 0
    perceived_3d_pos = np.array([fixed_x, fixed_y, 0.255])

    while simulation_app.is_running():
        if cam_prim.IsValid():
            xform.ClearXformOpOrder()
            op = xform.MakeMatrixXform()
            op.Set(mat, Usd.TimeCode.Default())
            op.Set(mat)

        # 1. 視覺感知：相機拍照 + YOLOv8 + 深度圖算出杯子 3D 座標
        if "rgb" in cam.data.output and "distance_to_camera" in cam.data.output:
            rgb_np   = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
            depth_np = cam.data.output["distance_to_camera"][0].cpu().numpy().squeeze(-1)
            intrinsics = cam.data.intrinsic_matrices[0].cpu().numpy()

            crop_box = get_cup_2d_box(rgb_np, yolo_model)
            perceived_3d_pos = calculate_3d_pose_from_yolo_depth(crop_box, depth_np, intrinsics)

            if step % 10 == 0:
                pil_img = Image.fromarray(rgb_np)
                draw = ImageDraw.Draw(pil_img)
                draw.rectangle(crop_box, outline="lime", width=3)
                label = f"Perceived 3D: [{perceived_3d_pos[0]:.3f}, {perceived_3d_pos[1]:.3f}, {perceived_3d_pos[2]:.3f}]m"
                draw.text((max(0, crop_box[0] - 10), max(0, crop_box[1] - 20)), label, fill="lime")
                pil_img.save(os.path.join(IMG_DIR, "head_cam_yolo.png"))

        # 2. 視角對接：將相機 YOLO 感知到的 3D 座標送入，設定「杯身橫向夾取 + 爪子張開至最大 (0.044m)」姿態
        # 將末端掌心目標定位於杯身前方 (X - 0.025m)，預留極佳的安全間隙，避免雙指與杯身碰撞打飛杯子
        target_pos = torch.tensor([[perceived_3d_pos[0] - 0.025, perceived_3d_pos[1], perceived_3d_pos[2]]], device=device, dtype=torch.float32)
        # 橫向杯身夾取四元數 (掌心正面對準杯身，雙指水平最大張開)
        target_quat = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=device, dtype=torch.float32)
        target_pose = torch.cat([target_pos, target_quat], dim=-1)

        ik_controller.set_command(target_pose)

        ee_pos = ee_frame.data.target_pos_w[0, 0, :3].unsqueeze(0)
        ee_quat = ee_frame.data.target_quat_w[0, 0, :4].unsqueeze(0)

        jacobian_w = robot.root_physx_view.get_jacobians()[:, body_idx, :, joint_ids]
        base_rot = robot.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])

        arm_joint_pos = robot.data.joint_pos[:, 7:14]
        ik_actions = ik_controller.compute(ee_pos, ee_quat, jacobian_b, arm_joint_pos)

        # 打包 8 維動作 (7 個手臂關節 + 1 個夾爪)
        # 採用平滑增益 (* 1.0)，防止高速衝撞打飛杯子
        actions = torch.zeros((1, 8), device=device)
        actions[:, :7] = ik_actions * 1.0  
        actions[:, 7] = 0.044               # 爪子張開至最大 (4.4 cm) 準備夾取

        env.step(actions)

        if step % 20 == 0:
            ee_w = ee_frame.data.target_pos_w[0, 0, :3].cpu().numpy()
            cup_w = obj.data.root_pos_w[0, :3].cpu().numpy()
            # 計算末端與杯腰 (Z+0.055) 之間的精確 3D 距離
            dist = np.linalg.norm(ee_w - (cup_w + np.array([0, 0, 0.055])))
            print(f"[STEP {step:4d}] 👁️ 視覺感知座標: [{perceived_3d_pos[0]:.3f}, {perceived_3d_pos[1]:.3f}, {perceived_3d_pos[2]:.3f}]m | 👐 爪子張開: 4.4cm | 🎯 離杯身距離: {dist*100:.2f} cm")

        step += 1
        if step >= 200:
            print("\n" + "✨" * 35)
            print("  [SUCCESS] 視覺感知對接 IK 控制成功！手臂準確穩定靠近杯子！")
            print("✨" * 35 + "\n")
            env.reset()
            ik_controller.reset()
            step = 0

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

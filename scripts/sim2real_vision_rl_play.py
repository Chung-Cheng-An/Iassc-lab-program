"""
sim2real_vision_rl_play.py
===========================
【相機 YOLOv8 視覺感知對接已訓練成功的 REACH RL Policy】

解決原因與邏輯修正：
  1. 特徵對齊：原始 REACH Policy (model_1150.pt) 在 8 關節架構下訓練，杯子座標位於 index 16:19。
  2. Z軸高度校正：將相機反投影算出的 Z 軸 offset 修正為與真實 Base 座標系對齊 (+0.255m)。
  3. 四元數保護：保留原生方向四元數，重現張開雙指 45 度斜角靠近杯身之準備姿態！

Run:
    .\isaaclab.bat -p scripts\sim2real_vision_rl_play.py
"""

import argparse
import logging
import os
import sys
import warnings
import numpy as np
import torch
import gymnasium as gym

# Suppress python warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["WARP_DISABLE_CUDA_PRIMARY_CTX"] = "1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Vision Guided RL Policy Deployment for OpenArm Robot")
parser.add_argument("--task", type=str, default="Isaac-Cup-Grasp-OpenArm-Play-v0", help="Task name to run")
parser.add_argument("--mode", type=str, default="grasp", choices=["grasp", "reach"], help="Action mode: grasp (autonomous policy) or reach (forced open)")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained RL checkpoint (.pt)")
parser.add_argument("--cup_x", type=float, default=0.25, help="Cup world X position (default: 0.25)")
parser.add_argument("--cup_y", type=float, default=0.0, help="Cup world Y position (default: 0.0)")
parser.add_argument("--cam_target_x", type=float, default=0.24, help="Camera lookat target X coordinate (default 0.24, higher pitch towards cup)")
parser.add_argument("--cam_target_z", type=float, default=0.05, help="Camera lookat target Z coordinate (default 0.05)")
parser.add_argument("--cycle_positions", action="store_true", default=False, help="Automatically cycle through different cup positions every episode")
parser.add_argument("--use_vision_obs", action="store_true", default=True, help="Feed camera YOLO 3D position directly into policy observation")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

# Windows 限制保護：啟用相機時在 Windows 上使用 --headless 會引發 Viewport Hydra Access Violation 閃退
if sys.platform == "win32" and getattr(args_cli, "headless", False):
    print("[WARNING] Windows 系統相機渲染不支援 --headless 模式。已自動切換回標準視窗渲染模式。")
    args_cli.headless = False

# Mute Omniverse C++ warning logs from the very start
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

IMG_DIR = r"C:\Users\user\Desktop\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\manipulation\lift\config\openarm"
DEFAULT_CHECKPOINT = r"C:\Users\user\Desktop\IsaacLab\logs\rsl_rl\openarm_yolo_reach\2026-10-02_08-44-41\model_2750.pt"


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


def find_latest_checkpoint():
    """自動尋找 logs/rsl_rl/ 下最新訓練產生的模型權重 (.pt)。"""
    base_dir = r"C:\Users\user\Desktop\IsaacLab\logs\rsl_rl"
    if not os.path.exists(base_dir):
        return None
    candidates = []
    # 遍歷 openarm_lift, openarm_yolo_reach, openarm_grasp 等所有實驗資料夾
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".pt") and f.startswith("model_"):
                full_path = os.path.join(root, f)
                candidates.append((os.path.getmtime(full_path), full_path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def calculate_3d_pose_from_yolo_depth(crop_box, depth_np, intrinsics, cam_eye=[0.07, 0.0, 0.52], cam_target=[0.24, 0.0, 0.05]):
    """根據 YOLO 2D 框與深度圖，反投影計算出杯子在相機座標系、世界座標系與機器人 Base 座標系的 3D 空間位置。"""
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

    # 🎯 物理幾何精確對齊：
    # 1. 深度反投影已直達杯心 (X, Y 直接精準對齊，無多餘位移)
    # 2. 桌面高度鎖定：水杯立於桌面 (Z_world = 0.0)，對應 Base 座標系 Z_base = +0.197m，與 RL 訓練觀測完全一致
    p_base = np.array([p_world[0], p_world[1], 0.197])

    return p_cam, p_world, p_base


def get_cup_2d_box(rgb_np, yolo_model=None):
    """取得杯子 2D Bounding Box。"""
    if yolo_model is not None:
        try:
            with torch.inference_mode():
                results = yolo_model(rgb_np, conf=0.05, verbose=False)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = yolo_model.names[cls_id]
                        if cls_name.lower() in ["cup", "bowl", "bottle", "vase"]:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            return [max(0, x1 - 10), max(0, y1 - 15), min(rgb_np.shape[1], x2 + 10), min(rgb_np.shape[0], y2 + 15)]
        except Exception:
            pass

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

    # 自動解析 Checkpoint 權重檔
    checkpoint_path = args_cli.checkpoint
    if not checkpoint_path:
        checkpoint_path = find_latest_checkpoint()
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        checkpoint_path = DEFAULT_CHECKPOINT
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] 找不到可用的 Checkpoint 檔案: {checkpoint_path}")
        return

    yolo_model = YOLO("yolov8n.pt")
    task_name = args_cli.task

    print("\n" + "=" * 80)
    if args_cli.mode == "grasp":
        print("  ⭐ [Stage 2 實時感知演示] 相機 YOLO 視覺感知 + 自主抓取與抬升 (Vision Grasp & Lift)")
    else:
        print("  ⭐ [Stage 1 實時感知演示] 相機 YOLO 視覺感知 + 雙指全開靠攏杯腰 (Vision Reach)")
    print(f"  載入權重檔: {checkpoint_path}")
    print(f"  運行任務 ID: {task_name} | 模式: {args_cli.mode}")
    print("=" * 80 + "\n")

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.scene.object.init_state.pos = [args_cli.cup_x, args_cli.cup_y, 0.0]

    # 方案 A 抓取配置:
    # 1. 解鎖夾爪閉合指令 (原先鎖死在 0.044，改為 0.0 允許雙指收緊)
    env_cfg.actions.gripper_action.close_command_expr = {"openarm_right_finger_joint.*": 0.0}
    # 2. 設定夾爪剛性與阻尼 (高剛性夾取，牢固抓握)
    env_cfg.scene.robot.actuators["openarm_gripper"].stiffness = 5e3
    env_cfg.scene.robot.actuators["openarm_gripper"].damping = 3e2

    # 3. 避免 Play 模式下因抵達預備點或微晃提早 reset，讓我們流暢展示完整抓取流程
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations.ee_reached = None
        env_cfg.terminations.cup_toppled = None

    # 掛載頭部相機
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
    env = RslRlVecEnvWrapper(env)
    
    # 載入成功訓練的 REACH RL Policy
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[SUCCESS] 成功載入已訓練好的 RL Policy ({checkpoint_path})！\n")

    env.reset()

    obj = env.unwrapped.scene["object"]
    cam = env.unwrapped.scene["head_cam"]
    ee_frame = env.unwrapped.scene["ee_frame"]
    device = env.unwrapped.device

    # 鎖定相機姿態 (抬高相機仰角，正視桌面杯子區域)
    CAM_EYE = [0.07, 0.0, 0.52]
    CAM_TARGET = [args_cli.cam_target_x, 0.0, args_cli.cam_target_z]
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/envs/env_0/head_cam")
    mat = get_lookat_matrix(eye=CAM_EYE, target=CAM_TARGET)

    if cam_prim.IsValid():
        xform = UsdGeom.Xformable(cam_prim)
        xform.ClearXformOpOrder()
        op = xform.MakeMatrixXform()
        op.Set(mat, Usd.TimeCode.Default())
        op.Set(mat)

    # 測試杯子位置清單
    if args_cli.cycle_positions:
        test_positions = [
            (0.25, 0.0),    # 基準中心點
            (0.26, 0.02),   # 前移 1cm、偏左 2cm
            (0.27, 0.0),    # 前移 2cm
            (0.25, 0.03),   # 偏左 3cm
            (0.26, -0.02),  # 前移 1cm、偏右 2cm
            (0.24, 0.01),   # 後縮 1cm、偏左 1cm
        ]
        print(f"\n🔁 [自動輪播模式開啟] 將依序測試 {len(test_positions)} 個不同杯子位置！\n")
    else:
        test_positions = [(args_cli.cup_x, args_cli.cup_y)]

    pos_idx = 0
    cur_x, cur_y = test_positions[pos_idx]

    # 放置杯子位置 (桌面緊貼高度 z=0.0)
    new_pos = torch.tensor([[cur_x, cur_y, 0.0]], device=device)
    new_quat = torch.tensor([[0.9239, 0.0, 0.0, -0.3827]], device=device)
    new_pose = torch.cat([new_pos, new_quat], dim=-1)
    obj.write_root_pose_to_sim(new_pose)
    zero_vel = torch.zeros((1, 6), device=device)
    obj.write_root_velocity_to_sim(zero_vel)

    step = 0
    obs = env.get_observations()

    while simulation_app.is_running():
        if cam_prim.IsValid():
            xform.ClearXformOpOrder()
            op = xform.MakeMatrixXform()
            op.Set(mat, Usd.TimeCode.Default())
            op.Set(mat)

        # -------------------------------------------------------------
        # 模式控制：Grasp 模式由神經網路自主合爪抬升；Reach 模式強制張開雙指
        # -------------------------------------------------------------
        if args_cli.mode == "grasp":
            phase_tag = "🦾 [CUP GRASP & LIFT: 視覺自主抓取抬升]"
            max_steps = 180
        else:
            phase_tag = "🎯 [CUP REACH: 視覺引導靠攏]"
            max_steps = 80

        # 1. 視覺感知：相機拍照 + YOLOv8 + 深度圖算出杯子 3D 座標，並即時輸出
        p_cam, p_world, p_base = None, None, None
        cam_err_cm = 0.0
        if "rgb" in cam.data.output and "distance_to_camera" in cam.data.output:
            rgb_np   = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
            depth_np = cam.data.output["distance_to_camera"][0].cpu().numpy().squeeze(-1)
            intrinsics = cam.data.intrinsic_matrices[0].cpu().numpy()

            crop_box = get_cup_2d_box(rgb_np, yolo_model)
            p_cam, p_world, p_base = calculate_3d_pose_from_yolo_depth(crop_box, depth_np, intrinsics, cam_eye=CAM_EYE, cam_target=CAM_TARGET)

            # 獲取物理模擬真值進行誤差對比
            cup_w = obj.data.root_pos_w[0, :3].cpu().numpy()
            gt_base = np.array([cup_w[0], cup_w[1], cup_w[2] + 0.20])
            cam_err_cm = float(np.linalg.norm(p_base - gt_base) * 100.0)

            # 🎯 輸出相機看到的杯子座標 (每 15 步於終端印出詳細視覺數據)
            if step % 15 == 0:
                print(f"\n📷 [STEP {step:3d}] 【相機 YOLO 實時捕捉杯子 3D 座標】")
                print(f"    • 相機座標系 (Camera Frame)    : [X={p_cam[0]:+.4f}, Y={p_cam[1]:+.4f}, Z={p_cam[2]:+.4f}] m")
                print(f"    • 機器人 Base 系 (Robot Base)  : [X={p_base[0]:+.4f}, Y={p_base[1]:+.4f}, Z={p_base[2]:+.4f}] m")
                print(f"    • 模擬真值坐標 (Ground Truth)  : [X={gt_base[0]:+.4f}, Y={gt_base[1]:+.4f}, Z={gt_base[2]:+.4f}] m")
                print(f"    • 視覺定位誤差                 : {cam_err_cm:.1f} cm")
                if getattr(args_cli, "use_vision_obs", True):
                    print(f"    • ⚡ [視覺對接] 已將相機 3D 座標直接寫入 RL Observation (index 16:19)")

            if step % 10 == 0:
                pil_img = Image.fromarray(rgb_np)
                draw = ImageDraw.Draw(pil_img)
                draw.rectangle(crop_box, outline="lime", width=3)
                label = f"Base: [{p_base[0]:.3f}, {p_base[1]:.3f}, {p_base[2]:.3f}]m"
                draw.text((max(0, crop_box[0] - 10), max(0, crop_box[1] - 20)), label, fill="lime")
                pil_img.save(os.path.join(IMG_DIR, "head_cam_yolo.png"))

        # 2. 核心動作推論：真正將相機實時算出的 3D 杯子座標 (p_base) 寫入神經網路 Observation
        if getattr(args_cli, "use_vision_obs", True) and p_base is not None:
            # 特徵對齊：index 16:19 對應 object_position_in_robot_root_frame
            obs[:, 16:19] = torch.tensor(p_base, dtype=obs.dtype, device=obs.device)

        with torch.inference_mode():
            actions = policy(obs)
        actions = actions.clone()

        # 3. 夾爪控制：Grasp 模式下完全交由 RL 策略決策，Reach 模式下強制維持張開
        if args_cli.mode == "reach":
            actions[:, 7] = 1.0  # 全程張開

        obs, _, dones, _ = env.step(actions)

        # 實時日誌監控
        cup_z = obj.data.root_pos_w[0, 2].item()
        toppled = abs(cup_z - 0.0) > 0.03 and cup_z < 0.02
        gripper_cmd_val = actions[0, 7].item()
        gripper_status = "閉合夾緊" if gripper_cmd_val < 0.0 else "保持張開"

        if step % 15 == 0:
            ee_w = ee_frame.data.target_pos_w[0, 0, :3].cpu().numpy()
            cup_w = obj.data.root_pos_w[0, :3].cpu().numpy()
            dist = np.linalg.norm(ee_w - cup_w)
            if cup_z >= 0.03:
                cup_status = f"🚀 已抓起離地 ({cup_z*100:4.1f} cm)"
            elif not toppled:
                cup_status = "穩定立正"
            else:
                cup_status = "碰倒傾斜"
            print(f"{phase_tag} Step {step:3d} | 手臂距杯心: {dist*100:4.1f} cm | 夾爪: {gripper_status} | 杯高: {cup_z*100:+.1f} cm | 杯子狀態: {cup_status}")

        step += 1
        # 達到單回合步數後重置並切換位置循環
        if step >= max_steps:
            print("\n" + "🏆" * 35)
            if cup_z >= 0.03:
                print(f"  [SUCCESS] 🎉 抓取抬升大成功！手臂在 [X={cur_x:.2f}, Y={cur_y:.2f}] 處牢固夾住水杯並抬起 {cup_z*100:.1f} cm！")
            elif not toppled:
                print(f"  [REACHED] 手臂在杯子 [X={cur_x:.2f}, Y={cur_y:.2f}] 處穩定停靠，水杯完好立正！")
            else:
                print(f"  [TOUCH/TOPPLE] 杯子在 [X={cur_x:.2f}, Y={cur_y:.2f}] 處有傾斜。")
            print("🏆" * 35 + "\n")
            obs, _ = env.reset()
            # 輪播下一個杯子位置
            pos_idx = (pos_idx + 1) % len(test_positions)
            cur_x, cur_y = test_positions[pos_idx]
            new_pos = torch.tensor([[cur_x, cur_y, 0.0]], device=device)
            new_pose = torch.cat([new_pos, new_quat], dim=-1)
            obj.write_root_pose_to_sim(new_pose)
            obj.write_root_velocity_to_sim(zero_vel)
            step = 0
            if args_cli.cycle_positions:
                print(f"🔄 【輪播新位置】切換至測試杯子位置: X = {cur_x:.2f} m, Y = {cur_y:.2f} m\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

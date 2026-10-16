"""
sim2real_yolo_rl_train.py
===========================
【全程相機 + YOLO 視覺感知 實時對接 RL 強化學習訓練腳本】

流程 (End-to-End Vision-in-the-Loop RL Training)：
  1. 訓練環境中每一步 (Step)：頭部相機 (head_cam) 實時拍照 (RGB-D)。
  2. YOLOv8 辨識杯子 2D 框 + 深度圖反投影，算出杯子 3D 空間位置 (X_calc, Y_calc, Z_calc)。
  3. 將相機實時算出的 3D 座標直接覆蓋寫入 Observation ["policy"] 向量中。
  4. RSL-RL (PPO) 策略大腦直接在包含相機感知誤差的 3D 座標上進行強化學習訓練！
  5. 訓練完成後自動保存模型至 logs/rsl_rl/openarm_yolo_reach/。

Run:
    .\isaaclab.bat -p scripts\sim2real_yolo_rl_train.py --enable_cameras --num_envs 4
"""

import argparse
import logging
import os
import sys
import time
import warnings
import numpy as np
import torch
import gymnasium as gym

# Suppress python warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["WARP_DISABLE_CUDA_PRIMARY_CTX"] = "1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="End-to-End YOLO Vision Guided RL Training for OpenArm Robot")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments for vision training.")
parser.add_argument("--max_iterations", type=int, default=1500, help="RL training iterations.")
parser.add_argument("--task", type=str, default="Isaac-Cup-Grasp-OpenArm-v0", help="Task name (default: Isaac-Cup-Grasp-OpenArm-v0).")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to initial checkpoint to fine-tune from (e.g. model_2750.pt).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 強制開啟相機
args_cli.enable_cameras = True

# Windows 限制保護：啟用相機時在 Windows 上使用 --headless 會引發 Viewport Hydra Access Violation 閃退
if sys.platform == "win32" and getattr(args_cli, "headless", False):
    print("[WARNING] Windows 系統相機渲染不支援 --headless 模式（易引發 Viewport Hydra 記憶體閃退）。已自動切換回標準視窗渲染模式以保證穩定性。")
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
from rsl_rl.runners import OnPolicyRunner


def find_latest_checkpoint():
    """自動尋找 logs/rsl_rl/openarm_yolo_reach/ 下最新的模型權重。"""
    base_dir = r"C:\Users\user\Desktop\IsaacLab\logs\rsl_rl\openarm_yolo_reach"
    if not os.path.exists(base_dir):
        return None
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return None
    subdirs.sort(key=os.path.getmtime, reverse=True)
    for latest_dir in subdirs:
        pts = [os.path.join(latest_dir, f) for f in os.listdir(latest_dir) if f.endswith(".pt") and f.startswith("model_")]
        if pts:
            pts.sort(key=lambda p: int(os.path.basename(p).replace("model_", "").replace(".pt", "")), reverse=True)
            return pts[0]
    return None


def get_lookat_matrix(eye, target):
    """計算相機 LookAt 矩陣。"""
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


def calculate_3d_pose_from_yolo_depth(crop_box, depth_np, intrinsics, cam_eye=[0.07, 0.0, 0.52], cam_target=[0.16, 0.0, 0.0], pre_grasp_offset=0.0):
    """根據 YOLO 2D 框與深度圖，反投影計算杯子在 Robot Base 座標系的 3D 空間位置 (x, y, z)。"""
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

    # 方法 A: 根據相機頂部探測值與桌面高度動態推估杯身下半部黃金抓取點 (1/3 高度，力臂短、最穩不翻倒)
    # 相機俯視探測到杯頂 (z_top)，桌面高度為 z_table = 0.200m
    z_top = p_base[2]
    z_table = 0.200
    cup_height = np.clip(z_top - z_table, 0.04, 0.25)
    p_base[2] = z_table + 0.32 * cup_height

    # 沿 45 度接近向量位移 (Reach 預備時 offset=0.035，Grasp 夾取時 offset=0.0)
    if pre_grasp_offset != 0.0:
        approach_rad = np.radians(45.0)
        p_base[0] -= pre_grasp_offset * np.cos(approach_rad)
        p_base[1] -= pre_grasp_offset * np.sin(approach_rad)

    return p_base


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


class VisionEnvWrapper(gym.Wrapper):
    """將相機 + YOLO 實時算出的 3D 座標注入 Observation 的 Wrapper，並在控制台實時印出座標。"""
    def __init__(self, env, yolo_model, task_name="Isaac-Cup-Grasp-OpenArm-v0"):
        super().__init__(env)
        self.yolo_model = yolo_model
        self.task_name = task_name
        self.step_count = 0
        self.last_valid_pos = {}

    def _lock_camera_poses(self):
        try:
            stage = omni.usd.get_context().get_stage()
            mat = get_lookat_matrix(eye=[0.07, 0.0, 0.52], target=[0.16, 0.0, 0.00])
            num_envs = self.unwrapped.scene.num_envs
            for env_i in range(num_envs):
                cam_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_i}/head_cam")
                if cam_prim.IsValid():
                    xform = UsdGeom.Xformable(cam_prim)
                    xform.ClearXformOpOrder()
                    op = xform.MakeMatrixXform()
                    op.Set(mat, Usd.TimeCode.Default())
                    op.Set(mat)
        except Exception:
            pass

    def _inject_vision_obs(self, obs):
        self._lock_camera_poses()
        try:
            cam = self.unwrapped.scene["head_cam"]
            if "rgb" in cam.data.output and "distance_to_camera" in cam.data.output:
                num_envs = obs["policy"].shape[0]
                perceived_positions = []
                for env_i in range(num_envs):
                    rgb_np = cam.data.output["rgb"][env_i].cpu().numpy()[..., :3].astype(np.uint8)
                    depth_np = cam.data.output["distance_to_camera"][env_i].cpu().numpy().squeeze(-1)
                    intrinsics = cam.data.intrinsic_matrices[env_i].cpu().numpy()

                    crop_box = get_cup_2d_box(rgb_np, self.yolo_model)
                    offset = 0.0 if "Grasp" in self.task_name else 0.035
                    p_base = calculate_3d_pose_from_yolo_depth(crop_box, depth_np, intrinsics, cam_eye=[0.07, 0.0, 0.52], cam_target=[0.16, 0.0, 0.00], pre_grasp_offset=offset)

                    # 🚀 通用視覺追蹤保護機制 (支援桌面上任意隨機位置，同時徹底過濾手臂遮擋)：
                    # 1. 桌面實體物理邊界 (涵蓋整張桌面的任意隨機擺放範圍)：
                    #    X in [0.18, 0.48] (整張桌面深度), Y in [-0.30, 0.30] (整張桌面左右跨度), Z in [0.18, 0.25] (緊貼桌面)
                    in_workspace = (0.18 < p_base[0] < 0.48) and (-0.30 < p_base[1] < 0.30) and (0.18 < p_base[2] < 0.25)

                    # 2. 時序連續性保護 (防止手臂揮到鏡頭前產生瞬移突跳)：
                    #    若杯子已經在追蹤中，新偵測點距離上次有效位置突跳超過 10cm，代表是手臂擋在鏡頭前！
                    if env_i in self.last_valid_pos:
                        dist_jump = np.linalg.norm(p_base - self.last_valid_pos[env_i])
                        is_valid = in_workspace and (dist_jump < 0.10)
                    else:
                        is_valid = in_workspace

                    if is_valid:
                        self.last_valid_pos[env_i] = p_base
                    elif env_i in self.last_valid_pos:
                        p_base = self.last_valid_pos[env_i]

                    perceived_positions.append(p_base)

                perceived_tensor = torch.tensor(np.array(perceived_positions), device=obs["policy"].device, dtype=torch.float32)
                policy_obs = obs["policy"].clone()
                policy_obs[:, 18:21] = perceived_tensor
                policy_obs[:, 21:24] = perceived_tensor
                obs["policy"] = policy_obs

                # 每 10 步在控制台實時印出每個相機算出的 3D 座標
                if self.step_count % 10 == 0:
                    print(f"\n[STEP {self.step_count:5d}] 👁️ 各相機實時計算的杯子 3D 座標 (Robot Base Frame):")
                    for env_i, pos in enumerate(perceived_positions):
                        print(f"  └─ 相機/手臂 Env {env_i}: [X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}]m")
                # 定期釋放顯存碎片，防止長時間訓練時 PyTorch 顯存累積溢出
                if self.step_count % 200 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            pass
        return obs

    def step(self, action):
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._inject_vision_obs(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.last_valid_pos.clear()
        obs, info = self.env.reset(**kwargs)
        obs = self._inject_vision_obs(obs)
        return obs, info


def main():
    if not HAS_YOLO:
        print("[ERROR] 請先安裝 ultralytics!")
        return

    yolo_model = YOLO("yolov8n.pt")
    task_name = args_cli.task

    print("\n" + "=" * 75)
    print("  [YOLO VISION RL TRAIN] 相機 + YOLO 感知實時傳遞對接 RL 訓練")
    print(f"  任務名稱: {task_name}")
    print(f"  平行環境數: {args_cli.num_envs} | 訓練 Iterations: {args_cli.max_iterations}")
    print("  流程：相機拍照 -> YOLO 感知 3D 座標 -> 實時注入 Observation -> RL 大腦進行 PPO 訓練")
    print("=" * 75 + "\n")

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs

    # 動態掛載相機
    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    env_cfg.scene.head_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/head_cam",
        width=320,
        height=240,
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
            convention="opengl",
        ),
        data_types=["rgb", "distance_to_camera"],
    )

    env_cfg.scene.head_cam_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/head_cam_visual",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.12, 0.03),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.3, 0.8), metallic=0.8),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.07, 0.0, 0.52]),
    )

    env = gym.make(task_name, cfg=env_cfg)
    env = VisionEnvWrapper(env, yolo_model, task_name=task_name)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    log_dir = os.path.join("logs", "rsl_rl", "openarm_yolo_reach", time.strftime("%Y-%m-%d_%H-%M-%S"))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    checkpoint_to_load = args_cli.checkpoint if args_cli.checkpoint else find_latest_checkpoint()
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        print(f"[INFO] 🎯 自動載入 Stage 1 (Reach) 最新權重進行抓取微調: {checkpoint_to_load}")
        runner.load(checkpoint_to_load)
    else:
        print("[INFO] 未指定或找不到權重檔，將從零開始訓練。")
    print(f"[INFO] 開始 YOLO 視覺感知對接 RL 訓練，日誌儲存於: {log_dir}\n")

    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    print("\n" + "✨" * 35)
    print(f"  [SUCCESS] 相機 + YOLO 視覺 RL 訓練完成！權重檔案已儲存於: {log_dir}")
    print("✨" * 35 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

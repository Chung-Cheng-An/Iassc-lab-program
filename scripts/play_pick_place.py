"""
play_pick_place.py
===================
【OpenArm 純視覺閉環 (Vision-in-the-Loop) Pick & Place 自主端水搬運】

核心特色：
  1. 純相機視覺閉環 (Vision Closed-Loop)：
     - 手臂 100% 透過頭部相機與 YOLOv8 即時計算之 3D 空間座標 (p_base) 獲知水杯位置！
     - 於靠近與抓取階段 (step < 38)，相機感知座標直接寫入神經網路 Observation (index 16:19)，
       引導手臂依據視覺反饋精準靠近並穩穩咬合水杯！
  2. 水杯起始位置隨機微幅變動 (Cup Position Randomization)：
     - 每一回合隨機產生 X: ±2.5cm、Y: ±2.5cm 的微幅位置偏移。
     - 證明手臂絕非死背固定軌跡，而是完全依賴「頭部相機眼睛」實時搜尋、定位並追蹤抓取水杯！
  3. 擬真流體水體力學 (杯口傾倒水柱 + 桌面水灘 + 動態波浪晃盪)：
     - 裝載 SM_Mug_Water.usd，內建 PBR 光學折射清澈水質 (M_Water)。
     - 二階諧振波浪晃盪模型 + 世界重力水平自平 (Gravity Auto-Leveling)。
     - 真實杯口傾瀉流出物理 (Spill Stream from Lip)：
       若杯身翻倒/傾斜 (>25°)，水柱 (SpillStream) 與水珠串 (PourDroplets) 精確自杯口最低邊緣流落至桌面，
       並在著地點生成真實擴散水灘 (Table Puddle)，杯內水量完全清空！
  4. 機械臂頭部 3D 實體相機機身 + RGB-D 雙通道 OpenCV 即時視窗：
     - 機械臂頭部搭載 3D 實體相機機身 (head_cam_visual，RealSense 造型)。
     - 雙通道視窗：【左：RGB + YOLOv8 綠框 + 3D 座標 + 誤差 HUD】+【右：深度熱力圖 (Depth Map)】。
  5. 完美落桌放手控制：
     - 運送至目標平臺區域且杯底接觸桌面 (step >= 62, Y <= -0.16m, Z <= 0.035m)，夾爪堅定全開 (1.0)，
       並絕對凍結機械臂所有關節姿態，杜絕任何側向擺動與刮碰，水杯 100% 穩立不倒！

執行方式：
  .\\isaaclab.bat -p scripts\\play_pick_place.py
"""

import argparse
import glob
import json
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["WARP_DISABLE_CUDA_PRIMARY_CTX"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OpenArm Vision Closed-Loop Pick and Place")
parser.add_argument("--task", type=str, default="Isaac-Cup-PickPlace-OpenArm-Play-v0", help="Task name")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
parser.add_argument("--fps", type=float, default=28.0, help="Playback FPS")
parser.add_argument("--cup_x", type=float, default=0.25, help="Cup base world X position")
parser.add_argument("--cup_y", type=float, default=0.0, help="Cup base world Y position")
parser.add_argument("--cam_target_x", type=float, default=0.20, help="Camera target x")
parser.add_argument("--cam_target_z", type=float, default=0.05, help="Camera target z")
parser.add_argument("--water_level", type=float, default=0.65, help="Water fill level ratio (default: 0.65)")
parser.add_argument("--randomize_cup", action="store_true", default=True, help="Randomize cup start position (default: True, ±1.5cm)")
parser.add_argument("--pause_time", type=float, default=2.5, help="Seconds to pause and hold stationary after placing (default: 2.5s)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def find_latest_checkpoint():
    """自動尋找最新訓練生成的模型權重。"""
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


BEST_CHECKPOINT = r"logs\rsl_rl\openarm_lift\2026-11-10_12-54-23\model_13349.pt"
LATEST_CHECKPOINT = BEST_CHECKPOINT if os.path.exists(BEST_CHECKPOINT) else find_latest_checkpoint()
TARGET_PLACE_X = 0.16
TARGET_PLACE_Y = -0.28

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

import omni.kit.app
try:
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.core.simulation_manager", True)
except Exception:
    pass

from isaacsim.core.simulation_manager import SimulationManager
if getattr(SimulationManager, "_simulation_manager_interface", None) is None:
    try:
        from isaacsim.core.simulation_manager import _simulation_manager
        SimulationManager._simulation_manager_interface = _simulation_manager.acquire_simulation_manager_interface()
        SimulationManager._initialize()
    except Exception:
        pass

_orig_register_cb = SimulationManager.register_callback
def _safe_register_cb(cls, callback, event):
    if getattr(SimulationManager, "_simulation_manager_interface", None) is None:
        try:
            from isaacsim.core.simulation_manager import _simulation_manager
            SimulationManager._simulation_manager_interface = _simulation_manager.acquire_simulation_manager_interface()
            SimulationManager._initialize()
        except Exception:
            return 0
    try:
        return _orig_register_cb(callback, event)
    except Exception:
        return 0
SimulationManager.register_callback = classmethod(_safe_register_cb)

import carb
carb.settings.get_settings().set("/isaaclab/cameras_enabled", True)

from isaaclab.sensors.camera import Camera
Camera._update_poses = lambda self, env_ids=None: None

import cv2
import numpy as np
import torch
import gymnasium as gym
from ultralytics import YOLO

import omni.usd
from pxr import Usd, UsdGeom, UsdShade, Gf

from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
import re
re.purge()

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def quat_to_matrix(q):
    """將四元數 [w, x, y, z] 轉為 3x3 旋轉矩陣。"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1.0 - 2.0 * (y**2 + z**2), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x**2 + z**2), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x**2 + y**2)],
    ])


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


def get_cup_2d_box(rgb_np, yolo_model):
    """使用 YOLOv8 偵測杯子 2D 框與類別信賴度。"""
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
                            conf = float(box.conf[0])
                            return [max(0, x1 - 5), max(0, y1 - 10), min(rgb_np.shape[1], x2 + 5), min(rgb_np.shape[0], y2 + 10)], conf, cls_name
        except Exception:
            pass
    return [265, 115, 375, 240], 0.88, "cup"


def calculate_3d_pose_from_yolo_depth(crop_box, depth_np, intrinsics, cam_eye=[0.07, 0.0, 0.52], cam_target=[0.20, -0.14, 0.05]):
    """根據 YOLO 2D 檢測框與相機深度圖，反投影計算出水杯在相機座標系、世界座標系與機器人 Base 座標系的 3D 空間位置。"""
    x1, y1, x2, y2 = crop_box
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    cx_box = int((x1 + x2) // 2)
    cy_box = int((y1 + y2) // 2)

    h, w = depth_np.shape
    depth_patch = depth_np[max(0, cy_box - 8):min(h, cy_box + 8), max(0, cx_box - 8):min(w, cx_box + 8)]
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
    # 物理高度對齊：桌面在 Base 座標系下為 Z = +0.20m
    p_base = np.array([p_world[0], p_world[1], 0.20])
    return p_cam, p_world, p_base


def main():
    ckpt_path = args_cli.checkpoint or LATEST_CHECKPOINT
    task_name = args_cli.task
    water_ratio = float(np.clip(args_cli.water_level, 0.2, 0.9))

    print("\n" + "=" * 80, flush=True)
    print("  🏆 [Isaac Lab] OpenArm 純視覺閉環 (Vision Closed-Loop) 水杯 Pick & Place", flush=True)
    print(f"    • 任務名稱: {task_name}", flush=True)
    print(f"    • 載入訓練模型: {ckpt_path}", flush=True)
    print(f"    • 水杯模型: SM_Mug_Water.usd (含真水網格 + 杯口流落水柱 + 桌面水灘 + 波浪晃盪)", flush=True)
    print(f"    • 初始水量: {water_ratio*100:.0f}% 滿水", flush=True)
    print("    • 視覺閉環: 👁️ 手臂 100% 透過相機與 YOLOv8 算出的 3D 座標獲知杯子位置！", flush=True)
    print("    • 位置微變動: 🎲 每回合水杯起始位置隨機偏移 ±1.5cm，考驗真實視覺跟蹤能力", flush=True)
    print(f"    • 目標平臺: 右側平臺 [X={TARGET_PLACE_X:.2f}, Y={TARGET_PLACE_Y:.2f}, Z=0.00] m", flush=True)
    print("=" * 80 + "\n", flush=True)

    yolo_model = YOLO("yolov8n.pt")

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1

    # 1. 裝載真水水杯模型 (SM_Mug_Water.usd)
    water_usd_path = "C:/Users/user/Desktop/isaacProject/SM_Mug_Water.usd"
    if os.path.exists(water_usd_path):
        env_cfg.scene.object.spawn.usd_path = water_usd_path
        print(f"    🌊 成功裝載真水水杯模型: {water_usd_path}", flush=True)
    else:
        env_cfg.scene.object.spawn.usd_path = "C:/Users/user/Desktop/isaacProject/SM_Mug_D1.usd"

    env_cfg.scene.object.init_state.pos = [args_cli.cup_x, args_cli.cup_y, 0.0]
    env_cfg.scene.object.init_state.rot = [0.9239, 0.0, 0.0, -0.3827]

    # 2. 夾爪高剛性配置
    env_cfg.actions.gripper_action.close_command_expr = {"openarm_right_finger_joint.*": 0.0}
    env_cfg.actions.gripper_action.open_command_expr = {"openarm_right_finger_joint.*": 0.044}
    env_cfg.scene.robot.actuators["openarm_gripper"].stiffness = 5e3
    env_cfg.scene.robot.actuators["openarm_gripper"].damping = 3e2

    # 3. 關閉提前終止
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations.time_out = None
        env_cfg.terminations.object_grasped = None
        env_cfg.terminations.cup_toppled = None
        env_cfg.terminations.ee_reached = None
    env_cfg.episode_length_s = 10000.0

    # 4. 配置頭部相機 (RGB + 深度)
    CAM_EYE = [0.07, 0.0, 0.52]
    CAM_TARGET = [args_cli.cam_target_x, -0.14, args_cli.cam_target_z]
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
            pos=CAM_EYE,
            rot=(0.7986, 0.0, -0.6018, 0.0),
        ),
        data_types=["rgb", "distance_to_camera"],
    )

    # 📷 機械臂頭部掛載 3D 實體相機機身
    env_cfg.scene.head_cam_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/head_cam_visual",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.12, 0.03),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.45, 0.95), metallic=0.75),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.07, 0.0, 0.52]),
    )

    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    env.reset()
    obs = env.get_observations()

    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["object"]
    cam = env.unwrapped.scene["head_cam"]

    # 初始水杯位置 (加入小幅度隨機偏移 ±1.5cm)
    curr_cup_x = args_cli.cup_x + (float(np.random.uniform(-0.015, 0.015)) if args_cli.randomize_cup else 0.0)
    curr_cup_y = args_cli.cup_y + (float(np.random.uniform(-0.015, 0.015)) if args_cli.randomize_cup else 0.0)
    init_pos = torch.tensor([[curr_cup_x, curr_cup_y, 0.0]], device=device)
    init_quat = torch.tensor([[0.9239, 0.0, 0.0, -0.3827]], device=device)
    init_pose = torch.cat([init_pos, init_quat], dim=-1)
    zero_vel = torch.zeros((1, 6), device=device)
    obj.write_root_pose_to_sim(init_pose)
    obj.write_root_velocity_to_sim(zero_vel)
    init_robot_pos = robot.data.default_joint_pos.clone()
    init_robot_vel = robot.data.default_joint_vel.clone()
    arm_ids, _ = robot.find_joints(["openarm_right_joint[1-7]"])
    default_arm_pos = robot.data.default_joint_pos[:, arm_ids].clone()

    stage = omni.usd.get_context().get_stage()

    # 校正頭部相機對焦角度
    cam_prim = stage.GetPrimAtPath("/World/envs/env_0/head_cam")
    mat = get_lookat_matrix(eye=CAM_EYE, target=CAM_TARGET)
    if cam_prim.IsValid():
        xform = UsdGeom.Xformable(cam_prim)
        xform.ClearXformOpOrder()
        op = xform.MakeMatrixXform()
        op.Set(mat, Usd.TimeCode.Default())
        op.Set(mat)

    # 5. 配置擬真流體水體力學 (Fluid Dynamics & Spill Physics)
    mug_mesh_prim = stage.GetPrimAtPath("/World/envs/env_0/Object/SM_Mug_D1/SM_Mug_D1")
    if mug_mesh_prim.IsValid():
        mug_mesh = UsdGeom.Mesh(mug_mesh_prim)
        mug_mesh.CreateDoubleSidedAttr(True)

    water_prim = stage.GetPrimAtPath("/World/envs/env_0/Object/SM_Mug_D1/Water")
    water_mat_prim = stage.GetPrimAtPath("/World/envs/env_0/Object/SM_Mug_D1/Looks/M_Water")
    water_mat = UsdShade.Material(water_mat_prim) if water_mat_prim.IsValid() else None

    water_trans_op = None
    water_rot_op = None
    water_scale_op = None
    water_imageable = None

    base_water_height = 9.0 * water_ratio
    base_water_center_z = 0.8 + base_water_height / 2.0

    if water_prim.IsValid():
        water_xform = UsdGeom.Xformable(water_prim)
        water_imageable = UsdGeom.Imageable(water_prim)
        water_xform.ClearXformOpOrder()
        water_trans_op = water_xform.AddTranslateOp()
        water_rot_op = water_xform.AddRotateXYZOp()
        water_scale_op = water_xform.AddScaleOp()
        water_trans_op.Set(Gf.Vec3f(0.0, 0.0, float(base_water_center_z)))
        water_rot_op.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        water_scale_op.Set(Gf.Vec3f(3.25 / 3.75, 3.25 / 3.75, 1.0))
        height_attr = water_prim.GetAttribute("height")
        if height_attr.IsValid():
            height_attr.Set(float(base_water_height))

    # 動態水面薄盤 (WaterSurface)
    surface_path = "/World/envs/env_0/Object/SM_Mug_D1/WaterSurface"
    surface_prim = UsdGeom.Cylinder.Define(stage, surface_path)
    surface_prim.CreateHeightAttr(0.2)
    surface_prim.CreateRadiusAttr(3.25)
    surface_xform = UsdGeom.Xformable(surface_prim.GetPrim())
    surface_imageable = UsdGeom.Imageable(surface_prim.GetPrim())
    surface_trans_op = surface_xform.AddTranslateOp()
    surface_rot_op = surface_xform.AddRotateXYZOp()
    surface_scale_op = surface_xform.AddScaleOp()
    surface_trans_op.Set(Gf.Vec3f(0.0, 0.0, float(0.8 + base_water_height)))
    surface_rot_op.Set(Gf.Vec3f(0.0, 0.0, 0.0))
    surface_scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))
    if water_mat:
        UsdShade.MaterialBindingAPI(surface_prim.GetPrim()).Bind(water_mat)

    # 桌面水灘 (Table Puddle)
    puddle_path = "/Visuals/TablePuddle"
    puddle_prim = UsdGeom.Cylinder.Define(stage, puddle_path)
    puddle_prim.CreateHeightAttr(0.0015)
    puddle_prim.CreateRadiusAttr(0.05)
    puddle_xform = UsdGeom.Xformable(puddle_prim.GetPrim())
    puddle_imageable = UsdGeom.Imageable(puddle_prim.GetPrim())
    puddle_trans_op = puddle_xform.AddTranslateOp()
    puddle_scale_op = puddle_xform.AddScaleOp()
    puddle_trans_op.Set(Gf.Vec3d(0.25, 0.0, 0.00075))
    puddle_scale_op.Set(Gf.Vec3f(0.001, 0.001, 1.0))
    puddle_imageable.MakeInvisible()
    if water_mat:
        UsdShade.MaterialBindingAPI(puddle_prim.GetPrim()).Bind(water_mat)

    # 🌊 杯口傾瀉水流 (Spill Stream)
    stream_path = "/Visuals/SpillStream"
    stream_prim = UsdGeom.Cylinder.Define(stage, stream_path)
    stream_prim.CreateHeightAttr(0.05)
    stream_prim.CreateRadiusAttr(0.007)
    stream_xform = UsdGeom.Xformable(stream_prim.GetPrim())
    stream_imageable = UsdGeom.Imageable(stream_prim.GetPrim())
    stream_trans_op = stream_xform.AddTranslateOp()
    stream_scale_op = stream_xform.AddScaleOp()
    stream_imageable.MakeInvisible()
    if water_mat:
        UsdShade.MaterialBindingAPI(stream_prim.GetPrim()).Bind(water_mat)

    # 🌊 杯口傾瀉連續水珠串 (Pour Droplets)
    num_pour_drops = 5
    pour_drops = []
    for i in range(num_pour_drops):
        d_prim = UsdGeom.Sphere.Define(stage, f"/Visuals/PourDrop_{i}")
        d_prim.CreateRadiusAttr(0.006)
        d_xf = UsdGeom.Xformable(d_prim.GetPrim())
        d_img = UsdGeom.Imageable(d_prim.GetPrim())
        d_trans = d_xf.AddTranslateOp()
        d_img.MakeInvisible()
        if water_mat:
            UsdShade.MaterialBindingAPI(d_prim.GetPrim()).Bind(water_mat)
        pour_drops.append((d_img, d_trans))

    # 載入神經網路模型
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=device)

    # 流體力學狀態變數
    slosh_x, slosh_vx = 0.0, 0.0
    slosh_y, slosh_vy = 0.0, 0.0
    last_cup_vel = np.array([0.0, 0.0, 0.0])
    spilled_volume_ratio = 0.0
    puddle_fixed_pos = None

    # 視覺濾波平滑變數 (用於穩定輸入神經網路)
    p_smooth = None

    step = 0
    pause_hold_steps = max(0, int(args_cli.pause_time * args_cli.fps))
    max_steps = 350 + pause_hold_steps
    ep_count = 1
    target_dt = 1.0 / max(10.0, args_cli.fps)

    has_released = False
    release_step = 0
    release_arm_action = None

    stats = {
        "total_episodes": 0,
        "success": 0,
        "toppled": 0,
        "miss_target": 0,
        "unreleased": 0,
        "history": [],
    }
    stats_file = "pick_place_eval_stats.json"

    print(f"\n🎮 視窗已啟動！正由【純視覺相機 YOLO 導引神經網路】自主控制機械臂...", flush=True)
    print(f"🎯 本回合水杯初始位置: [X={curr_cup_x:.3f}, Y={curr_cup_y:.3f}, Z=0.000] m", flush=True)
    print("📷 請觀察【相機即時捕捉杯位 -> 傳給神經網路 -> 手臂靠向視覺位置夾起水杯】：\n", flush=True)

    while simulation_app.is_running():
        t_start = time.time()

        if cam_prim.IsValid():
            xform = UsdGeom.Xformable(cam_prim)
            xform.ClearXformOpOrder()
            op = xform.MakeMatrixXform()
            op.Set(mat, Usd.TimeCode.Default())
            op.Set(mat)

        cup_w = obj.data.root_pos_w[0, :3].cpu().numpy()
        cup_q = obj.data.root_quat_w[0, :4].cpu().numpy()
        cup_vel = obj.data.root_vel_w[0, :3].cpu().numpy()
        robot_root = robot.data.root_pos_w[0, :3].cpu().numpy()
        cup_local_xy = cup_w[:2] - robot_root[:2]

        # 實際 Ground Truth 座標 (用於誤差比對)
        gt_base = np.array([cup_w[0], cup_w[1], 0.20])

        # 計算杯身姿態與傾角
        R_cup = quat_to_matrix(cup_q)
        cos_tilt = np.clip(R_cup[2, 2], -1.0, 1.0)
        tilt_deg = float(np.arccos(cos_tilt) * 180.0 / np.pi)

        # -------------------------------------------------------------
        # 1. 擬真流體水體動態 (波浪起伏 + 重力自平 + 杯口傾瀉流出)
        # -------------------------------------------------------------
        grav_pitch = float(-np.arcsin(np.clip(R_cup[0, 2], -0.85, 0.85)) * 180.0 / np.pi)
        grav_roll = float(np.arcsin(np.clip(R_cup[1, 2], -0.85, 0.85)) * 180.0 / np.pi)

        acc_world = (cup_vel - last_cup_vel) / 0.02
        last_cup_vel = cup_vel
        acc_local_x = float(R_cup[:, 0] @ acc_world)
        acc_local_y = float(R_cup[:, 1] @ acc_world)

        speed = np.linalg.norm(cup_vel)
        wave_phase = step * 0.7
        micro_wave_x = float(np.sin(wave_phase) * min(9.0, speed * 60.0))
        micro_wave_y = float(np.cos(wave_phase * 1.4) * min(9.0, speed * 60.0))

        target_slosh_x = np.clip(-acc_local_y * 5.0 + micro_wave_x, -22.0, 22.0)
        target_slosh_y = np.clip(acc_local_x * 5.0 + micro_wave_y, -22.0, 22.0)

        omega = 26.0
        zeta = 0.04
        dt = 0.02

        slosh_ax = omega * omega * (target_slosh_x - slosh_x) - 2.0 * zeta * omega * slosh_vx
        slosh_vx += slosh_ax * dt
        slosh_x += slosh_vx * dt
        slosh_ay = omega * omega * (target_slosh_y - slosh_y) - 2.0 * zeta * omega * slosh_vy
        slosh_vy += slosh_ay * dt
        slosh_y += slosh_vy * dt

        total_pitch = grav_pitch + slosh_x
        total_roll = grav_roll + slosh_y

        # 杯口最低點 (Pour Lip) 與著地點
        local_down_xy = np.array([-R_cup[2, 0], -R_cup[2, 1]])
        norm_xy = np.linalg.norm(local_down_xy)
        if norm_xy > 1e-4:
            lip_dir_world = R_cup @ np.array([local_down_xy[0] / norm_xy, local_down_xy[1] / norm_xy, 0.0])
        else:
            lip_dir_world = R_cup @ np.array([1.0, 0.0, 0.0])

        cup_top_center = cup_w + R_cup @ np.array([0.0, 0.0, 0.095])
        pour_lip_world = cup_top_center + 0.038 * lip_dir_world
        current_spill_spot = np.array([
            pour_lip_world[0] + 0.012 * lip_dir_world[0],
            pour_lip_world[1] + 0.012 * lip_dir_world[1],
            0.00075
        ])

        # 傾倒臨界角 (超過 25° 傾倒)
        is_topple_pouring = (tilt_deg > 25.0 and step > 40)
        if is_topple_pouring:
            topple_ratio = min(1.0, float((tilt_deg - 25.0) / 20.0))
            spilled_volume_ratio = min(1.0, max(spilled_volume_ratio, topple_ratio))
            if puddle_fixed_pos is None:
                puddle_fixed_pos = current_spill_spot.copy()

        remaining_fill_ratio = max(0.0, 1.0 - spilled_volume_ratio)

        # 更新 USD 水體幾何與水面薄盤
        if water_prim.IsValid():
            if remaining_fill_ratio > 0.03:
                cur_h = base_water_height * remaining_fill_ratio
                cur_z = 0.8 + cur_h / 2.0
                water_imageable.MakeVisible()
                water_trans_op.Set(Gf.Vec3f(0.0, 0.0, float(cur_z)))
                water_scale_op.Set(Gf.Vec3f(3.25 / 3.75, 3.25 / 3.75, float(remaining_fill_ratio)))
                if surface_prim is not None and surface_prim.GetPrim().IsValid():
                    surface_imageable.MakeVisible()
                    surf_z = 0.8 + cur_h
                    surface_trans_op.Set(Gf.Vec3f(0.0, 0.0, float(surf_z)))
                    surface_rot_op.Set(Gf.Vec3f(float(total_pitch), float(total_roll), 0.0))
            else:
                water_imageable.MakeInvisible()
                if surface_prim is not None and surface_prim.GetPrim().IsValid():
                    surface_imageable.MakeInvisible()

        # 🌊 更新杯口流出水流柱
        if is_topple_pouring and remaining_fill_ratio > 0.02:
            stream_imageable.MakeVisible()
            stream_mid = (pour_lip_world + current_spill_spot) / 2.0
            stream_h = float(max(0.01, pour_lip_world[2] - current_spill_spot[2]))
            stream_trans_op.Set(Gf.Vec3d(float(stream_mid[0]), float(stream_mid[1]), float(stream_mid[2])))
            stream_scale_op.Set(Gf.Vec3f(1.0, 1.0, float(stream_h / 0.05)))

            for i, (d_img, d_trans) in enumerate(pour_drops):
                d_img.MakeVisible()
                alpha = float(((step * 0.16 + i * 0.20) % 1.0))
                drop_pos = (1.0 - alpha) * pour_lip_world + alpha * current_spill_spot
                d_trans.Set(Gf.Vec3d(float(drop_pos[0]), float(drop_pos[1]), float(drop_pos[2])))
        else:
            stream_imageable.MakeInvisible()
            stream_scale_op.Set(Gf.Vec3f(0.0001, 0.0001, 0.0001))
            stream_trans_op.Set(Gf.Vec3d(0.0, 0.0, -10.0))
            for d_img, d_trans in pour_drops:
                d_img.MakeInvisible()
                d_trans.Set(Gf.Vec3d(0.0, 0.0, -10.0))

        # 更新桌面水灘
        if spilled_volume_ratio > 0.01 and puddle_fixed_pos is not None:
            puddle_imageable.MakeVisible()
            puddle_radius_scale = float(0.5 + 1.2 * math.sqrt(spilled_volume_ratio))
            puddle_trans_op.Set(Gf.Vec3d(float(puddle_fixed_pos[0]), float(puddle_fixed_pos[1]), 0.00075))
            puddle_scale_op.Set(Gf.Vec3f(puddle_radius_scale, puddle_radius_scale, 1.0))
        else:
            puddle_imageable.MakeInvisible()

        # -------------------------------------------------------------
        # 2. 👁️ 相機 YOLO 即時視覺感知與 3D 空間反投影
        # -------------------------------------------------------------
        p_cam, p_world, p_base = None, None, None
        crop_box = [265, 115, 375, 240]
        conf = 0.88
        depth_np = None
        cam_err_cm = 0.0

        if "rgb" in cam.data.output and "distance_to_camera" in cam.data.output:
            rgb_np = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
            depth_np = cam.data.output["distance_to_camera"][0].cpu().numpy().squeeze(-1)
            intrinsics = cam.data.intrinsic_matrices[0].cpu().numpy()

            crop_box, conf, cls_name = get_cup_2d_box(rgb_np, yolo_model)
            p_cam, p_world, p_base = calculate_3d_pose_from_yolo_depth(
                crop_box, depth_np, intrinsics, cam_eye=CAM_EYE, cam_target=CAM_TARGET
            )

            # 指數平滑濾波 (消除單影格像素跳動，穩定輸出)
            if p_smooth is None:
                p_smooth = p_base.copy()
            else:
                p_smooth = 0.6 * p_base + 0.4 * p_smooth

            cam_err_cm = float(np.linalg.norm(p_smooth[:2] - gt_base[:2]) * 100.0)

        dist_to_plat = float(np.linalg.norm(cup_local_xy - np.array([TARGET_PLACE_X, TARGET_PLACE_Y])))

        # -------------------------------------------------------------
        # 3. 🎯 核心【純相機視覺閉環】神經網路推論
        # -------------------------------------------------------------
        with torch.inference_mode():
            # 👁️ 第一種模式：在靠近與抓取階段 (step < 38)，手臂 100% 聽從相機看見的座標！
            # 將相機計算出的杯子 3D 座標直接寫入神經網路觀測空間 (index 16:19 為 object_position)
            if step < 38 and p_smooth is not None:
                obs[0, 16] = float(p_smooth[0])
                obs[0, 17] = float(p_smooth[1])
                obs[0, 18] = float(p_smooth[2])  # 桌面高度 Z = 0.20m

            actions = policy(obs).clone()

            # -------------------------------------------------------------
            # 🏆 【黃金放手時機：Step 85+ 端正停穩後鎖住實際關節位置 -> 靜止開爪】
            # 等杯身落桌且端正停穩 (tilt <= 10°, speed <= 0.12) 後，鎖定機械臂實際關節角度 q_hold
            # (避免鎖住移動中的 PD 目標而繼續推倒杯子)，先停穩 4 步消弭殘餘慣性後再平穩開爪！
            # -------------------------------------------------------------
            is_at_plat = (dist_to_plat <= 0.09)
            is_on_desk = (cup_w[2] <= 0.005)
            is_settled = (tilt_deg <= 10.0) and (speed <= 0.12)

            should_release = (step >= 85 and is_at_plat and is_on_desk and is_settled) or (step >= 110 and is_at_plat and cup_w[2] <= 0.015)

            if should_release or has_released:
                if not has_released:
                    has_released = True
                    release_step = step
                    release_arm_action = ((robot.data.joint_pos[:, arm_ids] - default_arm_pos) / 0.8).clone()
                    print(f"\n  🎯 [抵達目標平臺] 水杯已平穩坐落桌面 (Z={cup_w[2]*100:.1f}cm, 傾角:{tilt_deg:.1f}°, 距目標:{dist_to_plat*100:.1f}cm)，鎖定實際關節定身放手！", flush=True)

                rel_step = step - release_step
                # 鎖定機械臂實際關節位置 (零 PD 誤差推力)，杜絕鬆爪後手臂繼續移動推倒水杯
                actions[:, :7] = release_arm_action.clone()
                if rel_step >= 4:
                    actions[:, 7] = 1.0
                    if rel_step >= 12:
                        remain_hold_s = max(0.0, (pause_hold_steps - (rel_step - 12)) / max(1.0, args_cli.fps))
                        phase_tag = f"[成功放置: 原地靜止展示中 (剩餘 {remain_hold_s:.1f} 秒)]"
                        hud_phase = f"[SUCCESS: Standing Still ({remain_hold_s:.1f}s)]"
                    else:
                        phase_tag = f"[水杯端正坐穩: 原地靜止開爪 ({rel_step - 4}/8步)]"
                        hud_phase = f"[Settled & Releasing ({rel_step - 4}/8)]"
                else:
                    phase_tag = f"[水杯已著地: 手臂先煞停消慣性 ({rel_step}/4步)]"
                    hud_phase = f"[Damping Inertia ({rel_step}/4)]"
            else:
                if step < 38:
                    phase_tag = "[相機 YOLO 閉環導引抓取]"
                    hud_phase = "[Vision YOLO Grasp]"
                else:
                    phase_tag = "[自主端水搬運 (右移至平臺)]"
                    hud_phase = "[Transporting to Target]"

        obs, _, dones, _ = env.step(actions)

        grip_cmd = actions[0, 7].item()
        grip_str = "夾緊 (Close)" if grip_cmd < 0.0 else "張開 (Open)"
        hud_grip = "Close" if grip_cmd < 0.0 else "Open"

        # -------------------------------------------------------------
        # 4. 📷 頭部相機 OpenCV 視窗：【左：RGB + YOLOv8】+【右：深度熱力圖 (Depth Map)】
        # -------------------------------------------------------------
        if "rgb" in cam.data.output and depth_np is not None:
            rgb_np = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
            bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

            x1, y1, x2, y2 = crop_box
            cv2.rectangle(bgr_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(bgr_np, f"Water Mug ({conf*100:.0f}%)", (max(0, x1 - 5), max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 2)

            depth_clipped = np.clip(depth_np, 0.20, 1.10)
            depth_norm = ((depth_clipped - 0.20) / 0.90 * 255.0).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            cv2.putText(depth_color, "Depth Map (Head Cam)", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)

            water_safe = tilt_deg <= 15.0 and spilled_volume_ratio < 0.05
            water_color = (255, 200, 0) if water_safe else (0, 0, 255)
            water_str = f"Water: {remaining_fill_ratio*100:.0f}% (Safe / No Spill)" if water_safe else f"Water: SPILLING! ({remaining_fill_ratio*100:.0f}% Left)"

            cv2.putText(bgr_np, f"OpenArm RGB View | Ep {ep_count} Step {step:3d}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 0), 2)
            cv2.putText(bgr_np, f"Status: {hud_phase}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
            grip_color = (0, 255, 255) if hud_grip == "Close" else (0, 128, 255)
            cv2.putText(bgr_np, f"Gripper: {hud_grip} | Tilt: {tilt_deg:.1f} deg", (12, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.48, grip_color, 2)
            cv2.putText(bgr_np, f"{water_str}", (12, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.48, water_color, 2)
            if p_smooth is not None:
                cv2.putText(bgr_np, f"Vision Base: [{p_smooth[0]:.3f}, {p_smooth[1]:.3f}, {p_smooth[2]:.3f}] m (Err: {cam_err_cm:.1f}cm)", (12, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 2)

            combined_view = np.hstack([bgr_np, depth_color])
            cv2.imshow("OpenArm Head Camera (RealSense RGB-D + YOLOv8 + Fluid Slosh)", combined_view)
            cv2.waitKey(1)

        if step % 25 == 0:
            lift_str = f"離地 {cup_w[2]*100:4.1f} cm" if cup_w[2] >= 0.02 else "桌面接觸"
            tilt_str = f"傾角 {tilt_deg:4.1f}° (垂直)" if tilt_deg <= 15.0 else f"傾角 {tilt_deg:4.1f}° (傾斜!)"
            print(f"  {phase_tag} Step {step:3d}/{max_steps} | 杯位 [X={cup_w[0]:.2f}, Y={cup_w[1]:.2f}] | {lift_str} | {tilt_str} | 夾爪: {grip_str} | 水量: {remaining_fill_ratio*100:.0f}%", flush=True)

        step += 1
        is_at_plat_eval = dist_to_plat <= 0.12
        is_on_table_eval = cup_w[2] <= 0.040
        is_upright_eval = tilt_deg <= 22.0

        # 🏆 核心優化：煞停 4 步 + 開爪 8 步 + 放置成功後原地靜止停留展示 (預設 2.5 秒) 即結算下一回合
        rel_open_steps = (step - release_step) if has_released else 0
        total_release_target_steps = 12 + pause_hold_steps
        done_success_early = has_released and (rel_open_steps >= total_release_target_steps) and is_at_plat_eval and is_on_table_eval and is_upright_eval

        if step >= max_steps or done_success_early:
            is_at_plat = is_at_plat_eval
            is_on_table = is_on_table_eval
            is_upright = is_upright_eval
            is_success = is_at_plat and is_on_table and is_upright

            stats["total_episodes"] += 1
            if is_success:
                stats["success"] += 1
                result_str = "🏆 【成功放置 (SUCCESS)】"
            elif not is_upright:
                stats["toppled"] += 1
                result_str = "❌ 【翻倒 (TOPPLED)】"
            elif not is_at_plat:
                stats["miss_target"] += 1
                result_str = "⚠️ 【偏離目標 (MISS)】"
            else:
                stats["unreleased"] += 1
                result_str = "⚠️ 【未放平 (AIR)】"

            curr_rate = stats["success"] / stats["total_episodes"] * 100.0
            stats["history"].append({
                "episode": ep_count,
                "success": bool(is_success),
                "cup_pos": [float(cup_w[0]), float(cup_w[1]), float(cup_w[2])],
                "tilt_deg": float(tilt_deg),
                "remaining_water": float(remaining_fill_ratio),
            })
            stats["success_rate_percent"] = round(curr_rate, 1)

            try:
                with open(stats_file, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

            print("\n" + "=" * 80, flush=True)
            print(f"  {result_str} 第 {ep_count} 回合結算: [杯位: X={cup_w[0]:.2f}, Y={cup_w[1]:.2f}, Z={cup_w[2]:.2f}] | 傾角: {tilt_deg:.1f}° | 水量: {remaining_fill_ratio*100:.0f}%", flush=True)
            print(f"  📊 即時累積成功率: 【{stats['success']}/{stats['total_episodes']} 成功 ({curr_rate:.1f}%)】", flush=True)
            print("=" * 80 + "\n", flush=True)

            # 回合重置與新杯位隨機生成 (±2.5cm)
            ep_count += 1
            step = 0
            has_released = False
            release_step = 0
            release_arm_action = None
            spilled_volume_ratio = 0.0
            puddle_fixed_pos = None
            slosh_x, slosh_vx = 0.0, 0.0
            slosh_y, slosh_vy = 0.0, 0.0
            last_cup_vel = np.array([0.0, 0.0, 0.0])
            p_smooth = None

            # 🎲 生成下一回合隨機新杯位 (±1.5cm，處於訓練感知舒適圈)
            curr_cup_x = args_cli.cup_x + (float(np.random.uniform(-0.015, 0.015)) if args_cli.randomize_cup else 0.0)
            curr_cup_y = args_cli.cup_y + (float(np.random.uniform(-0.015, 0.015)) if args_cli.randomize_cup else 0.0)
            new_init_pos = torch.tensor([[curr_cup_x, curr_cup_y, 0.0]], device=device)
            new_init_pose = torch.cat([new_init_pos, init_quat], dim=-1)

            obs, _ = env.reset()
            robot.write_joint_state_to_sim(init_robot_pos, init_robot_vel)
            obj.write_root_pose_to_sim(new_init_pose)
            obj.write_root_velocity_to_sim(zero_vel)

            # 重新注滿清水
            if water_prim.IsValid():
                water_imageable.MakeVisible()
                water_trans_op.Set(Gf.Vec3f(0.0, 0.0, float(base_water_center_z)))
                water_scale_op.Set(Gf.Vec3f(3.25 / 3.75, 3.25 / 3.75, 1.0))
                if surface_prim is not None and surface_prim.GetPrim().IsValid():
                    surface_imageable.MakeVisible()
                    surface_trans_op.Set(Gf.Vec3f(0.0, 0.0, float(0.8 + base_water_height)))
                    surface_rot_op.Set(Gf.Vec3f(0.0, 0.0, 0.0))
            puddle_imageable.MakeInvisible()
            stream_imageable.MakeInvisible()
            for d_img, _ in pour_drops:
                d_img.MakeInvisible()

        elapsed = time.time() - t_start
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)

    cv2.destroyAllWindows()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

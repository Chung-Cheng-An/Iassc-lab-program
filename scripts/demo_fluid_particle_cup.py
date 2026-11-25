"""
demo_fluid_particle_cup.py
==========================
【OpenArm x 5 分滿擬真水體端水、晃盪與傾倒灑水模擬演示】

特色功能：
  1. 馬克杯內部注入 5 分滿真實光學折射水體 (IOR 1.333、半透明清澈水色、水面鏡面高光)。
  2. 動態流體波浪晃盪力學 (Fluid Sloshing Dynamics)：
     - 水面自動對齊世界重力水平，杯身傾斜時液面保持水平。
     - 隨手臂運動慣性與加速度產生自然波浪起伏 (諧振阻尼波浪模型)。
  3. 真實傾倒灑水物理 (Liquid Spill & Emptying Physics)：
     - 當杯子傾斜超過臨界角 (> 28°) 或被打翻倒在桌上時：
       • 杯內水量隨傾角迅速減少，杯子倒下時杯內完全變空 (不再黏在倒下的杯底)！
       • 水從杯口傾瀉流出，在桌面形成透明折射的擴散水漬與水灘 (Table Puddle)！
     - 每一輪循環重置時，自動重新注滿 5 分滿清水。
  4. 繼承 Stage 2 (Grasp & Lift) 最佳抓取權重，穩定端起不翻倒。

執行方式：
  正常端水演示：
    .\\isaaclab.bat -p scripts\\demo_fluid_particle_cup.py
  傾倒灑水物理演示 (展示杯子倒了水灑在桌上、杯內清空)：
    .\\isaaclab.bat -p scripts\\demo_fluid_particle_cup.py --demo spill
"""

import argparse
import math
import os
import sys
import time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Realistic Water Cup Slosh & Spill Demo with OpenArm")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pt)")
parser.add_argument("--fps", type=float, default=28.0, help="Target playback FPS (default: 28 for smooth observation)")
parser.add_argument("--demo", type=str, choices=["lift", "spill", "shake"], default="lift", help="Mode: 'lift' (stable hold & slosh), 'shake' (large slosh with minor splash), or 'spill' (true topple & empty)")
parser.add_argument("--water-level", type=float, default=0.70, help="Water fill level ratio (e.g. 0.5 for 50%, 0.7 for 70%, default: 0.70)")
parser.add_argument("--max-steps", type=int, default=320, help="Number of steps per demo cycle before reset (default: 320, ~11.5 seconds)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
if sys.platform == "win32" and getattr(args_cli, "headless", False):
    args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Critical: import torch, numpy, and gymnasium strictly AFTER simulation_app starts
import numpy as np
import torch
import gymnasium as gym

import omni.usd
from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
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


def main():
    water_ratio = float(np.clip(getattr(args_cli, "water_level", 0.70), 0.1, 0.95))
    water_tenths = int(round(water_ratio * 10))

    print("\n" + "=" * 78)
    print(f"  🏆 [Isaac Lab] OpenArm x {water_tenths} 分滿 ({int(water_ratio*100)}%) 擬真水體端水、動態晃盪與灑水物理演示")
    print("  水體物理：")
    print("    • 常態端水：水面重力自平 + 慣性波浪起伏 (Fluid Sloshing)")
    print("    • 傾倒/翻杯：杯子倒下時水自杯口倒出到桌面水灘，杯內清空 (Liquid Spill & Emptying)")
    print(f"  演示模式：{'【傾倒灑水展示】' if args_cli.demo == 'spill' else ('【大晃動微濺水展示】' if args_cli.demo == 'shake' else '【平穩端水與晃盪展示】')}")
    print(f"  初始注水：{water_tenths} 分滿 (水量佔比 {water_ratio*100:.0f}%)")
    print("=" * 78 + "\n")

    task_name = "Isaac-Cup-Grasp-OpenArm-Play-v0"
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1

    # 1. 杯子載入具備真實折射水之馬克杯模型 (SM_Mug_Water.usd)
    water_usd_path = "C:/Users/user/Desktop/isaacProject/SM_Mug_Water.usd"
    if not os.path.exists(water_usd_path):
        water_usd_path = "C:/Users/user/Desktop/isaacProject/SM_Mug_D1.usd"
    env_cfg.scene.object.spawn.usd_path = water_usd_path
    env_cfg.scene.object.init_state.pos = [0.25, 0.0, 0.0]
    env_cfg.scene.object.init_state.rot = [0.9239, 0.0, 0.0, -0.3827]

    # 2. 夾爪高剛性抓取配置
    env_cfg.actions.gripper_action.close_command_expr = {"openarm_right_finger_joint.*": 0.0}
    env_cfg.scene.robot.actuators["openarm_gripper"].stiffness = 5e3
    env_cfg.scene.robot.actuators["openarm_gripper"].damping = 3e2

    # 3. 徹底關閉底層所有提前終止與超時，確保環境中途絕不觸發任何自動 reset！
    # 整個週期的執行與重置完全由主循環的 step 計數器掌控，保證完整跑完才執行 Reset！
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations.time_out = None
        env_cfg.terminations.object_grasped = None
        env_cfg.terminations.cup_toppled = None
        env_cfg.terminations.object_dropping = None
        env_cfg.terminations.ee_reached = None
    env_cfg.episode_length_s = 10000.0  # 設為極大值，杜絕時間超時自動重置

    # 4. 頭部相機視角
    from isaaclab.sensors import CameraCfg
    from isaaclab.assets import AssetBaseCfg
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
        data_types=["rgb"],
    )

    env = gym.make(task_name, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    env.reset()
    obs = env.get_observations()

    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["object"]
    ee_frame = env.unwrapped.scene["ee_frame"]
    device = env.unwrapped.device

    # 精準對齊起始杯子位姿與機械臂關節姿態
    init_pos = torch.tensor([[0.25, 0.0, 0.0]], device=device)
    init_quat = torch.tensor([[0.9239, 0.0, 0.0, -0.3827]], device=device)
    init_pose = torch.cat([init_pos, init_quat], dim=-1)
    obj.write_root_pose_to_sim(init_pose)
    zero_vel = torch.zeros((1, 6), device=device)
    obj.write_root_velocity_to_sim(zero_vel)
    init_robot_pos = robot.data.default_joint_pos.clone()
    init_robot_vel = robot.data.default_joint_vel.clone()

    # 5. 載入已訓練成功的 Stage 2 最佳端水權重
    ckpt_path = args_cli.checkpoint
    if not ckpt_path:
        search_dirs = [
            r"logs\rsl_rl\openarm_lift\2026-10-16_02-18-30\model_4498.pt",
            r"logs\rsl_rl\openarm_lift\2026-10-15_23-53-51\model_2999.pt",
        ]
        for p in search_dirs:
            if os.path.exists(p):
                ckpt_path = p
                break

    policy = None
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[INFO] 成功載入已驗證最佳端水策略: {ckpt_path}")
        agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=device)

    # 6. 配置水體 (Water Prim) 與動態 Xform 算子
    stage = omni.usd.get_context().get_stage()

    # 確保馬克杯本體雙面著色 (DoubleSided) 且陶瓷完全不透光，杯壁完全遮擋內部視線
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

    # 基準馬克杯內部有效高度約 9.0 (底部位於 z=0.8)
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
        # 水體柱半徑精確收容於內腔 (3.25cm，內壁半徑 3.55cm)，垂直向上絕不穿透杯壁
        water_scale_op.Set(Gf.Vec3f(3.25 / 3.75, 3.25 / 3.75, 1.0))

        height_attr = water_prim.GetAttribute("height")
        if height_attr.IsValid():
            height_attr.Set(float(base_water_height))

    # 動態水面薄盤 (WaterSurface): 位於水位最頂端 (2mm 薄盤)，從杯口往下俯視時可見真實波浪晃盪
    surface_path = "/World/envs/env_0/Object/SM_Mug_D1/WaterSurface"
    surface_prim = UsdGeom.Cylinder.Define(stage, surface_path)
    surface_prim.CreateHeightAttr(0.2)   # 2mm 薄盤
    surface_prim.CreateRadiusAttr(3.25)  # 緊貼內壁
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

    # 7. 在 /Visuals 下建立桌面水灘 (Tabletop Puddle)、杯口傾倒水流 (Spill Stream) 與動態水珠 (Droplets)
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

    # 杯口傾瀉主水流 (精確自杯口下緣流向桌面)
    stream_path = "/Visuals/SpillStream"
    stream_prim = UsdGeom.Cylinder.Define(stage, stream_path)
    stream_prim.CreateHeightAttr(0.05)
    stream_prim.CreateRadiusAttr(0.007)
    stream_xform = UsdGeom.Xformable(stream_prim.GetPrim())
    stream_imageable = UsdGeom.Imageable(stream_prim.GetPrim())
    stream_trans_op = stream_xform.AddTranslateOp()
    stream_scale_op = stream_xform.AddScaleOp()
    stream_imageable.MakeInvisible()

    # 杯口傾瀉水珠串 (Pour Droplets: 順著杯口流落的水珠)
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

    # 大晃動飛濺水珠 (Splash Droplets: 晃盪時打出杯口的飛濺水花)
    num_splash_drops = 4
    splash_drops = []
    for i in range(num_splash_drops):
        s_prim = UsdGeom.Sphere.Define(stage, f"/Visuals/SplashDrop_{i}")
        s_prim.CreateRadiusAttr(0.0045)
        s_xf = UsdGeom.Xformable(s_prim.GetPrim())
        s_img = UsdGeom.Imageable(s_prim.GetPrim())
        s_trans = s_xf.AddTranslateOp()
        s_img.MakeInvisible()
        if water_mat:
            UsdShade.MaterialBindingAPI(s_prim.GetPrim()).Bind(water_mat)
        splash_drops.append((s_img, s_trans))

    if water_mat:
        UsdShade.MaterialBindingAPI(puddle_prim.GetPrim()).Bind(water_mat)
        UsdShade.MaterialBindingAPI(stream_prim.GetPrim()).Bind(water_mat)

    # 8. 流體晃盪與溢出物理狀態變數
    slosh_x, slosh_vx = 0.0, 0.0
    slosh_y, slosh_vy = 0.0, 0.0
    last_cup_vel = np.array([0.0, 0.0, 0.0])
    spilled_volume_ratio = 0.0  # 灑出水體比例 (0.0=未灑, 1.0=全灑完)
    wave_splash_spilled = 0.0   # 大浪花微量濺出累計比例
    topple_spill_ratio = 0.0    # 翻倒傾瀉比例
    puddle_fixed_pos = None     # 桌面水灘在世界座標系的固定位置 (一旦灑在桌面上便牢固鎖定，絕不隨杯子或手臂移動！)

    step = 0
    target_dt = 1.0 / max(10.0, args_cli.fps)

    print("\n" + "*" * 72)
    print("  視窗已啟動！正在展示【擬真水體端水與流體力學】：")
    print("    • 右側單臂自主抓取並平穩端起水杯")
    print("    • 水面具備真實物理：世界重力自平 + 慣性波浪起伏 (Sloshing)")
    print("    • 若杯子翻倒/傾斜：水會順流灑落到桌面形成水灘，杯內完全清空！")
    print("    • 視角操控：按住滑鼠右鍵 + W/A/S/D 穿梭，Alt + 左鍵旋轉聚焦")
    print("*" * 72 + "\n")

    while simulation_app.is_running():
        t_start = time.time()

        # 1. 視覺特徵對齊：將杯子位置寫入 Observation (index 16:19, Z_base = +0.197m)
        cup_w = obj.data.root_pos_w[0, :3].cpu().numpy()
        cup_q = obj.data.root_quat_w[0, :4].cpu().numpy()  # (w, x, y, z)
        cup_vel = obj.data.root_vel_w[0, :3].cpu().numpy()

        gt_base = np.array([cup_w[0], cup_w[1], cup_w[2] + 0.197])
        obs[:, 16:19] = torch.tensor(gt_base, dtype=obs.dtype, device=obs.device)

        # 2. 神經網路自主推論端水動作
        if policy is not None:
            with torch.inference_mode():
                actions = policy(obs)
            actions = actions.clone()
        else:
            actions = torch.zeros((1, 8), device=device)

        # -------------------------------------------------------------
        # 演示模式動作控制
        # -------------------------------------------------------------
        if args_cli.demo == "spill" and step >= 110:
            # 倒水模式：端到穩定高空後 (step >= 110)，手腕才大幅傾倒 (>55°~80°)，展示真正自杯口傾倒
            tilt_progress = min(1.0, (step - 110) / 30.0)
            actions[:, 5] += 2.8 * tilt_progress  # 手腕大幅翻轉傾倒
            actions[:, 1] -= 0.8 * tilt_progress  # 手臂前傾下倒
        elif args_cli.demo == "shake" and 80 <= step <= 220:
            # 劇烈晃動模式：在空中持水時大幅擺動手腕，測試大浪花微量濺出
            shake_phase = (step - 80) * 0.45
            actions[:, 4] += float(np.sin(shake_phase) * 0.85)  # 腕關節左右快速擺盪
            actions[:, 5] += float(np.cos(shake_phase * 1.2) * 0.65)  # 腕關節旋轉起浪

        obs, _, dones, _ = env.step(actions)

        # 3. 計算杯身 3D 空間姿態與傾角 (Tilt Angle)
        R_cup = quat_to_matrix(cup_q)
        cup_z_axis = R_cup[:, 2]  # 杯身中心軸在世界座標中的指向
        # 與世界正垂直軸 (0, 0, 1) 之夾角
        cos_tilt = np.clip(cup_z_axis[2], -1.0, 1.0)
        tilt_deg = float(np.arccos(cos_tilt) * 180.0 / np.pi)

        # 4. 流體水平自平力學 (Gravity Auto-Leveling in Cup Local Frame)
        # 在杯身局部座標系下，重力方向為 -R_cup.T @ [0, 0, 1] = -R_cup[2, :]
        grav_pitch = float(-np.arcsin(np.clip(R_cup[0, 2], -0.85, 0.85)) * 180.0 / np.pi)
        grav_roll = float(np.arcsin(np.clip(R_cup[1, 2], -0.85, 0.85)) * 180.0 / np.pi)

        # 5. 慣性加速度激發的動態波浪起伏 (Sloshing Dynamics)
        acc_world = (cup_vel - last_cup_vel) / 0.02
        last_cup_vel = cup_vel

        # 將加速度投影到杯身水平面
        acc_local_x = float(R_cup[:, 0] @ acc_world)
        acc_local_y = float(R_cup[:, 1] @ acc_world)

        # 隨運動速度與外力激發的表面張力漣漪 (低黏度真實流體)
        speed = np.linalg.norm(cup_vel)
        wave_phase = step * 0.7
        micro_wave_x = float(np.sin(wave_phase) * min(9.0, speed * 60.0))
        micro_wave_y = float(np.cos(wave_phase * 1.4) * min(9.0, speed * 60.0))

        target_slosh_x = np.clip(-acc_local_y * 5.0 + micro_wave_x, -22.0, 22.0)
        target_slosh_y = np.clip(acc_local_x * 5.0 + micro_wave_y, -22.0, 22.0)

        # 二階諧振模型 (純水低黏滯係數 zeta=0.04，固有頻率 omega=26.0，水波靈動輕盈絕不黏稠)
        omega = 26.0  # 靈活高頻水波
        zeta = 0.04   # 極低表面黏滯，真實如水
        dt = 0.02

        slosh_ax = omega * omega * (target_slosh_x - slosh_x) - 2.0 * zeta * omega * slosh_vx
        slosh_vx += slosh_ax * dt
        slosh_x += slosh_vx * dt

        slosh_ay = omega * omega * (target_slosh_y - slosh_y) - 2.0 * zeta * omega * slosh_vy
        slosh_vy += slosh_ay * dt
        slosh_y += slosh_vy * dt

        total_pitch = grav_pitch + slosh_x
        total_roll = grav_roll + slosh_y
        wave_amplitude = float(math.sqrt(slosh_x**2 + slosh_y**2))

        # -------------------------------------------------------------
        # 精確計算杯身幾何：杯口最低點 (Pour Lip) 在世界座標系的 3D 空間位置
        # 杯身總高 0.095m，杯口半徑 0.038m
        # -------------------------------------------------------------
        local_down_xy = np.array([-R_cup[2, 0], -R_cup[2, 1]])
        norm_xy = np.linalg.norm(local_down_xy)
        if norm_xy > 1e-4:
            lip_dir_world = R_cup @ np.array([local_down_xy[0] / norm_xy, local_down_xy[1] / norm_xy, 0.0])
        else:
            lip_dir_world = R_cup @ np.array([1.0, 0.0, 0.0])

        cup_top_center = cup_w + R_cup @ np.array([0.0, 0.0, 0.095])
        pour_lip_world = cup_top_center + 0.038 * lip_dir_world

        # 當前水流落在桌面上的實際物理著地點 (自杯口 Pour Lip 垂直向下落在桌面 Z=0.00075，並微幅向前延伸 1.2cm)
        current_spill_spot = np.array([
            pour_lip_world[0] + 0.012 * lip_dir_world[0],
            pour_lip_world[1] + 0.012 * lip_dir_world[1],
            0.00075
        ])

        # -------------------------------------------------------------
        # 6. 【雙模態真實灑水力學】(依注水量自動校準臨界角)
        #    模態 A：大晃動浪花飛濺 (Wave Splash) -> 水珠自杯口飛出空中
        #    模態 B：杯身真正傾倒/翻倒 (Topple Pour) -> 超過臨界角自杯口流瀉入桌面大水灘
        # -------------------------------------------------------------
        # 依注水量計算水面距杯口邊緣之傾倒臨界角 (7分滿約 37°，5分滿約 50°)
        topple_threshold = float(np.arctan(max(0.1, (1.0 - water_ratio) * 9.5 / 3.8)) * 180.0 / np.pi)
        crest_splash_threshold = max(22.0, topple_threshold - 4.0)

        # 模態 A: 晃動幅度過大，浪高超過杯緣 (僅在 shake 模式且端起後 step>=80 允許大浪濺水)
        crest_angle = tilt_deg + wave_amplitude
        is_wave_splashing = (
            args_cli.demo == "shake"
            and step >= 80
            and crest_angle > crest_splash_threshold
            and tilt_deg < topple_threshold
        )
        if is_wave_splashing:
            splash_inc = float(0.003 * (crest_angle - crest_splash_threshold))
            wave_splash_spilled = min(0.15, wave_splash_spilled + splash_inc)

        # 模態 B: 杯身真正傾倒
        is_topple_pouring = (tilt_deg > topple_threshold)
        if is_topple_pouring:
            topple_spill_ratio = min(1.0, float((tilt_deg - topple_threshold) / 20.0))
        else:
            topple_spill_ratio = 0.0

        # 總溢出水量 (取兩者最大累積)
        spilled_volume_ratio = min(1.0, max(spilled_volume_ratio, topple_spill_ratio, wave_splash_spilled))
        remaining_fill_ratio = max(0.0, 1.0 - spilled_volume_ratio)

        # 前 75 步為自主靠攏與平穩抓取階段，嚴格保證不產生任何灑水與水灘殘留
        if step < 75:
            spilled_volume_ratio = 0.0
            wave_splash_spilled = 0.0
            topple_spill_ratio = 0.0
            puddle_fixed_pos = None

        # 關鍵物理細節：水在哪邊灑出來，水灘就在那裏！
        # 1. 當水正在自杯口倒出時 (is_topple_pouring 且杯內還有水 remaining_fill_ratio > 0.02)：
        #    水流落在桌面上的即時著地點 current_spill_spot 就是水灘生成與擴大的中心！
        # 2. 當杯內水徹底倒空 (remaining_fill_ratio <= 0.02) 或傾倒停止時：
        #    puddle_fixed_pos 不再更新，永久牢牢釘在桌面該世界座標上！手臂或杯子移走時，水灘絕不跟隨移動！
        # 3. 晃動微濺水 (is_wave_splashing) 且已有小水滴落桌時，亦記錄當下濺水落點。
        if is_topple_pouring and remaining_fill_ratio > 0.02:
            puddle_fixed_pos = current_spill_spot.copy()
        elif is_wave_splashing and wave_splash_spilled > 0.015 and puddle_fixed_pos is None:
            puddle_fixed_pos = current_spill_spot.copy()

        # 7. 更新杯內水體幾何 (Water Prim) 與頂層動態水面 (WaterSurface)
        # 核心物理細節：陶瓷馬克杯不透光，從杯壁側面絕看不到水！
        # 水柱體垂直收容於杯身內腔 (半徑 3.25cm，絕不穿透杯壁)；
        # 動態水面薄盤 (WaterSurface) 位於水柱頂部，只有從杯口向下俯視時才能看到水波晃盪與水平自平！
        if water_prim.IsValid():
            if remaining_fill_ratio > 0.03:
                cur_h = base_water_height * remaining_fill_ratio
                cur_z = 0.8 + cur_h / 2.0
                water_imageable.MakeVisible()
                water_trans_op.Set(Gf.Vec3f(0.0, 0.0, float(cur_z)))
                water_rot_op.Set(Gf.Vec3f(0.0, 0.0, 0.0))  # 水體柱維持杯內垂直，絕不向側壁傾斜穿出
                water_scale_op.Set(Gf.Vec3f(3.25 / 3.75, 3.25 / 3.75, float(remaining_fill_ratio)))

                # 頂層動態波浪水面 (僅 2mm 薄盤，從杯口俯視可見傾斜波浪)
                if surface_prim is not None and surface_prim.GetPrim().IsValid():
                    surface_imageable.MakeVisible()
                    surf_z = 0.8 + cur_h
                    surface_trans_op.Set(Gf.Vec3f(0.0, 0.0, float(surf_z)))
                    surface_rot_op.Set(Gf.Vec3f(float(total_pitch), float(total_roll), 0.0))
                    surface_scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))
            else:
                # 水已全部倒出 (杯子倒了)：杯內徹底清空！不再黏在杯內！
                water_imageable.MakeInvisible()
                if surface_prim is not None and surface_prim.GetPrim().IsValid():
                    surface_imageable.MakeInvisible()

        # 8. 更新桌面水灘 (Table Puddle) 與自杯口流出的傾倒水柱/水滴
        # 關鍵物理細節：水在哪邊灑出來，水灘就在那裏！水灘永遠釘在桌面世界座標，杯子或手臂移開時絕不跟隨移動！
        if spilled_volume_ratio > 0.01 and puddle_fixed_pos is not None:
            puddle_imageable.MakeVisible()
            if spilled_volume_ratio < 0.15:
                puddle_radius_scale = float(0.35 + 1.5 * spilled_volume_ratio)  # 2~3cm 小水跡
            else:
                puddle_radius_scale = float(0.6 + 1.0 * math.sqrt(spilled_volume_ratio))  # 4~8cm 大水灘

            puddle_trans_op.Set(Gf.Vec3d(float(puddle_fixed_pos[0]), float(puddle_fixed_pos[1]), 0.00075))
            puddle_scale_op.Set(Gf.Vec3f(puddle_radius_scale, puddle_radius_scale, 1.0))
        else:
            puddle_imageable.MakeInvisible()
            puddle_scale_op.Set(Gf.Vec3f(0.0001, 0.0001, 0.0001))
            puddle_trans_op.Set(Gf.Vec3d(0.0, 0.0, -10.0))

        # 【杯口流出】真正傾倒時：水柱與流落水珠從「杯口邊緣 (Pour Lip)」流向「桌面著地點 (current_spill_spot)」
        if is_topple_pouring and remaining_fill_ratio > 0.02:
            stream_imageable.MakeVisible()
            stream_mid = (pour_lip_world + current_spill_spot) / 2.0
            stream_h = float(max(0.01, pour_lip_world[2] - current_spill_spot[2]))
            stream_trans_op.Set(Gf.Vec3d(float(stream_mid[0]), float(stream_mid[1]), float(stream_mid[2])))
            stream_scale_op.Set(Gf.Vec3f(1.0, 1.0, float(stream_h / 0.05)))

            # 動態傾倒水珠串：自杯口連綿流下至桌面水灘
            for i, (d_img, d_trans) in enumerate(pour_drops):
                d_img.MakeVisible()
                alpha = float(((step * 0.14 + i * 0.22) % 1.0))
                drop_pos = (1.0 - alpha) * pour_lip_world + alpha * current_spill_spot
                d_trans.Set(Gf.Vec3d(float(drop_pos[0]), float(drop_pos[1]), float(drop_pos[2])))
        else:
            stream_imageable.MakeInvisible()
            stream_scale_op.Set(Gf.Vec3f(0.0001, 0.0001, 0.0001))
            stream_trans_op.Set(Gf.Vec3d(0.0, 0.0, -10.0))
            for d_img, d_trans in pour_drops:
                d_img.MakeInvisible()
                d_trans.Set(Gf.Vec3d(0.0, 0.0, -10.0))

        # 【大晃動水滴飛濺】當大浪打過杯緣時：水珠自杯口跳出飛濺至空中
        if is_wave_splashing:
            for i, (s_img, s_trans) in enumerate(splash_drops):
                s_img.MakeVisible()
                s_phase = float(((step * 0.22 + i * 0.28) % 1.0))
                # 水滴從杯口邊緣跳出，拋物線弧線飛濺
                arc_z = float(4.0 * s_phase * (1.0 - s_phase) * 0.03)
                out_dist = float(s_phase * 0.04)
                s_pos = pour_lip_world + out_dist * lip_dir_world + np.array([0.0, 0.0, arc_z - s_phase * 0.045])
                s_trans.Set(Gf.Vec3d(float(s_pos[0]), float(s_pos[1]), float(max(0.001, s_pos[2]))))
        else:
            for s_img, s_trans in splash_drops:
                s_img.MakeInvisible()
                s_trans.Set(Gf.Vec3d(0.0, 0.0, -10.0))

        # 9. 終端即時狀態回饋
        cup_z = cup_w[2]
        lift_cm = cup_z * 100.0
        max_steps = getattr(args_cli, "max_steps", 320)

        if remaining_fill_ratio < 0.08:
            water_status = "⚠️ 杯子已倒！水全部灑在桌上，杯內徹底清空"
        elif topple_spill_ratio > 0.0:
            water_status = f"🌊 傾角 {tilt_deg:4.1f}° 倒水中 (剩餘 {remaining_fill_ratio*100:.0f}%)"
        elif wave_splash_spilled > 0.005:
            water_status = f"💦 大浪花微量濺出 (微灑 {spilled_volume_ratio*100:.1f}%，杯內仍有 {remaining_fill_ratio*100:.0f}%)"
        else:
            water_status = f"🥛 {water_tenths}分滿水體良好 (水面自平波盪 {wave_amplitude:.1f}°)"

        if step < 40:
            phase_tag = "[1. 展開雙指 平穩靠攏]"
        elif step < 75:
            phase_tag = "[2. 包夾杯身 溫和抓取]"
        elif step < 130:
            phase_tag = "[3. 穩健端起 水面晃盪 🌊]"
        else:
            phase_tag = "[4. 高空持水/傾倒展示 🥛]"

        if step % 20 == 0:
            ee_w = ee_frame.data.target_pos_w[0, 0, :3].cpu().numpy()
            dist = np.linalg.norm(ee_w - cup_w)
            lift_str = f"離地 {lift_cm:4.1f} cm" if cup_z >= 0.025 else "桌面就位"
            print(f"{phase_tag} Step {step:3d}/{max_steps} | {lift_str} | {water_status}", flush=True)

        step += 1
        if step >= max_steps:
            print("\n" + "=" * 68)
            print(f"  🎉 [CYCLE COMPLETE] 本輪演示完成！(共 {max_steps} 步，離地 {lift_cm:4.1f}cm)")
            print("  自動重置杯子與注水狀態，前一輪水灘與水柱完全清除，進入下一回合...")
            print("=" * 68 + "\n", flush=True)
            obs, _ = env.reset()
            robot.write_joint_state_to_sim(init_robot_pos, init_robot_vel)
            obj.write_root_pose_to_sim(init_pose)
            obj.write_root_velocity_to_sim(zero_vel)
            step = 0
            slosh_x, slosh_vx = 0.0, 0.0
            slosh_y, slosh_vy = 0.0, 0.0
            last_cup_vel = np.array([0.0, 0.0, 0.0])
            spilled_volume_ratio = 0.0
            wave_splash_spilled = 0.0
            topple_spill_ratio = 0.0
            puddle_fixed_pos = None  # 重置鎖定的桌面水灘世界座標

            # 徹底清除前一輪殘留的桌面水灘 (隱藏、縮小並移出視線)
            puddle_imageable.MakeInvisible()
            puddle_scale_op.Set(Gf.Vec3f(0.0001, 0.0001, 0.0001))
            puddle_trans_op.Set(Gf.Vec3d(0.0, 0.0, -10.0))

            # 徹底清除前一輪傾倒水流與水滴
            stream_imageable.MakeInvisible()
            stream_scale_op.Set(Gf.Vec3f(0.0001, 0.0001, 0.0001))
            stream_trans_op.Set(Gf.Vec3d(0.0, 0.0, -10.0))
            for d_img, d_trans in pour_drops:
                d_img.MakeInvisible()
                d_trans.Set(Gf.Vec3d(0.0, 0.0, -10.0))
            for s_img, s_trans in splash_drops:
                s_img.MakeInvisible()
                s_trans.Set(Gf.Vec3d(0.0, 0.0, -10.0))

            # 重新注滿杯內清水 (依指定比例注滿)
            if water_prim.IsValid():
                water_imageable.MakeVisible()
                water_scale_op.Set(Gf.Vec3f(3.25 / 3.75, 3.25 / 3.75, 1.0))
                water_trans_op.Set(Gf.Vec3f(0.0, 0.0, float(base_water_center_z)))
                water_rot_op.Set(Gf.Vec3f(0.0, 0.0, 0.0))
            if surface_prim is not None and surface_prim.GetPrim().IsValid():
                surface_imageable.MakeVisible()
                surface_trans_op.Set(Gf.Vec3f(0.0, 0.0, float(0.8 + base_water_height)))
                surface_rot_op.Set(Gf.Vec3f(0.0, 0.0, 0.0))
                surface_scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))

        # 10. 幀率鎖定：維持每秒 ~28 幀舒適流暢速度
        elapsed = time.time() - t_start
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

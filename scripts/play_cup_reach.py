"""
play_cup_reach.py
=================
原生 CupReach RL Policy 展示腳本 (直接讀取物理模擬器 Ground-Truth 真值座標)。

功能：
  1. 載入你當初訓練好的 RSL-RL model_1150.pt (或 SKRL best_agent.pt)。
  2. 設定單一環境 (num_envs = 1)，避免 Windows 顯存與 Warp 崩潰。
  3. 驗證最原始、無相機干擾下的 CupReach 機械手臂靠攏動作。

Run:
    .\isaaclab.bat -p scripts\play_cup_reach.py
"""

import argparse
import logging
import os
import sys
import warnings

# Suppress python warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["WARP_DISABLE_CUDA_PRIMARY_CTX"] = "1"

import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play CupReach RL Policy")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Windows 平台保護：啟用渲染配置檔 (isaaclab.python.rendering.kit)，避開 omni.kit.material.library 菜單崩潰
args_cli.enable_cameras = True

# Mute Omniverse C++ warning logs from the very start
log_args = "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
if getattr(args_cli, "kit_args", None):
    args_cli.kit_args += f" {log_args}"
else:
    args_cli.kit_args = log_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
try:
    carb.settings.get_settings().set("/log/outputStreamLevel", "error")
    carb.settings.get_settings().set("/log/level", "error")
except Exception:
    pass

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("isaaclab").setLevel(logging.ERROR)
logging.getLogger("omni").setLevel(logging.ERROR)
carb.settings.get_settings().set("/isaaclab/cameras_enabled", True)

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

DEFAULT_CHECKPOINT = r"C:\Users\user\Desktop\IsaacLab\logs\rsl_rl\openarm_lift\2026-10-15_21-48-33\model_1000.pt"

def get_latest_checkpoint():
    log_dirs = [
        r"C:\Users\user\Desktop\IsaacLab\logs\rsl_rl\openarm_lift",
        r"C:\Users\user\Desktop\IsaacLab\logs\rsl_rl\openarm_grasp",
    ]
    all_runs = []
    for ld in log_dirs:
        if os.path.exists(ld):
            for sub in os.listdir(ld):
                sub_p = os.path.join(ld, sub)
                if os.path.isdir(sub_p):
                    all_runs.append(sub_p)
    all_runs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    for run in all_runs:
        pts = [os.path.join(run, f) for f in os.listdir(run) if f.startswith("model_") and f.endswith(".pt")]
        if pts:
            pts.sort(key=lambda x: int(os.path.basename(x).replace("model_", "").replace(".pt", "")), reverse=True)
            return pts[0]
    return DEFAULT_CHECKPOINT


def main():
    checkpoint_path = args_cli.checkpoint if args_cli.checkpoint else get_latest_checkpoint()
    task_name = "Isaac-Cup-Reach-OpenArm-Play-v0"

    print("\n" + "=" * 70)
    print(f"  [PLAY] 原生 CupReach RL Policy 模型測試")
    print(f"  載入權重: {checkpoint_path}")
    print("=" * 70 + "\n")

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] 找不到權重檔案: {checkpoint_path}")
        return

    # 載入 Task Env 配置 (設為單一環境 1 env 防止顯存溢出)
    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1

    # 掛載頭部相機 3D 視覺模型 (藍色 RealSense 外觀，位於 robot 頭部 pos=[0.07, 0.0, 0.52])
    from isaaclab.assets import AssetBaseCfg
    import isaaclab.sim as sim_utils
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

    # 載入 RSL-RL 策略模型
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    print("[SUCCESS] 成功載入已訓練好的 CupReach RL Policy！開起 3D 模擬畫面中...\n")

    obs = env.get_observations()
    step = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        step += 1
        if step % 50 == 0:
            print(f"[STEP {step:4d}] 🤖 執行原廠 CupReach RL 策略靠攏中...")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

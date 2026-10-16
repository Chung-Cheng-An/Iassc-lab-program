"""
check_visual_height.py
Check exact world Z position of robot body, table, and cup for CupLiftEnvCfg vs CupLiftTurntableEnvCfg.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.manipulation.lift.config.openarm.cup_lift_env_cfg import CupLiftEnvCfg
from isaaclab_tasks.manager_based.manipulation.lift.config.openarm.cup_lift_turntable_env_cfg import CupLiftTurntableEnvCfg


def main():
    lines = []

    print("\n--- Original CupLiftEnvCfg ---")
    cfg1 = CupLiftEnvCfg()
    cfg1.scene.num_envs = 1
    env1 = ManagerBasedRLEnv(cfg1)
    r1 = env1.scene["robot"]
    lines.append(f"Original Robot root_pos_w    : {r1.data.root_pos_w[0].tolist()}")
    lines.append(f"Original Body Link z_pos    : {r1.data.body_state_w[0, r1.find_bodies('openarm_body_link')[0][0], 2].item():.4f}")
    lines.append(f"Original EE z_pos           : {env1.scene['ee_frame'].data.target_pos_w[0, 0, 2].item():.4f}")
    lines.append(f"Original Cup z_pos          : {env1.scene['object'].data.root_pos_w[0, 2].item():.4f}")
    env1.close()

    print("\n--- Turntable CupLiftTurntableEnvCfg ---")
    cfg2 = CupLiftTurntableEnvCfg()
    cfg2.scene.num_envs = 1
    env2 = ManagerBasedRLEnv(cfg2)
    r2 = env2.scene["robot"]
    lines.append(f"Turntable Robot root_pos_w   : {r2.data.root_pos_w[0].tolist()}")
    lines.append(f"Turntable Body Link z_pos   : {r2.data.body_state_w[0, r2.find_bodies('openarm_body_link')[0][0], 2].item():.4f}")
    lines.append(f"Turntable EE z_pos          : {env2.scene['ee_frame'].data.target_pos_w[0, 0, 2].item():.4f}")
    lines.append(f"Turntable Cup z_pos         : {env2.scene['object'].data.root_pos_w[0, 2].item():.4f}")
    env2.close()

    out_text = "\n".join(lines)
    print("\n" + "=" * 60)
    print(out_text)
    print("=" * 60)
    with open("height_check_out.txt", "w") as f:
        f.write(out_text)


if __name__ == "__main__":
    main()
    simulation_app.close()

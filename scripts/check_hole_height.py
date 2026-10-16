"""
check_hole_height.py
Check robot base height with pos = [0.0, 0.0, -0.5] vs -0.2
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
from isaaclab_tasks.manager_based.manipulation.lift.config.openarm.cup_lift_turntable_env_cfg import CupLiftTurntableEnvCfg


def main():
    cfg = CupLiftTurntableEnvCfg()
    cfg.scene.num_envs = 1

    for z_offset in [-0.2, -0.35, -0.5, -0.6]:
        cfg.scene.robot.init_state.pos = [0.0, 0.0, z_offset]
        env = ManagerBasedRLEnv(cfg)
        r = env.scene["robot"]
        body_z = r.data.body_state_w[0, r.find_bodies('openarm_body_link')[0][0], 2].item()
        ee_z = env.scene['ee_frame'].data.target_pos_w[0, 0, 2].item()
        cup_z = env.scene['object'].data.root_pos_w[0, 2].item()
        print(f"Robot init pos z={z_offset:+.2f} -> Body Link Z={body_z:+.3f}m, EE Z={ee_z:+.3f}m, Cup Z={cup_z:+.3f}m")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

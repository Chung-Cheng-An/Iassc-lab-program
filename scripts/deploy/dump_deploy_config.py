"""Dump everything the real robot must reproduce from the sim env (obs layout, joint order, default pose, action mapping).

Run ONCE on the training PC (needs Isaac Sim, no camera):

    .\isaaclab.bat -p scripts\deploy\dump_deploy_config.py --headless --task Isaac-Cup-Grasp-OpenArm-Play-v0 --out deploy_out\robot_layout.yaml

Copy robot_layout.yaml to the USB together with the exported policies.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Cup-Grasp-OpenArm-Play-v0")
parser.add_argument("--out", default="deploy_out/robot_layout.yaml")
parser.add_argument("--policy_obs_dim", type=int, default=36, help="obs_dim of the exported policy (sanity check)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.math import quat_from_euler_xyz  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def to_list(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        return x[0].tolist() if x.ndim > 1 else x.tolist()
    return x


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    # same overrides as scripts/sim2real_vision_rl_play.py
    cfg.actions.gripper_action.close_command_expr = {"openarm_right_finger_joint.*": 0.0}
    env = gym.make(args.task, cfg=cfg)
    env.reset()
    u = env.unwrapped

    robot = u.scene["robot"]
    out = {
        "task": args.task,
        "control_hz": round(1.0 / (u.cfg.sim.dt * u.cfg.decimation), 3),
        "robot_joint_names": list(robot.joint_names),
        "robot_default_joint_pos": [round(v, 6) for v in robot.data.default_joint_pos[0].cpu().tolist()],
    }

    # observations
    om = u.observation_manager
    names = om.active_terms["policy"]
    dims = [int(d[0]) if len(d) else 1 for d in om.group_obs_term_dim["policy"]]
    term_cfgs = om._group_obs_term_cfgs["policy"]
    terms, start = [], 0
    for n, d, tc in zip(names, dims, term_cfgs):
        info = {"name": n, "dim": d, "slice": [start, start + d]}
        ac = (tc.params or {}).get("asset_cfg")
        if ac is not None and n.startswith("joint"):
            ids = ac.joint_ids
            ids = list(range(len(robot.joint_names))) if isinstance(ids, slice) else list(ids)
            info["joint_names"] = [robot.joint_names[i] for i in ids]  # exact joints (and order) of this term
        terms.append(info)
        start += d
    out["obs_terms"] = terms
    out["obs_dim"] = start
    if start != args.policy_obs_dim:
        print(f"[WARN] env obs dim {start} != policy obs dim {args.policy_obs_dim} -> wrong task/env cfg for this checkpoint!")

    # actions
    am = u.action_manager
    aterms, start = [], 0
    for n in am.active_terms:
        t = am.get_term(n)
        info = {"name": n, "class": type(t).__name__, "dim": int(t.action_dim), "slice": [start, start + int(t.action_dim)]}
        start += int(t.action_dim)
        if hasattr(t, "_joint_names"):
            info["joint_names"] = list(t._joint_names)
        if hasattr(t, "_scale"):
            info["scale"] = to_list(t._scale) if isinstance(t._scale, torch.Tensor) else t._scale
        if hasattr(t, "_offset"):
            info["offset"] = to_list(t._offset)  # = default joint pos when use_default_offset=True
        if hasattr(t, "_open_command"):
            info["open_command"] = to_list(t._open_command)
            info["close_command"] = to_list(t._close_command)
            info["binary_rule"] = "action < 0 -> close, else open"
        aterms.append(info)
    out["action_terms"] = aterms
    out["action_dim"] = start

    # target pose command that is part of the observation
    try:
        r = cfg.commands.object_pose.ranges
        mid = lambda t: 0.5 * (t[0] + t[1])  # noqa: E731
        q = quat_from_euler_xyz(torch.tensor([mid(r.roll)]), torch.tensor([mid(r.pitch)]), torch.tensor([mid(r.yaw)]))
        out["default_target_command"] = [mid(r.pos_x), mid(r.pos_y), mid(r.pos_z)] + [round(v, 5) for v in q[0].tolist()]
        out["target_command_ranges"] = {"pos_x": list(r.pos_x), "pos_y": list(r.pos_y), "pos_z": list(r.pos_z)}
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] could not read command ranges: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)
    print(yaml.safe_dump(out, sort_keys=False))
    print(f"[DONE] wrote {os.path.abspath(args.out)}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

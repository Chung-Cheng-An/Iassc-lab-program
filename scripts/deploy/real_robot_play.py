#!/usr/bin/env python3
"""ROS 2 node: run the exported Isaac Lab policy (REACH / GRASP) on the real OpenArm.

Pipeline (same as scripts/sim2real_vision_rl_play.py):
    camera image -> YOLO cup box -> table-plane ray intersection -> cup position in base frame
    + /joint_states -> 36-d observation (layout from robot_layout.yaml) -> policy.pt -> action
    -> arm: default + scale * action ; gripper: binary (action < 0 closes)

Needs on the robot PC: ROS 2, torch, numpy, pyyaml, ultralytics (no Isaac Sim).

    # 1) dry run (default): nothing is sent to the robot
    python3 real_robot_play.py --mode reach --policy reach/policy.pt --layout robot_layout.yaml --params real_robot_params.yaml
    # 2) real motion (after verifying the printed numbers!)
    python3 real_robot_play.py --mode reach ... --execute
"""

import argparse
import time

import numpy as np
import rclpy
import torch
import yaml
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def image_to_rgb(msg: Image) -> np.ndarray:
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.step // 3 if msg.step else msg.width, 3)[:, : msg.width]
        return img if msg.encoding == "rgb8" else img[..., ::-1].copy()
    if msg.encoding == "mono8":
        g = buf.reshape(msg.height, msg.step)[:, : msg.width]
        return np.repeat(g[..., None], 3, axis=2)
    raise ValueError(f"unsupported image encoding {msg.encoding}")


class Deployer(Node):
    def __init__(self, a):
        super().__init__("openarm_policy_deploy")
        self.a = a
        self.layout = yaml.safe_load(open(a.layout, encoding="utf-8"))
        self.P = yaml.safe_load(open(a.params, encoding="utf-8"))
        self.S, self.V = self.P["safety"], self.P["vision"]
        self.execute = a.execute
        self.policy = torch.jit.load(a.policy, map_location="cpu").eval()
        self.hz = float(self.layout["control_hz"])
        self.max_steps = self.S["max_steps_reach" if a.mode == "reach" else "max_steps_grasp"]

        # --- layout ---
        L = self.layout
        self.jn = L["robot_joint_names"]
        self.jdef = dict(zip(self.jn, L["robot_default_joint_pos"]))
        self.obs_terms = L["obs_terms"]
        self.obs_dim = L["obs_dim"]
        arm = next(t for t in L["action_terms"] if "joint_names" in t and "open_command" not in t)
        grip = next(t for t in L["action_terms"] if "open_command" in t)
        self.arm_names = arm["joint_names"]
        self.arm_scale = np.array(arm["scale"], dtype=np.float32) * np.ones(len(self.arm_names), dtype=np.float32)
        self.arm_default = np.array(arm["offset"], dtype=np.float32)
        self.arm_slice = slice(*arm["slice"])
        self.grip_names = grip["joint_names"]
        self.grip_open = np.array(grip["open_command"], dtype=np.float32)
        self.grip_close = np.array(grip["close_command"], dtype=np.float32)
        self.grip_idx = grip["slice"][0]
        self.act_dim = L["action_dim"]
        self.target_cmd = np.array(L.get("default_target_command", [0.3, 0, 0.475, 1, 0, 0, 0]), dtype=np.float32)
        self.name_map = self.P.get("joint_name_map") or {}

        # --- camera ---
        c = self.P["camera"]
        self.K_inv = np.linalg.inv(np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], dtype=np.float64))
        T = np.array(c["T_base_cam"], dtype=np.float64)
        self.R, self.t = T[:3, :3], T[:3, 3]

        from ultralytics import YOLO

        self.yolo = YOLO(self.V["yolo_weights"])

        # --- state ---
        self.q, self.dq = {}, {}
        self.cup = None
        self.cup_stamp = 0.0
        self.last_action = np.zeros(self.act_dim, dtype=np.float32)
        self.cmd_prev = None
        self.step = 0
        self.warned = set()
        self.started = False

        T_ = self.P["topics"]
        self.create_subscription(JointState, T_["joint_states"], self.on_joints, 20)
        self.create_subscription(Image, T_["image"], self.on_image, 1)
        self.pub_arm = self.create_publisher(JointTrajectory, T_["arm_cmd"], 10)
        self.pub_grip = self.create_publisher(JointTrajectory, T_["gripper_cmd"], 10)
        self.create_timer(1.0 / self.hz, self.on_timer)
        self.get_logger().info(
            f"mode={a.mode} execute={self.execute} hz={self.hz} obs_dim={self.obs_dim} act_dim={self.act_dim}"
        )

    # ------------------------------------------------------------------ inputs
    def on_joints(self, msg: JointState):
        for i, n in enumerate(msg.name):
            self.q[n] = msg.position[i] if i < len(msg.position) else 0.0
            self.dq[n] = msg.velocity[i] if i < len(msg.velocity) else 0.0

    def on_image(self, msg: Image):
        try:
            rgb = image_to_rgb(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        res = self.yolo(rgb, conf=self.V["conf"], verbose=False)[0]
        best = None
        for b in res.boxes:
            if self.yolo.names[int(b.cls)] == self.V["yolo_class"] and (best is None or b.conf > best.conf):
                best = b
        if best is None:
            return
        x1, y1, x2, y2 = best.xyxy[0].tolist()
        p = self.pixel_to_base(0.5 * (x1 + x2), y2)  # bottom-centre of the box touches the table
        if p is None:
            return
        self.cup = p if self.cup is None else (self.V["ema"] * p + (1 - self.V["ema"]) * self.cup)
        self.cup_stamp = time.time()

    def pixel_to_base(self, u, v):
        d = self.R @ (self.K_inv @ np.array([u, v, 1.0]))
        if abs(d[2]) < 1e-6:
            return None
        s = (self.V["cup_z_base"] - self.t[2]) / d[2]
        if s <= 0:
            return None
        p = self.t + s * d
        horiz = d[:2] / (np.linalg.norm(d[:2]) + 1e-9)
        p[:2] += horiz * self.V["cup_radius"]  # contact point -> cup centre
        p[2] = self.V["cup_z_base"]
        return p.astype(np.float32)

    # ------------------------------------------------------------------ obs
    def real(self, sim_name):
        return self.name_map.get(sim_name, sim_name)

    def build_obs(self):
        parts = []
        for t in self.obs_terms:
            n = t["name"]
            jn = t.get("joint_names", self.jn)
            if n in ("joint_pos", "joint_pos_rel"):
                parts.append([self.q.get(self.real(j), self.jdef[j]) - self.jdef[j] for j in jn])
            elif n in ("joint_vel", "joint_vel_rel"):
                parts.append([self.dq.get(self.real(j), 0.0) for j in jn])
            elif n == "object_position":
                parts.append(self.cup)
            elif n == "target_object_position":
                parts.append(self.target_cmd)
            elif n in ("actions", "last_action"):
                parts.append(self.last_action)
            else:
                raise RuntimeError(f"unknown obs term '{n}' - add it to build_obs()")
            if len(parts[-1]) != t["dim"]:
                raise RuntimeError(f"obs term {n}: got {len(parts[-1])}, expected {t['dim']}")
        return torch.tensor(np.concatenate(parts), dtype=torch.float32).unsqueeze(0)

    # ------------------------------------------------------------------ loop
    def on_timer(self):
        missing = [self.real(j) for j in self.arm_names if self.real(j) not in self.q]
        if missing:
            self.get_logger().warn(f"waiting for joint_states of {missing}", throttle_duration_sec=2.0)
            return
        q_arm = np.array([self.q[self.real(j)] for j in self.arm_names], dtype=np.float32)

        if not self.started:
            err = float(np.abs(q_arm - self.arm_default).max())
            if err > self.S["start_tolerance"]:
                self.get_logger().error(
                    f"arm is {err:.2f} rad from the sim start pose {np.round(self.arm_default, 2).tolist()} -> "
                    "move the arm to that pose first.",
                    throttle_duration_sec=3.0,
                )
                return
            self.cmd_prev = q_arm.copy()
            self.started = True

        if self.cup is None or time.time() - self.cup_stamp > self.V["max_age_s"]:
            self.get_logger().warn("no fresh cup detection -> holding position", throttle_duration_sec=2.0)
            self.send(self.cmd_prev, None)
            return
        if self.step >= self.max_steps:
            if "done" not in self.warned:
                self.warned.add("done")
                self.get_logger().info("episode finished -> holding. Ctrl+C to exit.")
            self.send(self.cmd_prev, self.last_grip)
            return

        obs = self.build_obs()
        with torch.inference_mode():
            act = self.policy(obs)[0].numpy().astype(np.float32)
        if self.a.mode == "reach":
            act[self.grip_idx] = 1.0  # gripper forced open, like the sim demo
        self.last_action = act

        target = self.arm_default + self.arm_scale * act[self.arm_slice]
        target = self.cmd_prev + self.S["lowpass"] * (target - self.cmd_prev)
        target = self.cmd_prev + np.clip(target - self.cmd_prev, -self.S["max_joint_step"], self.S["max_joint_step"])
        self.cmd_prev = target
        self.last_grip = self.grip_close if act[self.grip_idx] < 0 else self.grip_open

        if self.step % 10 == 0:
            self.get_logger().info(
                f"[{self.step:3d}] cup(base)={np.round(self.cup, 3).tolist()} "
                f"arm_target={np.round(target, 2).tolist()} gripper={'CLOSE' if act[self.grip_idx] < 0 else 'OPEN'}"
            )
        self.send(target, self.last_grip)
        self.step += 1

    last_grip = None

    def send(self, arm_target, grip_target):
        if not self.execute:
            return
        dt = Duration(sec=0, nanosec=int(1e9 / self.hz * 2))
        m = JointTrajectory()
        m.joint_names = [self.real(j) for j in self.arm_names]
        pt = JointTrajectoryPoint(positions=[float(x) for x in arm_target], time_from_start=dt)
        m.points = [pt]
        self.pub_arm.publish(m)
        if grip_target is not None:
            g = JointTrajectory()
            g.joint_names = [self.real(j) for j in self.grip_names]
            g.points = [JointTrajectoryPoint(positions=[float(x) for x in grip_target], time_from_start=dt)]
            self.pub_grip.publish(g)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["reach", "grasp"], required=True)
    p.add_argument("--policy", required=True, help="exported policy.pt of the chosen mode")
    p.add_argument("--layout", default="robot_layout.yaml")
    p.add_argument("--params", default="real_robot_params.yaml")
    p.add_argument("--execute", action="store_true", help="actually command the robot (default: dry run)")
    a = p.parse_args()
    rclpy.init()
    node = Deployer(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

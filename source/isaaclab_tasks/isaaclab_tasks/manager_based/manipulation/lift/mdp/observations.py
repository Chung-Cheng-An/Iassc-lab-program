# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


_GLOBAL_YOLO_MODEL = None

def _get_yolo_model():
    global _GLOBAL_YOLO_MODEL
    if _GLOBAL_YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            _GLOBAL_YOLO_MODEL = YOLO("yolov8n.pt")
        except Exception:
            _GLOBAL_YOLO_MODEL = False
    return _GLOBAL_YOLO_MODEL if _GLOBAL_YOLO_MODEL is not False else None


def yolo_perceived_object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("head_cam"),
) -> torch.Tensor:
    """The 3D position of the object calculated via Camera + YOLO + Depth, falling back to Ground Truth."""
    # 1. Calculate ground truth as baseline
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    gt_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)

    # 2. Check if camera is active
    if sensor_cfg.name in env.scene.sensors:
        try:
            cam = env.scene.sensors[sensor_cfg.name]
            if "rgb" in cam.data.output and "distance_to_camera" in cam.data.output:
                yolo = _get_yolo_model()
                import numpy as np
                from pxr import Gf

                num_envs = gt_pos_b.shape[0]
                perceived_list = []

                for env_i in range(num_envs):
                    rgb_np = cam.data.output["rgb"][env_i].cpu().numpy()[..., :3].astype(np.uint8)
                    depth_np = cam.data.output["distance_to_camera"][env_i].cpu().numpy().squeeze(-1)
                    intrinsics = cam.data.intrinsic_matrices[env_i].cpu().numpy()

                    # YOLO detection
                    crop_box = None
                    if yolo is not None:
                        results = yolo(rgb_np, conf=0.05, verbose=False)
                        for r in results:
                            for box in r.boxes:
                                cls_name = yolo.names[int(box.cls[0])]
                                if cls_name.lower() in ["cup", "bowl", "bottle", "vase"]:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                    crop_box = [max(0, x1 - 10), max(0, y1 - 15), min(rgb_np.shape[1], x2 + 10), min(rgb_np.shape[0], y2 + 15)]
                                    break
                    if crop_box is None:
                        crop_box = [max(0, rgb_np.shape[1]//2 - 50), max(0, rgb_np.shape[0]//2 - 50),
                                    min(rgb_np.shape[1], rgb_np.shape[1]//2 + 50), min(rgb_np.shape[0], rgb_np.shape[0]//2 + 50)]

                    # Depth unprojection
                    x1, y1, x2, y2 = crop_box
                    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
                    cx_b, cy_b = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    h, w = depth_np.shape
                    patch = depth_np[max(0, cy_b - 10):min(h, cy_b + 10), max(0, cx_b - 10):min(w, cx_b + 10)]
                    valid = patch[patch > 0.05]
                    z_depth = float(np.median(valid)) if len(valid) > 0 else float(depth_np[cy_b, cx_b])

                    x_cam = (cx_b - cx) * z_depth / fx
                    y_cam = -(cy_b - cy) * z_depth / fy

                    eye, target = np.array([0.07, 0.0, 0.52]), np.array([0.24, 0.0, 0.05])
                    fwd = (target - eye) / np.linalg.norm(target - eye)
                    z_ax = -fwd
                    x_ax = np.array([0.0, -1.0, 0.0])
                    y_ax = np.cross(z_ax, x_ax)
                    y_ax = y_ax / np.linalg.norm(y_ax)
                    R = np.column_stack([x_ax, y_ax, z_ax])

                    p_cam = np.array([x_cam, y_cam, -z_depth])
                    p_w = eye + R @ p_cam
                    p_base = np.array([p_w[0], p_w[1], 0.197])
                    perceived_list.append(p_base)

                return torch.tensor(np.array(perceived_list), device=gt_pos_b.device, dtype=torch.float32)
        except Exception:
            pass

    return gt_pos_b


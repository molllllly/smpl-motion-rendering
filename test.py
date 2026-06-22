#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final bbox-based renderer (no `camera` field):
- 使用 4D-Humans / 3DPW 的 SMPL 参数 + bounding box
- 完全不用 pkl 里的 `camera`
- 近大远小，比例在不同视频之间保持一致
- 支持多人 + overlay
"""

import os
import numpy as np
import cv2
import imageio.v2 as imageio
import torch
import pyrender
import trimesh
from smplx import SMPL

# ==============================
# CONFIG（按需改这两个）
# ==============================
PKL_PATH = "downtown_rampAndStairs_00.pkl"   # 你的 pkl
OUT_DIR  = "render_bbox_only_fixed"          # 输出目录

SMPL_MODEL_DIR = "rendering/data/"           # SMPL 模型目录
GENDER = "NEUTRAL"
FPS = 30

# 人体真实高度假设（米）
TARGET_BODY_HEIGHT_M = 1.70

# 颜色表（RGBA）
COLORS = [
    (72, 133, 237, 255),   # blue
    (230, 76, 60, 255),    # red
    (255, 198, 0, 255),    # yellow
    (24, 153, 100, 255),   # green
    (150, 0, 150, 255),    # purple
]

# ==============================
# 工具函数
# ==============================
def to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

def matrix_to_axis_angle(mat):
    """3x3 旋转矩阵 -> 轴角 (batch)."""
    cos = (mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2] - 1) / 2
    cos = torch.clamp(cos, -1.0, 1.0)
    theta = torch.acos(cos)

    axis = torch.stack([
        mat[:, 2, 1] - mat[:, 1, 2],
        mat[:, 0, 2] - mat[:, 2, 0],
        mat[:, 1, 0] - mat[:, 0, 1]
    ], dim=1)

    axis = torch.nn.functional.normalize(axis, dim=1)
    aa = axis * theta.unsqueeze(1)
    aa[torch.isnan(aa)] = 0.0
    return aa

def smpl_to_mesh(smpl_model, smpl_dict, color_rgba):
    """用你原来的方式，从 SMPL 参数生成 mesh。"""
    betas = to_tensor(smpl_dict.get("betas", np.zeros(10, np.float32))).unsqueeze(0)

    body_pose_mats = to_tensor(smpl_dict["body_pose"]).reshape(-1, 3, 3)
    global_orient_mats = to_tensor(smpl_dict["global_orient"]).reshape(-1, 3, 3)

    body_aa = matrix_to_axis_angle(body_pose_mats).reshape(1, -1)
    global_aa = matrix_to_axis_angle(global_orient_mats).reshape(1, -1)

    out = smpl_model(
        betas=betas,
        body_pose=body_aa,
        global_orient=global_aa
    )
    verts = out.vertices[0].detach().cpu().numpy()

    # 坐标系修正（保持你以前的做法）
    R_fix = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]], dtype=np.float32)
    verts = (R_fix @ verts.T).T

    tri = trimesh.Trimesh(vertices=verts, faces=smpl_model.faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=np.array(color_rgba, dtype=np.float32) / 255.0,
        metallicFactor=0.0,
        roughnessFactor=0.9,
        doubleSided=True
    )
    return pyrender.Mesh.from_trimesh(tri, material=material, smooth=False)

# ==============================
# 从 frame 里取多人数据
# ==============================
def pick_people(frame):
    """
    返回 [(pid, smpl_dict, bbox)], 并且保证：
        smpl[i] 对应 tid[i]
        bbox 对应 tid[i]
    """
    tids = frame.get("tracked_ids") or frame.get("tid")
    if tids is None:
        return []

    tids = [int(i) for i in tids]

    smpl_list = frame.get("smpl")
    bbox_list = frame.get("tracked_bbox") or frame.get("bbox")

    if smpl_list is None or bbox_list is None:
        return []

    smpl_list = list(smpl_list)
    bbox_list = list(bbox_list)

    n = min(len(tids), len(smpl_list), len(bbox_list))

    people = []
    for i in range(n):
        pid = tids[i]
        smpl_dict = smpl_list[i]
        bbox = np.asarray(bbox_list[i], dtype=float).ravel()
        if bbox.shape[0] < 4:
            continue
        people.append((pid, smpl_dict, bbox))

    return people

# ==============================
#   估计内参 + 深度
# ==============================
def estimate_intrinsics(W, H):
    """
    给一个“比较合理”的虚拟内参。
    不依赖真实相机，也不会拉伸：
      - fx, fy ~ 1.2 * max(W, H)
    """
    f = 1.2 * max(W, H)
    fx = fy = f
    cx = W / 2.0
    cy = H / 2.0
    return fx, fy, cx, cy

def depth_from_bbox_height(bbox_h, fx, body_height_m=TARGET_BODY_HEIGHT_M):
    """
    近似：Z = fx * H_person / h_pixels
    返回负值（pyrender 相机朝 -Z 看）。
    """
    if bbox_h < 10:   # 太小的 bbox 就给一个默认深度
        return -5.0
    Z = fx * body_height_m / bbox_h
    return -float(Z)

def translation_from_bbox(bbox, fx, fy, cx, cy):
    """
    由 bbox 估计 3D 平移：
      X = (u - cx) * Z / fx
      Y = -(v - cy) * Z / fy
    这里为了修复“左右反了”，我们在 X 前面加一个负号。
    """
    x, y, w, h = bbox
    u = x + w / 2.0
    v = y + h / 2.0

    Z = depth_from_bbox_height(h, fx)

    # ★★ 关键改动：这里多了一个负号，修复左右镜像 ★★
    X = -(u - cx) * (Z / fx)
    Y = -(v - cy) * (Z / fy)

    return np.array([X, Y, Z], dtype=np.float32)

# ==============================
# 主渲染逻辑
# ==============================
def apply_render(data, out_dir, fps=30):
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_keys = sorted(list(data.keys()))

    # --- 确定分辨率（用第一帧真实图像） ---
    first_frame = data[frame_keys[0]]
    first_bg_path = first_frame.get("frame_path")
    if isinstance(first_bg_path, (list, tuple)):
        first_bg_path = first_bg_path[0]

    bg_img = cv2.imread(first_bg_path)
    if bg_img is None:
        H, W = 720, 1280
    else:
        H, W = bg_img.shape[:2]

    fx, fy, cx, cy = estimate_intrinsics(W, H)
    print(f"[INFO] Resolution: {W} x {H}")
    print(f"[INFO] Intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # --- SMPL & renderer ---
    smpl_model = SMPL(model_path=SMPL_MODEL_DIR, gender=GENDER)
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.1, zfar=1000.0)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    fg_paths = []
    bg_paths = []

    # 1) 渲染前景 RGBA 序列
    for fi, key in enumerate(frame_keys):
        frame = data[key]
        bg_path = frame.get("frame_path")
        if isinstance(bg_path, (list, tuple)):
            bg_path = bg_path[0]
        bg_paths.append(bg_path)

        people = pick_people(frame)

        if len(people) == 0:
            rgba = np.zeros((H, W, 4), dtype=np.uint8)
            out_path = os.path.join(frames_dir, f"{fi:06d}.png")
            imageio.imwrite(out_path, rgba)
            fg_paths.append(out_path)
            continue

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
        scene.add(camera, pose=np.eye(4, dtype=np.float32))
        scene.add(light, pose=np.eye(4, dtype=np.float32))

        for pi, (pid, smpl_dict, bbox) in enumerate(people):
            color = COLORS[pi % len(COLORS)]

            T = translation_from_bbox(bbox, fx, fy, cx, cy)

            mesh = smpl_to_mesh(smpl_model, smpl_dict, color)
            pose = np.eye(4, dtype=np.float32)
            pose[:3, 3] = T
            scene.add(mesh, pose=pose)

        rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        out_path = os.path.join(frames_dir, f"{fi:06d}.png")
        imageio.imwrite(out_path, rgba)
        fg_paths.append(out_path)

        if (fi + 1) % 50 == 0 or fi == len(frame_keys) - 1:
            print(f"[{fi+1:04d}/{len(frame_keys)}] frames rendered")

    renderer.delete()

    # 2) 导出 SMPL-only 视频
    clean_video_path = os.path.join(out_dir, "smpl_clean.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_clean = cv2.VideoWriter(clean_video_path, fourcc, fps, (W, H))

    for fg_path in fg_paths:
        fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
        if fg is None:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            frame = fg[:, :, :3]
        writer_clean.write(frame)
    writer_clean.release()
    print("[DONE] SMPL-only video:", clean_video_path)

    # 3) 合成 overlay 视频
    overlay_video_path = os.path.join(out_dir, "smpl_overlay.mp4")
    writer_ov = cv2.VideoWriter(overlay_video_path, fourcc, fps, (W, H))

    for fg_path, bg_path in zip(fg_paths, bg_paths):
        fg = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
        if fg is None or fg.shape[2] != 4:
            alpha = np.zeros((H, W, 1), dtype=np.float32)
            fg_rgb = np.zeros((H, W, 3), dtype=np.float32)
        else:
            alpha = fg[:, :, 3:4].astype(np.float32) / 255.0
            fg_rgb = fg[:, :, :3].astype(np.float32)

        bg = cv2.imread(bg_path) if bg_path is not None else None
        if bg is None:
            bg_rgb = np.zeros((H, W, 3), dtype=np.float32)
        else:
            if bg.shape[:2] != (H, W):
                bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)
            bg_rgb = bg.astype(np.float32)

        comp = fg_rgb * alpha + bg_rgb * (1.0 - alpha)
        comp = np.clip(comp, 0, 255).astype(np.uint8)
        writer_ov.write(comp)

    writer_ov.release()
    print("[DONE] Overlay video:", overlay_video_path)

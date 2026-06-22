#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终版本：使用 3DPW 标准内参 (IntrinsicsCamera) 和修正后的米制 T 向量进行 Mesh 定位。
解决了 Z 轴缩放因子 22.5 导致的近景人物比例错误问题。
"""
from pycocotools import mask as maskUtils

import os, math
import numpy as np
import torch
import trimesh
import pyrender
import imageio.v2 as imageio
import cv2
from smplx import SMPL
import joblib  # 用于加载 PKL 文件

# ====== Knobs / 配置 ======
# 注意：你需要将 PKL_PATH 设置为你的 3DPW 序列 PKL 文件路径
PKL_PATH = "demo_surv_cut.pkl"
OUT_DIR = "render_smpl_3dpw_out"
BG_ROOT_DIR = None  # 如果你的 PKL 中的 frame_path 是绝对路径，可以设置为 None

GENDER = "NEUTRAL"
FPS = 30
OUT_SIZE = None  # None 表示使用背景图的原始分辨率
SMPL_MODEL_DIR = "rendering/data/"  # Directory containing SMPL_* files

# --- 3DPW 标准相机参数（基于 1920x1080 分辨率）---
# 在缺乏 PKL 中 cam_intrinsics 准确值的情况下，使用 3DPW 的常用平均参数
FX_3DPW_NATIVE = 1962.2
FY_3DPW_NATIVE = 1962.2
NATIVE_W_3DPW = 1920
NATIVE_H_3DPW = 1080
# 主点 CX/CY 会根据最终分辨率 W_out/H_out 重新计算为中心点

CONFIDENCE_THRESHOLD = 0.3

COLORS = [
    (72, 133, 237, 255),  # Blue
    (230, 76, 60, 255),  # Red
    (255, 198, 0, 255),  # Yellow
    (24, 153, 100, 255),  # Green
    (150, 0, 150, 255)  # Purple
]


# ====== Utilities / 工具函数  ======

def to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)


def matrix_to_axis_angle(mat):
    cos_theta = (mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2] - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    axis = torch.stack([
        mat[:, 2, 1] - mat[:, 1, 2],
        mat[:, 0, 2] - mat[:, 2, 0],
        mat[:, 1, 0] - mat[:, 0, 1]
    ], dim=1)
    axis = torch.nn.functional.normalize(axis, dim=1)
    aa = axis * theta.unsqueeze(1)
    aa[torch.isnan(aa)] = 0.0
    return aa


# smpl_to_mesh 函数现在只返回 Mesh，不计算 Y 偏移
def smpl_to_mesh(smpl_model, smpl_dict, color_rgba=(200, 200, 200, 255)):
    """Build a pyrender mesh from SMPL parameters."""
    betas = to_tensor(smpl_dict.get("betas", np.zeros(10, np.float32))).unsqueeze(0)
    body_pose_mats = to_tensor(smpl_dict["body_pose"]).reshape(-1, 3, 3)
    body_aa = matrix_to_axis_angle(body_pose_mats).reshape(1, -1)
    g = to_tensor(smpl_dict["global_orient"])
    g_aa = matrix_to_axis_angle(g.reshape(-1, 3, 3)).reshape(1, -1)
    out = smpl_model(betas=betas, body_pose=body_aa, global_orient=g_aa)
    verts = out.vertices[0].detach().cpu().numpy()

    # 核心修正：旋转到 Pyrender 坐标系 (X, -Y, -Z)
    # Pyrender 使用 Y-up, Z-in (看向-Z)。SMPL 通常是 Z-up或Y-up但面向-Y。
    # trimesh 的旋转矩阵可以保证正确的转换
    R_fix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    verts = (R_fix @ verts.T).T

    faces = smpl_model.faces
    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    # 如果要使用 pyrender.Mesh.from_trimesh，material 优先
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=np.array(color_rgba, dtype=np.float32) / 255.0,
        metallicFactor=0.0, roughnessFactor=0.9, doubleSided=True
    )
    return pyrender.Mesh.from_trimesh(tri, material=material, smooth=False)


def _to_xywh(b):
    b = np.asarray(b, dtype=float).ravel()
    x, y, w, h = b[0], b[1], b[2], b[3]
    return (x, y, w, h)


def get_frame_image_size(frame):
    DEFAULT_WH = (1280, 720)
    s = frame.get("size", None)
    if not s:
        return DEFAULT_WH
    else:
        # frame['size'] 可能是 [[W, H]] 的格式
        first = s[0]
        W, H = int(first[0]), int(first[1])
        return (W, H)


# 确保 frame_path 存在并返回 W, H
def determine_output_resolution(data, out_size):
    frame_keys = list(data.keys())
    W_out, H_out = 1280, 720  # 默认回退值

    if out_size and len(out_size) == 2:
        W_out, H_out = int(out_size[0]), int(out_size[1])
        return W_out, H_out

    # 尝试从第一帧的 frame_path 读取实际图像尺寸
    for k in frame_keys[:50]:
        frame = data[k]
        img_path = frame.get("frame_path")
        if isinstance(img_path, (list, tuple)): img_path = img_path[0]

        if img_path:
            bg_img = cv2.imread(img_path)
            if bg_img is not None and bg_img.ndim == 3:
                H_out, W_out = bg_img.shape[:2]
                return W_out, H_out

    # 否则使用 get_frame_image_size 尝试从元数据中获取
    for k in frame_keys[:50]:
        sz = get_frame_image_size(data[k])
        if sz is not None:
            W_out, H_out = sz
            return W_out, H_out

    return W_out, H_out  # 使用默认回退值


def pick_two_people(frame):
    """Pick up to two (bbox, smpl, id) tuples from the frame dict."""
    ids = frame.get("tracked_ids") or frame.get("tid")
    if ids is None: return []
    ids = [int(i) for i in ids]
    tb = frame.get("tracked_bbox", None)
    fb = frame.get("bbox", None)
    smpl_obj = frame.get("smpl", None)
    tids = frame.get("tid", None)

    # 假设 'camera' 数据列表与 smpl/tid 列表顺序一致
    camera_data = frame.get("camera", [])

    if smpl_obj is None or tids is None: return []

    tracked_bboxes = list(tb) if tb is not None else None
    fallback_bboxes = list(fb) if fb is not None else None
    smpl_list = list(smpl_obj)
    tids = [int(t) for t in tids]
    pid2sidx = {}

    # 构建 Person ID 到 SMPL/Camera 列表索引的映射
    if len(tids) == len(smpl_list) and len(tids) > 0:
        for si, pid in enumerate(tids): pid2sidx[int(pid)] = si
    elif len(smpl_list) == len(ids) and len(ids) > 0:
        for j, pid in enumerate(ids): pid2sidx[int(pid)] = j
    elif len(smpl_list) == 1 and len(ids) >= 1:
        pid2sidx[int(ids[0])] = 0

    pairs = []
    for j, pid in enumerate(ids):
        sidx = pid2sidx.get(int(pid), None)
        # 确保索引 sidx 在 smpl_list 和 camera_data 范围内
        if sidx is None or sidx < 0 or sidx >= len(smpl_list) or sidx >= len(camera_data):
            continue

        smpl_dict = smpl_list[sidx]
        T_pkl = camera_data[sidx]  # 获取对应的 T 向量

        bb = None
        if (tracked_bboxes is not None) and (j < len(tracked_bboxes)) and (tracked_bboxes[j] is not None):
            bb = _to_xywh(tracked_bboxes[j])
        elif (fallback_bboxes is not None) and (j < len(fallback_bboxes)) and (fallback_bboxes[j] is not None):
            bb = _to_xywh(fallback_bboxes[j])

        if bb is None or T_pkl is None or T_pkl.shape != (3,):
            continue

        pairs.append((bb, smpl_dict, int(pid), T_pkl))
        # if len(pairs) >= 2: break # 移除限制，渲染所有检测到的人
    return pairs


# ====== apply_render 函数 (核心修改) ======

def apply_render(data, output_dir, out_size=None, fps=30):
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    frame_keys = sorted(list(data.keys()))

    # 1. 确定输出分辨率
    W_out, H_out = determine_output_resolution(data, out_size)

    # 2. **核心修正：使用 IntrinsicsCamera 参数**

    # 使用 3DPW 的标准参数进行缩放
    FX_NATIVE = FX_3DPW_NATIVE
    FY_NATIVE = FY_3DPW_NATIVE
    NATIVE_W = NATIVE_W_3DPW
    NATIVE_H = NATIVE_H_3DPW

    # 按比例缩放内参
    scale_factor_w = W_out / NATIVE_W
    scale_factor_h = H_out / NATIVE_H

    # 注意：这里假设焦距的缩放与 W_out/H_out 的比例一致
    f_x_px = FX_NATIVE * scale_factor_w
    f_y_px = FY_NATIVE * scale_factor_h

    # 主点 (cx, cy) 设为输出图像的中心
    c_x_px = W_out / 2.0
    c_y_px = H_out / 2.0

    print(f"[INFO] Output size = {W_out}x{H_out}")
    print(f"[INFO] Using Intrinsics: fx={f_x_px:.2f}, fy={f_y_px:.2f}, cx={c_x_px:.2f}, cy={c_y_px:.2f}")

    # -------- SMPL 渲染准备 --------
    smpl_model = SMPL(model_path=SMPL_MODEL_DIR, gender=GENDER)
    renderer = pyrender.OffscreenRenderer(viewport_width=W_out, viewport_height=H_out)

    # **替换：使用 IntrinsicsCamera**
    camera = pyrender.IntrinsicsCamera(
        fx=f_x_px,
        fy=f_y_px,
        cx=c_x_px,
        cy=c_y_px,
        zfar=1e12  # 设置一个远裁剪平面
    )
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    fg_paths = []
    bg_paths = []

    # -------- 1. 渲染黑底 SMPL 前景 --------
    for fi, fkey in enumerate(frame_keys):
        frame = data[fkey]
        # pick_two_people 现在返回 (bbox, smpl_dict, pid, T_pkl)
        pairs = pick_two_people(frame)

        img_path = frame.get("frame_path", None)
        if isinstance(img_path, (list, tuple)): img_path = img_path[0]
        bg_paths.append(img_path)

        fg_path = os.path.join(frames_dir, f"{fi:06d}.png")

        if not pairs:
            color_rgba = np.zeros((H_out, W_out, 4), dtype=np.uint8)
            imageio.imwrite(fg_path, color_rgba)
            fg_paths.append(fg_path)
            continue

        # 设置场景为透明背景
        scene = pyrender.Scene(bg_color=np.array([0., 0., 0., 0.], dtype=np.float32))

        # 将相机添加到原点 pose=np.eye(4)
        scene.add(camera, pose=np.eye(4, dtype=np.float32))
        scene.add(light, pose=np.eye(4, dtype=np.float32))

        for k, (bb, smpl_dict, pid, T_pkl) in enumerate(pairs):

            X_world, Y_world, Z_world = T_pkl

            # **核心修正：移除 22.5 缩放**
            # T_pkl (X, Y, Z) 是米制世界坐标

            # T_pyrender = (X, -Y, Z) 转换到 Pyrender 相机空间
            # Z_world 是负值 (e.g., -5.0)，保留其负值作为 Pyrender 深度 (Z < 0)
            Z_pyrender = Z_world

            T_pyrender = np.array([X_world, -Y_world, Z_pyrender], dtype=np.float32)

            # 引入最小安全距离限制 (解决极近人物的裁剪/渲染错误)
            MIN_Z_DEPTH = -0.1  # 最小深度 0.1 米
            if T_pyrender[2] > MIN_Z_DEPTH:
                T_pyrender[2] = MIN_Z_DEPTH  # 限制 Z 轴分量不超过 -0.1m

            T = T_pyrender

            print(f"Frame {fi}, Person {pid}: T (Meter) used for Pyrender: {T}")

            color = COLORS[k % len(COLORS)]
            mesh = smpl_to_mesh(smpl_model, smpl_dict, color_rgba=color)

            M = np.eye(4, dtype=np.float32);
            M[:3, 3] = T  # 直接使用修正后的米制 T 向量

            # 可选：引入 Y 轴微调解决浮空问题（如果需要）
            # M[1, 3] += 0.15

            scene.add(mesh, pose=M)

        color_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        imageio.imwrite(fg_path, color_rgba)
        fg_paths.append(fg_path)

        if (fi + 1) % 50 == 0 or fi == len(frame_keys) - 1:
            print(f"[{fi + 1:03d}/{len(frame_keys)}] saved fg {fg_path}")

    renderer.delete()

    # -------- 2. 输出黑底 SMPL 视频 --------
    video_path_clean = os.path.join(output_dir, "demo_smpl_clean.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_clean = cv2.VideoWriter(video_path_clean, fourcc, fps, (W_out, H_out))

    for p in fg_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None: continue
        if (img.shape[1], img.shape[0]) != (W_out, H_out):
            img = cv2.resize(img, (W_out, H_out), interpolation=cv2.INTER_AREA)
        writer_clean.write(img)
    writer_clean.release()
    print(f"[DONE] Clean SMPL video saved to: {video_path_clean}")

    # -------- 3. Overlay 合成到真实背景 --------
    video_path_overlay = os.path.join(output_dir, "overlay_mask.mp4")
    writer_ov = cv2.VideoWriter(video_path_overlay, fourcc, fps, (W_out, H_out))

    for i, (fg_path, fkey, bg_path) in enumerate(zip(fg_paths, frame_keys, bg_paths)):
        fg_rgba = cv2.imread(fg_path, cv2.IMREAD_UNCHANGED)
        if fg_rgba is None: continue

        bg = None
        if bg_path is not None:
            clean_bg_path = os.path.normpath(bg_path)

            if BG_ROOT_DIR is not None and not os.path.isabs(clean_bg_path):
                clean_bg_path = os.path.join(BG_ROOT_DIR, clean_bg_path)
                clean_bg_path = os.path.normpath(clean_bg_path)

            if os.path.exists(clean_bg_path):
                bg = cv2.imread(clean_bg_path, cv2.IMREAD_COLOR)

        # 回退逻辑/尺寸调整
        if bg is None or bg.size == 0 or np.mean(bg) < 1.0:
            bg = np.zeros((H_out, W_out, 3), dtype=np.uint8)
        else:
            if (bg.shape[1], bg.shape[0]) != (W_out, H_out):
                bg = cv2.resize(bg, (W_out, H_out), interpolation=cv2.INTER_AREA)

        # 合成逻辑
        alpha_render = fg_rgba[:, :, 3:4].astype(np.float32) / 255.0
        fg_rgb = fg_rgba[:, :, :3].astype(np.float32)
        bg_rgb = bg.astype(np.float32)

        comp_rgb = fg_rgb * alpha_render + bg_rgb * (1.0 - alpha_render)
        comp_rgb = np.clip(comp_rgb, 0, 255).astype(np.uint8)

        writer_ov.write(comp_rgb)

    writer_ov.release()
    print(f"[DONE] Overlay video saved to: {video_path_overlay}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(SMPL_MODEL_DIR):
        print(f"[CRITICAL SETUP] SMPL 模型目录不存在: {SMPL_MODEL_DIR}。请下载 SMPL 模型文件。")
    try:
        data = joblib.load(PKL_PATH)
        apply_render(data, OUT_DIR)
    except FileNotFoundError:
        print(f"[ERROR] 找不到 PKL 文件: {PKL_PATH}。请确保文件路径正确。")
    except Exception as e:
        print(f"[ERROR] 运行时发生错误: {e}")
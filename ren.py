#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4SMPL Reconstruction Rendering (two-person, minimal params)
-----------------------------------------------------------
- Load a joblib-serialized PKL (your filtered/interpolated tracking output).
- For each frame, render up to TWO people:
    * Prefer the first two entries in `tracked_ids` if present;
      otherwise, take the first two from the lists.
- Estimate camera-space translation T = (X, Y, Z) from bbox height and vertical FOV,
  assuming a constant human height (DEFAULT_HEIGHT_M).
- Render clean meshes on a black background to PNG frames and then encode to MP4.
"""

import os, re, math, platform
import numpy as np
import torch
import trimesh
import pyrender
import imageio.v2 as imageio
import cv2
from smplx import SMPL

# ====== Minimal knobs only ======
#PKL_PATH         = "demo_portcamera_compressed3.pkl"
PKL_PATH         = "surv_cut_SG.pkl"
OUT_DIR          = "render_4smpl_out_two_bilk"           # Output folder
SMPL_MODEL_DIR   = "data/smpl"                           # Directory containing SMPL_* files
GENDER           = "NEUTRAL"                             # "NEUTRAL" / "MALE" / "FEMALE"
FPS              = 30
OUT_SIZE         = None                                  # (W,H) or None to infer; fallback 1280x720
YFOV_DEG         = 60.0                                  # Vertical field-of-view in degrees
DEFAULT_HEIGHT_M = 1.70                                  # Assumed person height (meters)

# --- [改动1] Camera 单位与换算表（假设 camera 是厘米） ---
CAMERA_UNITS = "cm"                 # 可选 "m" / "cm" / "mm"
_UNIT_SCALE  = {"m": 1.0, "cm": 0.01, "mm": 0.001}

# Two distinct colors so two persons are easy to differentiate (RGBA)
COLORS = [
    (72, 133, 237, 255),   # Blue
    (230, 76, 60, 255),    # Red
]

# ====== Utilities ======
def to_tensor(x):
    """Ensure a float32 torch tensor (no grads)."""
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

def matrix_to_axis_angle(mat):
    """
    Convert rotation matrices to axis-angle.
    Args:
        mat: [N,3,3] torch tensor of rotation matrices.
    Returns:
        aa:  [N,3] axis-angle per rotation.
    """
    # trace(R) = 1 + 2 cos(theta)
    cos_theta = (mat[:,0,0] + mat[:,1,1] + mat[:,2,2] - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    # skew-symmetric extraction (unnormalized axis)
    axis = torch.stack([
        mat[:,2,1] - mat[:,1,2],
        mat[:,0,2] - mat[:,2,0],
        mat[:,1,0] - mat[:,0,1]
    ], dim=1)
    axis = torch.nn.functional.normalize(axis, dim=1)
    aa = axis * theta.unsqueeze(1)
    aa[torch.isnan(aa)] = 0.0
    return aa

def smpl_to_mesh(smpl_model, smpl_dict, color_rgba=(200,200,200,255)):
    """
    Build a pyrender mesh from SMPL parameters (global_orient/body_pose given as 3x3 rotmats).
    Uses a 180° rotation around X to fix upside-down orientation, which preserves face winding.
    """
    # Betas (shape); default to zeros if missing
    betas = to_tensor(smpl_dict.get("betas", np.zeros(10, np.float32))).unsqueeze(0)

    # Body pose: expect (J,3,3) rotation matrices -> convert to axis-angle (J*3)
    body_pose_mats = to_tensor(smpl_dict["body_pose"]).reshape(-1,3,3)
    body_aa = matrix_to_axis_angle(body_pose_mats).reshape(1,-1)

    # Global orient as 3x3 -> axis-angle (3)
    g = to_tensor(smpl_dict["global_orient"])
    g_aa = matrix_to_axis_angle(g.reshape(-1,3,3)).reshape(1,-1)

    # Forward SMPL to get vertices
    out = smpl_model(betas=betas, body_pose=body_aa, global_orient=g_aa)
    verts = out.vertices[0].detach().cpu().numpy()

    # Fix "upside-down" orientation using a proper rotation (det=+1).
    R_fix = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]], dtype=np.float32)  # 180° about X axis
    verts = (R_fix @ verts.T).T

    faces = smpl_model.faces

    # Construct a trimesh and then a pyrender mesh with double-sided material
    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    tri.visual.vertex_colors = np.tile(np.array(color_rgba, dtype=np.uint8), (tri.vertices.shape[0], 1))
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=np.array(color_rgba, dtype=np.float32)/255.0,
        metallicFactor=0.0, roughnessFactor=0.9, doubleSided=True
    )
    return pyrender.Mesh.from_trimesh(tri, material=material, smooth=False)

def _to_xywh(b):
    # 兼容 numpy/list/tuple；直接展平后取前4个
    b = np.asarray(b, dtype=float).ravel()
    x, y, w, h = b[0], b[1], b[2], b[3]
    return (x, y, w, h)

def get_frame_image_size(frame):
    DEFAULT_WH = (1280,720)
    """
    Read (W,H) from frame['size'] by always taking the first pair.
    Accepts:
      - [[W, H]] or [[W, H], [W, H], ...]
      - [W, H] (fallback)
    """
    s = frame.get("size",None)
    if not s:
        W, H = DEFAULT_WH
        return (W, H)
    else:
        first = s[0]
        W, H = int(first[0]), int(first[1])
        return (W, H)

def uvh_to_camera_T(cx, v_foot, h_px, W_img, H_img, yfov_rad, Hm):
    """
    Estimate camera-space translation T = (X,Y,Z) from bbox:
      - Z from person height (Hm) and vertical FOV (yfov_rad).
      - (X,Y) from pinhole projection with focal f derived from yfov and image height.
    Conventions:
      - Camera looks down -Z
      - We return (X, -Y, Z) to get +Y pointing upwards in render space.
    """
    f_px = H_img / (2 * math.tan(yfov_rad / 2.0))
    Z = -(Hm * f_px) / max(h_px, 1.0)
    Xc = (cx - W_img/2.0) * (-Z) / f_px
    Yc = (v_foot - H_img/2.0) * (-Z) / f_px
    return np.array([Xc, -Yc, Z], dtype=np.float32)

def pick_two_people(frame):
    """
    Pick up to two (bbox, smpl, id) tuples from the frame dict.
    优先：tracked_ids（活跃的人）与 tracked_bbox（同序）；
    SMPL 用 tid ↔ smpl 的一一对应映射来取。
    """
    # --- 取活跃ID（优先 tracked_ids，否则回退到 tid，再不行就空） ---
    ids = None
    if "tracked_ids" in frame:
        ids = frame["tracked_ids"]
    elif "tid" in frame:
        ids = frame["tid"]
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    if ids is None:
        ids = []
    if not isinstance(ids, (list, tuple)):
        ids = [ids]
    ids = [int(i) for i in ids]  # 统一成 int

    # --- 取 bbox：优先 tracked_bbox；否则回退 bbox（会按 ids 的顺序取下标 j） ---
    tb = frame.get("tracked_bbox", None)
    if isinstance(tb, np.ndarray):
        tracked_bboxes = tb.tolist() if tb.ndim > 1 else [tb.tolist()]
    elif isinstance(tb, (list, tuple)):
        tracked_bboxes = list(tb)
    else:
        tracked_bboxes = None

    fb = frame.get("bbox", None)
    if isinstance(fb, np.ndarray):
        fallback_bboxes = fb.tolist() if fb.ndim > 1 else [fb.tolist()]
    elif isinstance(fb, (list, tuple)):
        fallback_bboxes = list(fb)
    else:
        fallback_bboxes = None

    # --- SMPL 列表 ---
    smpl_obj = frame.get("smpl", None)
    if isinstance(smpl_obj, dict):
        smpl_list = [smpl_obj]
    elif isinstance(smpl_obj, (list, tuple, np.ndarray)):
        smpl_list = list(smpl_obj)
    else:
        smpl_list = []

    # --- tid 列表（与 smpl 对齐）并建立 pid->smpl_index 映射 ---
    tids = frame.get("tid", None)
    if isinstance(tids, np.ndarray):
        tids = tids.tolist()
    if tids is None:
        tids = []
    if not isinstance(tids, (list, tuple)):
        tids = [tids]
    tids = [int(t) for t in tids]

    pid2sidx = {}
    if len(tids) == len(smpl_list) and len(tids) > 0:
        for si, pid in enumerate(tids):
            pid2sidx[int(pid)] = si
    else:
        # 回退：如果 smpl_list 与 ids 等长，则按 ids 顺序临时对齐
        if len(smpl_list) == len(ids) and len(ids) > 0:
            for j, pid in enumerate(ids):
                pid2sidx[int(pid)] = j
        elif len(smpl_list) == 1 and len(ids) >= 1:
            # 只有一个 SMPL，就映射给第一个 id
            pid2sidx[int(ids[0])] = 0
        # 其他情况找不到稳定映射的 pid 会被跳过

    # --- 组装 (bbox, smpl, id)（不强行截断到2，保持你的原逻辑） ---
    pairs = []
    for j, pid in enumerate(ids):
        # 取 smpl
        sidx = pid2sidx.get(int(pid), None)
        if sidx is None or sidx < 0 or sidx >= len(smpl_list):
            continue
        smpl_dict = smpl_list[sidx]

        # 取 bbox：优先 tracked_bbox 的第 j 个；否则回退 bbox 第 j 个
        bb = None
        if (tracked_bboxes is not None) and (j < len(tracked_bboxes)) and (tracked_bboxes[j] is not None):
            bb = _to_xywh(tracked_bboxes[j])
        elif (fallback_bboxes is not None) and (j < len(fallback_bboxes)) and (fallback_bboxes[j] is not None):
            bb = _to_xywh(fallback_bboxes[j])

        if bb is None:
            continue

        pairs.append((bb, smpl_dict, int(pid)))

    return pairs

# --- [改动2] 修复并完善：按下标读 camera，做单位换算与坐标系翻转 ---
def camT_from_frame(frame, idx):
    """
    从 frame['camera'][idx] 读取 T=[tx,ty,tz]（OpenCV坐标：X右,Y下,Z前），
    做单位换算到米，再转换到渲染坐标（X右,Y上,Z后；相机看向 -Z）。
    """
    cam = frame.get("camera")
    if cam is None:
        return None

    if isinstance(cam, np.ndarray):
        cam = cam.tolist() if cam.ndim > 1 else [cam.tolist()]
    if not isinstance(cam, (list, tuple)) or idx >= len(cam) or cam[idx] is None:
        return None

    T = np.array(cam[idx], dtype=np.float32).reshape(-1)[:3]
    if not np.isfinite(T).all():
        return None

    # 单位换算
    T *= _UNIT_SCALE.get(CAMERA_UNITS, 1.0)
    # OpenCV -> 渲染坐标：Y取反，Z取反（使物体位于相机前方，Z为负）
    T[1] = -T[1]
    T[2] = -T[2]
    return T

# --- [改动3] 新增：用 pid 在 tid 里找 index，再去 camera 取 ---
def camT_from_pid(frame, pid):
    """
    用 track id (pid) 去 frame['tid'] 里找下标，再从 frame['camera'] 取对应的 T。
    """
    cam = frame.get("camera")
    tids = frame.get("tid")
    if cam is None or tids is None:
        return None

    if isinstance(cam, np.ndarray):
        cam = cam.tolist() if cam.ndim > 1 else [cam.tolist()]
    if isinstance(tids, np.ndarray):
        tids = tids.tolist()

    try:
        tids_int = [int(t) for t in tids]
        idx = tids_int.index(int(pid))
    except Exception:
        return None

    return camT_from_frame(frame, idx)

def apply_render(data, output_dir, out_size=None, fps=30):
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # --- No sorting: use the original order as-is ---
    if isinstance(data, dict):
        frame_keys = list(data.keys())
    elif isinstance(data, list):
        frame_keys = list(range(len(data)))
    else:
        raise TypeError("Unsupported data container; expected dict or list.")

    # Output size
    if out_size and len(out_size) == 2:
        W_out, H_out = int(out_size[0]), int(out_size[1])
    else:
        W_out = H_out = None
        for k in frame_keys[:50]:
            sz = get_frame_image_size(data[k])
            if sz is not None:
                W_out, H_out = sz
                break

    print(f"[INFO] Output size = {W_out}x{H_out}")

    # SMPL model + renderer + camera + one directional light
    smpl_model = SMPL(model_path=SMPL_MODEL_DIR, gender=GENDER)
    renderer = pyrender.OffscreenRenderer(viewport_width=W_out, viewport_height=H_out)
    camera = pyrender.PerspectiveCamera(yfov=math.radians(YFOV_DEG))
    light  = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    img_paths = []

    for fi, fkey in enumerate(frame_keys):
        frame = data[fkey]
        pairs = pick_two_people(frame)
        if not pairs:
            continue

        W_img, H_img = (W_out, H_out)

        # Build a simple scene for this frame
        scene = pyrender.Scene(bg_color=np.array([0.,0.,0.,0.], dtype=np.float32))
        scene.add(camera, pose=np.eye(4, dtype=np.float32))
        scene.add(light,  pose=np.eye(4, dtype=np.float32))

        # Add up to two persons
        for k, (bb, smpl_dict, pid) in enumerate(pairs):
            x, y, w, h = bb
            cx = x + 0.5*w
            v_foot = y + h  # use bbox bottom as foot anchor

            # --- 优先使用 camera 中与 pid 对齐的平移；其次尝试按渲染下标；最后用 bbox+FOV 估计 ---
            T = camT_from_pid(frame, pid)
            if T is None:
                T = camT_from_frame(frame, k)
            if T is None:
                T = uvh_to_camera_T(cx, v_foot, h, W_img, H_img, math.radians(YFOV_DEG), DEFAULT_HEIGHT_M)

            color = COLORS[k % len(COLORS)]
            mesh = smpl_to_mesh(smpl_model, smpl_dict, color_rgba=color)
            M = np.eye(4, dtype=np.float32); M[:3,3] = T
            scene.add(mesh, pose=M)

        # Render to RGBA and save
        color_rgba, _ = renderer.render(scene)
        out_path = os.path.join(frames_dir, f"{fi:06d}.png")
        imageio.imwrite(out_path, color_rgba)
        img_paths.append(out_path)

        if (fi+1) % 50 == 0 or fi == len(frame_keys)-1:
            print(f"[{fi+1:03d}/{len(frame_keys)}] saved {out_path}")

    renderer.delete()

    # Encode frames to MP4 (black background render)
    video_path = os.path.join(output_dir, "output_two_sur_SG.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (W_out, H_out))
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        if (img.shape[1], img.shape[0]) != (W_out, H_out):
            img = cv2.resize(img, (W_out, H_out), interpolation=cv2.INTER_AREA)
        writer.write(img)
    writer.release()
    print(f"Done: {video_path}")

if __name__ == "__main__":
    import joblib
    os.makedirs(OUT_DIR, exist_ok=True)
    data = joblib.load(PKL_PATH)
    apply_render(data, OUT_DIR)

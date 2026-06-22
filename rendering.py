import os, re, math, platform
import numpy as np
import torch
import trimesh
import pyrender
import imageio.v2 as imageio
import cv2
from smplx import SMPL

# ===================== 可调参数 =====================
SMPL_MODEL_DIR = "data/smpl"
GENDER = "NEUTRAL"
FPS = 30

# 输出分辨率；None 表示从数据里推断
OUT_SIZE = None   # 比如 (1920,1080)

# 垂直视场角
YFOV_DEG = 60.0

# 身高假设（米）
DEFAULT_HEIGHT_M = 1.70
PERSON_HEIGHT_OVERRIDE = {}  # 例如 {0:1.76, 1:1.62}

# 锚点：'foot' 用脚底；'bbox' 用 bbox 中心
ANCHOR_MODE = "foot"

# 相机重心对齐：'first_frame' 仅用于前若干帧居中；'none' 完全不动
RECENTER_MODE = "first_frame"
LOCK_RECENTER_AFTER = 80   # 仅前 80 帧动态居中，之后锁定

# 平滑
EMA_BETA_POS = 0.85
EMA_BETA_RECENTER = 0.90

# 垂直翻转网格（人物倒了就 True，否则 False）
FLIP_Y = True

# 水平镜像（方向反了就 True）
LR_MIRROR = True

# 如感觉“朝向”也反了，可把所有网格绕Y轴旋转180度
ROTATE_YAW_180 = False

# 光照
AMBIENT = np.array([0.25, 0.25, 0.25, 1.0])
DIR_INT = 3.0
LIGHT_DIRS = [(0, 1, 2), (2, 2, 1), (-2, 2, 1)]

BASE_COLORS = [
    (230, 76, 60, 255),
    (72, 133, 237, 255),
    (60, 186, 84, 255),
    (244, 180, 0, 255),
    (171, 71, 188, 255),
    (0, 172, 193, 255),
    (255, 112, 67, 255),
]

# ===================== 工具函数 =====================
def to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

def matrix_to_axis_angle(mat):
    """ mat: [N,3,3] torch -> [N,3] """
    cos_theta = (mat[:,0,0] + mat[:,1,1] + mat[:,2,2] - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    axis = torch.stack([
        mat[:,2,1] - mat[:,1,2],
        mat[:,0,2] - mat[:,2,0],
        mat[:,1,0] - mat[:,0,1]
    ], dim=1)
    axis = torch.nn.functional.normalize(axis, dim=1)
    aa = axis * theta.unsqueeze(1)
    aa[torch.isnan(aa)] = 0.0
    return aa

def smpl_to_mesh(smpl_model, smpl_dict, color_rgba=(200,200,200,255), flip_y=False, yaw180=False):
    betas = to_tensor(smpl_dict.get("betas", np.zeros(10, np.float32))).unsqueeze(0)

    body_pose_mats = to_tensor(smpl_dict["body_pose"]).reshape(-1,3,3)
    body_aa = matrix_to_axis_angle(body_pose_mats).reshape(1,-1)

    g = to_tensor(smpl_dict["global_orient"])
    g_aa = matrix_to_axis_angle(g.reshape(-1,3,3)).reshape(1,-1)

    out = smpl_model(betas=betas, body_pose=body_aa, global_orient=g_aa)
    verts = out.vertices[0].detach().cpu().numpy()
    faces = smpl_model.faces

    if flip_y:
        verts[:,1] *= -1.0

    if yaw180:
        # 绕Y轴旋转 180°
        R = np.array([[-1,0,0],[0,1,0],[0,0,-1]], dtype=np.float32)
        verts = (R @ verts.T).T

    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    tri.visual.vertex_colors = np.tile(np.array(color_rgba, dtype=np.uint8), (tri.vertices.shape[0], 1))
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=np.array(color_rgba, dtype=np.float32)/255.0,
        metallicFactor=0.0, roughnessFactor=0.9, doubleSided=True
    )
    return pyrender.Mesh.from_trimesh(tri, material=material, smooth=False)

# ---- 帧序排序（兼容数字/路径/文件名） ----
def _is_frame_entry(v):
    if isinstance(v, dict):
        keys = v.keys()
        return any(k in keys for k in (
            "smpl","smpl_list","smplx","smpl_data",
            "bbox","tracked_bbox","camera_bbox","annotations",
            "frame_path","img_path","img_name","2d_joints","3d_joints"
        ))
    return False

def _extract_idx_from_key(k):
    try:
        return int(k)
    except Exception:
        pass
    s = str(k)
    base = os.path.splitext(os.path.basename(s))[0]
    m = re.search(r'(\d+)$', base) or re.search(r'(\d+)', base)
    return int(m.group(1)) if m else 0

def _extract_idx_from_value(v):
    for kk in ("frame_idx","frameId","frame","fid"):
        if kk in v and isinstance(v[kk], (int,float)):
            return int(v[kk])
    return None

def sorted_frame_keys(data):
    frame_items = []
    for k, v in data.items():
        if _is_frame_entry(v):
            idx = _extract_idx_from_value(v)
            if idx is None:
                idx = _extract_idx_from_key(k)
            frame_items.append((idx, k))
    if frame_items:
        frame_items.sort(key=lambda x: x[0])
        return [k for _,k in frame_items]
    if isinstance(data, list):
        return list(range(len(data)))
    return list(data.keys())

# ---- bbox 统一为 (x,y,w,h) ----
def _xyxy_to_xywh(bb):
    x1,y1,x2,y2 = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    return (x1, y1, x2-x1, y2-y1)

def _guess_one_bbox(b):
    b = np.array(b).reshape(-1)
    if len(b) < 4: return None
    x,y,a,b2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    # 兼容 xywh/xyxy
    if a > 0 and b2 > 0 and (a < 30000 and b2 < 30000):
        if (a > x) and (b2 > y) and (a-x) > 0 and (b2-y) > 0:
            return _xyxy_to_xywh([x,y,a,b2])
        return (x,y,a,b2)
    return _xyxy_to_xywh([x,y,a,b2])

def find_bboxes(frame):
    cands, ids = [], None
    for key in ("bbox","tracked_bbox","camera_bbox","annotations"):
        if key in frame:
            val = frame[key]
            if isinstance(val, (list,tuple)):
                cands = val
            elif isinstance(val, np.ndarray):
                cands = list(val) if val.ndim > 1 else [val]
            break
    for key in ("tracked_ids","tid","ids"):
        if key in frame:
            ids = frame[key]
            if isinstance(ids, np.ndarray): ids = ids.tolist()
            break
    bboxes = []
    for b in cands:
        bb = _guess_one_bbox(b)
        if bb is not None and bb[2] > 1 and bb[3] > 1:
            bboxes.append(bb)
    if ids is None:
        ids = list(range(len(bboxes)))
    n = min(len(bboxes), len(ids))
    return bboxes[:n], ids[:n]

# ---- 图像尺寸 ----
def get_frame_image_size(frame):
    for key in ("size","img_size","image_size","H_W","WH","shape"):
        if key in frame:
            arr = np.array(frame[key]).astype(np.int64).reshape(-1).tolist()
            if len(arr) >= 2:
                H,W = int(arr[0]), int(arr[1])
                if H < 10 and W > H:  # 兼容 (W,H)
                    H,W = W,H
                return (W,H)
    for k in ("frame_path","img_path","image_path"):
        if k in frame and isinstance(frame[k], str) and os.path.isfile(frame[k]):
            im = cv2.imread(frame[k])
            if im is not None:
                return (im.shape[1], im.shape[0])
    return None

# ---- 像素→相机系（含左右镜像开关） ----
def uvh_to_camera_T(u, v_anchor, h_px, W_img, H_img, yfov_rad, Hm, lr_mirror=False):
    # 相机朝 -Z，看向前方；f 使用竖直 FOV
    f_px = H_img / (2 * math.tan(yfov_rad / 2.0))
    Z = -(Hm * f_px) / max(h_px, 1.0)  # 越近 |Z| 越小（更接近 0）
    Xc = (u - W_img/2.0) * (-Z) / f_px
    Yc = (v_anchor - H_img/2.0) * (-Z) / f_px
    if lr_mirror:
        Xc = -Xc                # 修“方向反了”
    return np.array([Xc, -Yc, Z], dtype=np.float32)  # OpenGL +Y向上

# ---- EMA ----
class EMA:
    def __init__(self, beta, init=None):
        self.beta = float(beta)
        self.x = None if init is None else np.array(init, dtype=np.float32)
    def update(self, v):
        v = np.array(v, dtype=np.float32)
        if self.x is None:
            self.x = v
        else:
            self.x = self.beta*self.x + (1.0-self.beta)*v
        return self.x

# ===================== 主流程 =====================
def render_frames_from_pkl(pkl_path, output_dir, out_size=None, fps=30):
    import joblib
    data = joblib.load(pkl_path)

    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_keys = sorted_frame_keys(data)
    print(f"Parsed {len(frame_keys)} frames. First 5: {frame_keys[:5]}")

    smpl_model = SMPL(model_path=SMPL_MODEL_DIR, gender=GENDER)

    # 输出尺寸
    if out_size and len(out_size) == 2:
        W_out, H_out = int(out_size[0]), int(out_size[1])
    else:
        W_out = H_out = None
        for k in frame_keys:
            sz = get_frame_image_size(data[k])
            if sz is not None:
                W_out, H_out = sz
                break
        if W_out is None:
            W_out, H_out = 1280, 720
    print(f"[INFO] Output size = {W_out}x{H_out}")

    # 渲染器
    renderer = pyrender.OffscreenRenderer(viewport_width=W_out, viewport_height=H_out)

    # 相机
    camera = pyrender.PerspectiveCamera(yfov=math.radians(YFOV_DEG))
    cam_pose = np.eye(4, dtype=np.float32)

    # 灯光
    lights = [pyrender.DirectionalLight(color=np.ones(3), intensity=DIR_INT) for _ in LIGHT_DIRS]

    # 颜色/平滑
    id_to_color = {}
    id_to_ema = {}
    recenter_ema = EMA(EMA_BETA_RECENTER, init=[0,0,0])
    base_offset = np.zeros(3, dtype=np.float32)
    base_set = False

    img_paths = []

    for fi, fkey in enumerate(frame_keys):
        frame = data[fkey]

        bboxes, tids = find_bboxes(frame)
        smpl_list = frame.get("smpl", frame.get("smpl_list", None))
        if smpl_list is None:
            smpl_one = frame.get("smpl_data", None)
            if isinstance(smpl_one, dict):
                smpl_list = [smpl_one]
        if isinstance(smpl_list, dict):
            smpl_list = [smpl_list]
        if smpl_list is None:
            smpl_list = []

        n = min(len(smpl_list), len(bboxes))
        extra = max(0, len(smpl_list) - n)

        sz = get_frame_image_size(frame)
        W_img, H_img = (sz if sz is not None else (W_out, H_out))

        T_list = []
        pack_list = []

        for i in range(n):
            smpl_dict = smpl_list[i]
            x, y, w, h = bboxes[i]
            cx = x + 0.5*w
            cy = y + 0.5*h
            v_anchor = (y + h) if (ANCHOR_MODE == "foot") else cy

            pid = tids[i] if i < len(tids) else i
            color = id_to_color.setdefault(pid, BASE_COLORS[pid % len(BASE_COLORS)])
            Hm = PERSON_HEIGHT_OVERRIDE.get(pid, DEFAULT_HEIGHT_M)

            T = uvh_to_camera_T(cx, v_anchor, h, W_img, H_img,
                                math.radians(YFOV_DEG), Hm, lr_mirror=LR_MIRROR)
            ema = id_to_ema.setdefault(pid, EMA(EMA_BETA_POS))
            T_smooth = ema.update(T)

            T_list.append(T_smooth)

            mesh = smpl_to_mesh(smpl_model, smpl_dict,
                                color_rgba=color, flip_y=FLIP_Y, yaw180=ROTATE_YAW_180)
            M = np.eye(4, dtype=np.float32)
            M[:3,3] = T_smooth
            pack_list.append((mesh, M))

            print(f"[{fi:05d}] id={pid} use=bbox u={cx:.1f} v={(y+h if ANCHOR_MODE=='foot' else cy):.1f} "
                  f"h={h:.1f} | Hm={Hm:.2f} -> (X,Y,Z)=({T[0]:.1f},{T[1]:.1f},{T[2]:.2f})")

        # 处理多出的 smpl（无 bbox）
        for j in range(extra):
            idx = n + j
            smpl_dict = smpl_list[idx]
            pid = 10_000 + idx
            color = id_to_color.setdefault(pid, BASE_COLORS[pid % len(BASE_COLORS)])
            mesh = smpl_to_mesh(smpl_model, smpl_dict,
                                color_rgba=color, flip_y=FLIP_Y, yaw180=ROTATE_YAW_180)
            M = np.eye(4, dtype=np.float32)
            M[:3,3] = np.array([0.0, -0.9, -5.0], dtype=np.float32)
            pack_list.append((mesh, M))
            T_list.append(M[:3,3])

        # 相机位姿（只在前 LOCK_RECENTER_AFTER 帧动态对齐）
        cam_pose = np.eye(4, dtype=np.float32)
        if RECENTER_MODE == "first_frame" and len(T_list) > 0:
            T_arr = np.stack(T_list, axis=0)
            mean_xy = np.mean(T_arr[:,:2], axis=0)
            if not base_set:
                base_offset = np.array([-mean_xy[0], -mean_xy[1], 0.0], dtype=np.float32)
                base_set = True
            target_off = np.array([-mean_xy[0], -mean_xy[1], 0.0], dtype=np.float32)
            if fi < LOCK_RECENTER_AFTER:
                base_offset = recenter_ema.update(target_off)
            # 超过锁定帧数后，不再更新 base_offset
            cam_pose[:3,3] = -base_offset

        # 渲染
        scene = pyrender.Scene(bg_color=np.array([0.,0.,0.,0.], dtype=np.float32),
                               ambient_light=AMBIENT)
        scene.add(camera, pose=cam_pose)
        for d in LIGHT_DIRS:
            L = np.eye(4, dtype=np.float32); L[:3,3] = np.array(d, dtype=np.float32)
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=DIR_INT), pose=L)
        for mesh, M in pack_list:
            scene.add(mesh, pose=M)

        color_rgba, _ = renderer.render(scene)
        out_path = os.path.join(frames_dir, f"{fi:06d}.png")
        imageio.imwrite(out_path, color_rgba)
        img_paths.append(out_path)

        if (fi+1) % 50 == 0 or fi == len(frame_keys)-1:
            print(f"[{fi+1:03d}/{len(frame_keys)}] saved {out_path}")

    renderer.delete()

    # 写视频
    video_path = os.path.join(output_dir, "output.mp4")
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
    print(f"✅ Done: {video_path}")

# ===================== CLI =====================
if __name__ == "__main__":
    import joblib
    PKL_PATH = "demo_surv_cut.pkl"
    OUT_DIR = "render_pixel_align_fix_dir"

    # 仅 Linux 且无显示时尝试 headless；mac/Windows 不要强制
    if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("PYGLET_HEADLESS", "True")

    os.makedirs(OUT_DIR, exist_ok=True)
    render_frames_from_pkl(PKL_PATH, OUT_DIR, out_size=OUT_SIZE, fps=FPS)


# export_objs_any_pkl.py
# 通吃 4D-Humans 的 track/demo 两类 PKL，导出逐帧 .OBJ
import os, sys, pickle, json, math
from pathlib import Path
import numpy as np
import torch

# ---------- 兼容 joblib(pkl压缩) 与 pickle ----------
def _detect_and_load(path):
    with open(path, "rb") as f:
        head = f.read(2)
    # 优先尝试 joblib（很多项目用 joblib.dump 保存为压缩pkl）
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        pass
    # 再试 pickle
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------- SMPL 前向 ----------
def build_smpl(model_dir: str, device: str = "cpu"):
    import os
    import smplx
    model_dir = os.path.abspath(model_dir)
    # 用“位置参数”避免关键字被误解析
    smpl = smplx.create(model_dir, 'smpl', gender='NEUTRAL', use_pca=False)
    return smpl.to(device)


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)

def smpl_forward_verts(smpl, pose72, betas10, device):
    """
    pose72: (72,) axis-angle, global(3)+body(69)
    betas10: (10,)
    return verts: (6890, 3)
    """
    global_orient = to_tensor(pose72[:3], device).view(1, 3)
    body_pose     = to_tensor(pose72[3:], device).view(1, -1)
    betas_t       = to_tensor(betas10, device).view(1, -1)
    out = smpl(global_orient=global_orient, body_pose=body_pose, betas=betas_t)
    return out.vertices[0].detach().cpu().numpy()

def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # SMPL faces 是 0-based 索引，.obj 需要 1-based
        for fa in faces:
            f.write(f"f {fa[0]+1} {fa[1]+1} {fa[2]+1}\n")

# ---------- 导出逻辑 ----------
def export_from_tracklike(data, out_dir: Path, smpl, device, stride=1, max_frames=None):
    """
    处理 track.py 结构：data['tracklets'] 或 data['persons'] 是 dict
    每个 track: pose(T,72), betas(T,10 or 1x10), frames(T,)
    """
    tracks = None
    if isinstance(data, dict):
        if "tracklets" in data and isinstance(data["tracklets"], dict):
            tracks = data["tracklets"]
        elif "persons" in data and isinstance(data["persons"], dict):
            tracks = data["persons"]
    if tracks is None:
        return False  # 不是这种格式

    faces = smpl.faces
    for tid, t in tracks.items():
        if t is None or not isinstance(t, dict):
            continue
        pose  = np.asarray(t.get("pose", []),  dtype=np.float32)  # (T,72)
        betas = np.asarray(t.get("betas", []), dtype=np.float32)  # (T,10) or (10,)
        frames = np.asarray(t.get("frames", np.arange(len(pose))), dtype=int)

        if pose.ndim != 2 or pose.shape[1] != 72 or len(pose) == 0:
            continue

        if betas.ndim == 1:
            betas = np.tile(betas[None, :], (len(pose), 1))
        elif betas.ndim == 2 and betas.shape[0] != len(pose):
            # 有些导出只存了单个 betas；做个保险
            betas = np.tile(betas[0][None, :], (len(pose), 1))

        # 采样控制
        idxs = np.arange(len(pose))[::max(1, int(stride))]
        if max_frames is not None:
            idxs = idxs[:max_frames]

        for i in idxs:
            verts = smpl_forward_verts(smpl, pose[i], betas[i], device)
            fn = out_dir / f"track{tid}_frame{int(frames[i]):06d}.obj"
            write_obj(fn, verts, faces)

    return True

def export_from_demolike(data, out_dir: Path, smpl, device, start_idx=0, stride=1, max_frames=None):
    """
    处理 demo.py 结构：dict 的 key 是图片路径；value 可能是：
    - dict: 单人结果（含 'pose','betas'）
    - list: 多人结果列表（每个含 'pose','betas'）
    我们按字典键的排序作为帧序（0,1,2,...）
    """
    if not isinstance(data, dict):
        return False

    faces = smpl.faces

    # 排序稳定（按路径名）
    keys = sorted(list(data.keys()))
    # 采样控制
    frame_keys = keys[start_idx::max(1, int(stride))]
    if max_frames is not None:
        frame_keys = frame_keys[:max_frames]

    for fi, k in enumerate(frame_keys):
        rec = data[k]
        # 多人
        if isinstance(rec, list):
            persons = rec
        else:
            persons = [rec]

        for pi, p in enumerate(persons):
            if not isinstance(p, dict):
                continue
            pose  = np.asarray(p.get("pose", []),  dtype=np.float32)  # (72,) 或 (1,72)
            betas = np.asarray(p.get("betas", []), dtype=np.float32)  # (10,) 或 (1,10)

            # 统一形状
            pose  = pose.reshape(-1)
            betas = betas.reshape(-1)
            if pose.size != 72 or betas.size == 0:
                # 有些 demo 结果可能直接包含 mesh 顶点；如果你想直接写 .obj，
                # 可在此添加从 'verts' 字段直接写入的分支。
                continue

            verts = smpl_forward_verts(smpl, pose, betas, device)
            fn = out_dir / f"frame{fi:06d}_person{pi:02d}.obj"
            write_obj(fn, verts, faces)

    return True

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Export .OBJ sequence from 4D-Humans PKL (demo or track)")
    ap.add_argument("pkl", type=str, help="input .pkl (pickle or joblib)")
    ap.add_argument("out_dir", type=str, help="output directory for OBJ sequence")
    ap.add_argument("--smpl_dir", type=str, default="data", help="directory containing SMPL neutral model")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--stride", type=int, default=1, help="sample every N frames (track) or images (demo)")
    ap.add_argument("--max_frames", type=int, default=None, help="limit number of frames exported")
    ap.add_argument("--start", type=int, default=0, help="start index for demo-format (ignored for track)")
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _detect_and_load(str(pkl_path))
    smpl = build_smpl(args.smpl_dir, args.device)

    # 尝试 track-like
    ok = export_from_tracklike(data, out_dir, smpl, args.device, stride=args.stride, max_frames=args.max_frames)
    if not ok:
        # 尝试 demo-like
        ok = export_from_demolike(data, out_dir, smpl, args.device, start_idx=args.start, stride=args.stride, max_frames=args.max_frames)

    if not ok:
        print("未识别的数据结构。请确认这是 4D-Humans 的输出 pkl。")
        sys.exit(2)

    print(f"[OK] OBJ 导出完成 → {out_dir}")

if __name__ == "__main__":
    main()

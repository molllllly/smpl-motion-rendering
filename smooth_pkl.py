import os, sys, pickle, numpy as np
import joblib

def ema_filter(arr, alpha=0.2):
    if arr is None or len(arr)==0: return arr
    out = arr.copy()
    for t in range(1, len(arr)):
        out[t] = alpha * arr[t] + (1.0 - alpha) * out[t-1]
    return out

def load_pkl(p):
    with open(p, "rb") as f:  print(f.read(20)); return pickle.load(f)

def save_pkl(obj, p):
    with open(p, "wb") as f: pickle.dump(obj, f)

def main(in_pkl, out_pkl=None, alpha=0.2):
    #data = load_pkl(in_pkl)

    data = joblib.load(in_pkl)
    print(type(data), list(data.keys()))
    tracks = data.get("tracklets", data.get("persons", {}))
    if not isinstance(tracks, dict):
        raise ValueError("Unrecognized PKL structure; expect dict under 'tracklets' or 'persons'.")

    for tid, t in tracks.items():
        if t is None: continue
        if "pose" in t and isinstance(t["pose"], (list, np.ndarray)):
            t["pose"] = ema_filter(np.asarray(t["pose"], dtype=np.float32), alpha)
        if "cam" in t and isinstance(t["cam"], (list, np.ndarray)):
            t["cam"]  = ema_filter(np.asarray(t["cam"],  dtype=np.float32), alpha)

    out_pkl = out_pkl or in_pkl.replace(".pkl", f"_smooth_a{alpha}.pkl")
    save_pkl(data, out_pkl)
    print(f"[OK] saved smoothed PKL → {out_pkl}")



if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python smooth_pkl.py input.pkl [alpha]")
        sys.exit(1)
    in_pkl = sys.argv[1]
    alpha = float(sys.argv[2]) if len(sys.argv)>=3 else 0.2
    main(in_pkl, alpha=alpha)
import os, sys, pickle, numpy as np
import joblib

def ema_filter(arr, alpha=0.2):
    if arr is None or len(arr)==0: return arr
    out = arr.copy()
    for t in range(1, len(arr)):
        out[t] = alpha * arr[t] + (1.0 - alpha) * out[t-1]
    return out

def load_pkl(p):
    with open(p, "rb") as f:  print(f.read(20)); return pickle.load(f)

def save_pkl(obj, p):
    with open(p, "wb") as f: pickle.dump(obj, f)

def main(in_pkl, out_pkl=None, alpha=0.2):
    #data = load_pkl(in_pkl)

    data = joblib.load(in_pkl)
    print(type(data), list(data.keys()))
    tracks = data.get("tracklets", data.get("persons", {}))
    if not isinstance(tracks, dict):
        raise ValueError("Unrecognized PKL structure; expect dict under 'tracklets' or 'persons'.")

    for tid, t in tracks.items():
        if t is None: continue
        if "pose" in t and isinstance(t["pose"], (list, np.ndarray)):
            t["pose"] = ema_filter(np.asarray(t["pose"], dtype=np.float32), alpha)
        if "cam" in t and isinstance(t["cam"], (list, np.ndarray)):
            t["cam"]  = ema_filter(np.asarray(t["cam"],  dtype=np.float32), alpha)

    out_pkl = out_pkl or in_pkl.replace(".pkl", f"_smooth_a{alpha}.pkl")
    save_pkl(data, out_pkl)
    print(f"[OK] saved smoothed PKL → {out_pkl}")



if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python smooth_pkl.py input.pkl [alpha]")
        sys.exit(1)
    in_pkl = sys.argv[1]
    alpha = float(sys.argv[2]) if len(sys.argv)>=3 else 0.2
    main(in_pkl, alpha=alpha)

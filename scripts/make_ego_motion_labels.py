# scripts/make_ego_motion_labels.py
import json
import math
from pathlib import Path
import numpy as np
from nuscenes.nuscenes import NuScenes

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"
OUT_PATH = r"artifacts\ego_motion_labels_T8_keyframes.json"

def l2(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def main():
    nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)

    clips = json.loads(Path(INDEX_PATH).read_text())
    # Expect each entry has: start_sd_token, T, stride, keyframes_only
    speeds = []
    per_clip = []

    for ci, c in enumerate(clips):
        sd_token = c["start_sd_token"]
        T = int(c["T"])
        stride = int(c["stride"])

        toks = []
        cur = sd_token
        for _ in range(T):
            toks.append(cur)
            sd = nusc.get("sample_data", cur)
            cur = sd["next"]
            # stride>1 support: advance extra
            for _k in range(stride-1):
                if cur == "":
                    break
                cur = nusc.get("sample_data", cur)["next"]
            if cur == "" and len(toks) < T:
                break

        if len(toks) < 2:
            continue

        ts = []
        pos = []
        for tkn in toks:
            sd = nusc.get("sample_data", tkn)
            ts.append(int(sd["timestamp"]))
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            tr = ep["translation"]  # [x,y,z]
            pos.append([float(tr[0]), float(tr[1]), float(tr[2])])

        vs = []
        for i in range(1, len(toks)):
            dt = (ts[i] - ts[i-1]) * 1e-6  # us -> s
            if dt <= 0:
                continue
            dx = l2(pos[i], pos[i-1])
            vs.append(dx / dt)

        if len(vs) == 0:
            continue

        clip_speed = float(np.median(vs))
        speeds.append(clip_speed)
        per_clip.append({"clip_idx": ci, "speed_mps": clip_speed})

    speeds_np = np.array(speeds, dtype=np.float64)
    q1 = float(np.quantile(speeds_np, 1/3))
    q2 = float(np.quantile(speeds_np, 2/3))

    for d in per_clip:
        v = d["speed_mps"]
        if v <= q1:
            b = 0
        elif v <= q2:
            b = 1
        else:
            b = 2
        d["bucket"] = int(b)

    out = {
        "index_path": INDEX_PATH,
        "num_labeled": len(per_clip),
        "quantiles_mps": {"q33": q1, "q66": q2},
        "labels": per_clip,
        "bucket_names": ["slow", "medium", "fast"],
    }

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_PATH).write_text(json.dumps(out, indent=2))
    print("wrote:", OUT_PATH)
    print("num_labeled:", out["num_labeled"])
    print("quantiles (m/s):", out["quantiles_mps"])
    print("speed stats (m/s): mean", float(speeds_np.mean()), "std", float(speeds_np.std()),
          "min", float(speeds_np.min()), "max", float(speeds_np.max()))

if __name__ == "__main__":
    main()

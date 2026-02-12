from nuscenes.nuscenes import NuScenes

from src.data.clip_index import load_clip_index
from src.data.clip_sampler import sample_clip_from_start

DATAROOT = r"C:\DS\TPV\nuScenes"
INDEX_PATH = r"artifacts\clip_index_T8_s1_keyframes.json"

nusc = NuScenes(version="v1.0-trainval", dataroot=DATAROOT, verbose=False)
items = load_clip_index(INDEX_PATH)

print("loaded clips:", len(items))

bad = 0
for i, spec in enumerate(items):
    try:
        clip = sample_clip_from_start(
            nusc,
            start_sd_token=spec.start_sd_token,
            T=spec.T,
            stride=spec.stride,
            keyframes_only=spec.keyframes_only,
        )

        # Invariants
        assert len(clip.frames) == spec.T
        assert all(f.is_key_frame for f in clip.frames)
        assert all(not f.filename.startswith("sweeps/") for f in clip.frames)

        ts = [f.timestamp for f in clip.frames]
        assert all(ts[j] < ts[j + 1] for j in range(len(ts) - 1))

        deltas = [ts[j + 1] - ts[j] for j in range(len(ts) - 1)]
        assert all(d > 0 for d in deltas)

        # Optional sanity (not a strict spec): keyframes are usually ~0.5s apart.
        # We won't hardcode exact expected deltas, but we can flag extreme irregularity.
        if max(deltas) > 2_000_000 or min(deltas) < 200_000:
            print(f"[warn] clip {i} unusual deltas: {deltas}")

    except Exception as e:
        bad += 1
        print(f"[FAIL] clip {i} spec={spec} err={repr(e)}")

print("bad clips:", bad)
if bad != 0:
    raise SystemExit("Clip index validation failed")
print("OK: clip index validation passed")

from __future__ import annotations

import argparse
import os
import sys
from nuscenes.nuscenes import NuScenes


def main() -> int:
    p = argparse.ArgumentParser(description="nuScenes CAM_FRONT sanity check (load + keys + temporal clip walk)")
    p.add_argument("--dataroot", type=str, required=True, help="Path to nuScenes root folder")
    p.add_argument("--version", type=str, default="v1.0-trainval", help="nuScenes version (e.g., v1.0-trainval)")
    p.add_argument("--scene-idx", type=int, default=0, help="Scene index to inspect")
    p.add_argument("--T", type=int, default=8, help="Clip length to walk forward")
    p.add_argument("--verbose", action="store_true", help="Enable NuScenes verbose loading logs")
    args = p.parse_args()

    dataroot = args.dataroot
    version = args.version

    if not os.path.isdir(dataroot):
        print(f"ERROR: dataroot is not a directory: {dataroot}")
        return 2

    print("CHECK.PY STARTED")
    print("python =", sys.executable)
    print("cwd    =", os.getcwd())
    print("DATAROOT =", dataroot)
    print("VERSION  =", version)
    print("======")

    # --- Part A: Load NuScenes and inspect keys (your earlier Step 1) ---
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=args.verbose)
    print("======")
    print("NuScenes init OK")
    print("Number of scenes:", len(nusc.scene))

    if not (0 <= args.scene_idx < len(nusc.scene)):
        print(f"ERROR: scene-idx out of range: {args.scene_idx}")
        return 3

    scene = nusc.scene[args.scene_idx]
    print("Scene keys:", list(scene.keys()))

    first_sample_token = scene["first_sample_token"]
    sample = nusc.get("sample", first_sample_token)

    print("Sample keys:", list(sample.keys()))
    sensors = list(sample["data"].keys())
    print("Available sensors:", sensors)

    if "CAM_FRONT" not in sample["data"]:
        print("ERROR: CAM_FRONT not present in sample['data']")
        return 4

    cam_front_token = sample["data"]["CAM_FRONT"]
    cam_data = nusc.get("sample_data", cam_front_token)

    print("CAM_FRONT sample_data keys:", list(cam_data.keys()))
    print("Filename:", cam_data["filename"])

    full_path = os.path.join(dataroot, cam_data["filename"])
    print("Full path:", full_path)
    print("Full path exists:", os.path.exists(full_path))

    if not os.path.exists(full_path):
        print("ERROR: CAM_FRONT file does not exist on disk. Check dataroot.")
        return 5

    # --- Part B: Walk forward to build a clip (your Step 2) ---
    print("======")
    print(f"Temporal walk (CAM_FRONT) for T={args.T} from scene_idx={args.scene_idx}")

    token = first_sample_token
    timestamps = []
    paths = []

    for i in range(args.T):
        sample_i = nusc.get("sample", token)
        cam_token = sample_i["data"]["CAM_FRONT"]
        cam_i = nusc.get("sample_data", cam_token)

        path_i = os.path.join(dataroot, cam_i["filename"])
        ok = os.path.exists(path_i)

        print(
            f"[{i}] sample_ts={sample_i['timestamp']} cam_ts={cam_i['timestamp']} "
            f"key={cam_i['is_key_frame']} exists={ok}"
        )
        print("     ", cam_i["filename"])

        if not ok:
            print("ERROR: Missing CAM_FRONT image file during walk:", path_i)
            return 6

        timestamps.append(cam_i["timestamp"])
        paths.append(path_i)

        if sample_i["next"] == "":
            print("Reached end of scene early.")
            break
        token = sample_i["next"]

    is_monotonic = all(timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1))
    print("Monotonic camera timestamps:", is_monotonic)
    print("Clip length collected:", len(paths))

    if len(paths) < args.T:
        print("NOTE: Collected fewer than T frames because scene ended (this is OK).")

    if not is_monotonic:
        print("ERROR: Non-monotonic camera timestamps. This indicates a traversal or indexing issue.")
        return 7

    print("======")
    print("SANITY CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Synthesize a fake intermediate dir end-to-end (no Waymo, no GPU).

Produces a directory that looks like waymo_loader.py output, so the rest of
the pipeline (perturb, projection, rds_writer, yolo_eval) can be exercised
without downloading anything.

Layout produced:
    <root>/
      manifest.json
      calibrations.json
      ego_poses.parquet
      tracklets.parquet
      frames/FRONT/00000.png ... 00019.png

The scene: a vehicle driving 8m ahead of ego, moving forward at 5 m/s, plus
a pedestrian standing 6m ahead and 1m to the left.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# Imports from the package — assume cwd contains pipeline/.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema import EGO_POSE_SCHEMA, TRACKLET_SCHEMA  # noqa: E402


# Realistic pinhole intrinsics for a 1920x1280 front camera.
_INTRINSIC = np.array([1500.0, 1500.0, 960.0, 640.0, 0, 0, 0, 0, 0], dtype=np.float64)
# Waymo convention: extrinsic = vehicle→camera (4x4). Mount the camera 1.5m up,
# 1.5m forward of ego origin, pointing straight ahead.
_EXTRINSIC = np.array([
    [1, 0, 0, -1.5],
    [0, 1, 0, 0.0],
    [0, 0, 1, -1.5],
    [0, 0, 0, 1.0],
], dtype=np.float64)
_WIDTH, _HEIGHT = 1920, 1280
_NUM_FRAMES = 20
_FPS = 10.0


def write_synth(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "frames" / "FRONT").mkdir(parents=True, exist_ok=True)

    # Black PNG per frame.
    blank = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    for i in range(_NUM_FRAMES):
        Image.fromarray(blank).save(root / "frames" / "FRONT" / f"{i:05d}.png")

    # Tracklets: two objects across all frames.
    rows = []
    dt = 1.0 / _FPS
    for i in range(_NUM_FRAMES):
        # Vehicle 8m ahead, moving forward 5 m/s, will close on ego.
        rows.append({
            "object_id": "veh_001",
            "frame_idx": i,
            "type": 1,
            "cx": 8.0 + 5.0 * i * dt,
            "cy": 0.0,
            "cz": 0.75,
            "length": 4.5,
            "width": 1.8,
            "height": 1.5,
            "heading": 0.0,
            "speed_x": 5.0,
            "speed_y": 0.0,
            "num_lidar_points": 200,
            "perturbed": False,
            "perturb_kind": None,
        })
        # Pedestrian 6m ahead, 1m left, stationary.
        rows.append({
            "object_id": "ped_001",
            "frame_idx": i,
            "type": 2,
            "cx": 6.0,
            "cy": 1.0,
            "cz": 0.9,
            "length": 0.6,
            "width": 0.6,
            "height": 1.8,
            "heading": 0.0,
            "speed_x": 0.0,
            "speed_y": 0.0,
            "num_lidar_points": 50,
            "perturbed": False,
            "perturb_kind": None,
        })

    df = pd.DataFrame(rows)
    # Force perturb_kind dtype so pyarrow schema cast doesn't blow up.
    df["perturb_kind"] = df["perturb_kind"].astype("object")
    table = pa.Table.from_pandas(df, schema=TRACKLET_SCHEMA, preserve_index=False)
    pq.write_table(table, root / "tracklets.parquet")

    # Ego poses (identity).
    pose_rows = [{
        "frame_idx": i,
        "timestamp_us": int(i * 1e5),
        "transform": np.eye(4).flatten().tolist(),
    } for i in range(_NUM_FRAMES)]
    pose_df = pd.DataFrame(pose_rows)
    pose_table = pa.Table.from_pandas(pose_df, schema=EGO_POSE_SCHEMA, preserve_index=False)
    pq.write_table(pose_table, root / "ego_poses.parquet")

    # Calibrations.
    calibs = {
        "FRONT": {
            "width": _WIDTH,
            "height": _HEIGHT,
            "intrinsic": _INTRINSIC.tolist(),
            "extrinsic": _EXTRINSIC.tolist(),
        }
    }
    with open(root / "calibrations.json", "w") as f:
        json.dump(calibs, f, indent=2)

    # Manifest.
    manifest = {
        "segment_name": "synth_segment",
        "num_frames": _NUM_FRAMES,
        "frame_rate_hz": _FPS,
        "cameras": ["FRONT"],
        "intermediate_dir": str(root),
    }
    with open(root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return root


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    write_synth(args.out)
    print(f"wrote synthetic intermediate at {args.out}")

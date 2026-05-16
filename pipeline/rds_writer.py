"""DreamLoop intermediate → Cosmos-Drive-Dreams RDS-HQ format.

NOTE — schema gap:
    The exact RDS-HQ on-disk layout is defined inside nv-tlabs/Cosmos-Drive-Dreams.
    The fields we know we need:
      - per-frame RGB (have these as PNGs)
      - per-frame 3D tracklets in ego frame (have these in tracklets.parquet)
      - per-camera intrinsics + extrinsics (have these in calibrations.json)
      - ego pose stream (have this)
      - per-frame wireframe conditioning images (we render these ourselves)

    Until Team Data clones Cosmos-Drive-Dreams and confirms the exact directory
    + file naming convention this script expects, this module:
      1. exposes a pure function `build_rds_hq_dir(...)` that writes our best guess
         at the layout, derived from the official nv-tlabs README + scripts/
      2. emits a `_rds_hq_manifest.json` we control, so Cosmos can be patched to
         read from it directly if their layout doesn't match
      3. renders wireframe overlays per frame, which is the most important and
         least ambiguous deliverable

    HOUR 1–2 TODO when on the GX10:
      - Run scripts/render_from_rds_hq.py in Cosmos-Drive-Dreams on the *sample
        data they ship* with --dry-run
      - Print the directory tree and file extensions it expects
      - Adjust DEFAULT_LAYOUT below to match
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

import config
from box_projection import (
    box_corners_ego,
    load_calibration,
    project_corners_to_image,
)


# Best-guess layout. Confirm against Cosmos-Drive-Dreams repo before relying on it.
DEFAULT_LAYOUT = {
    "rgb_dir": "rgb/{camera}",
    "wireframe_dir": "wireframe/{camera}",
    "depth_dir": "depth/{camera}",                  # optional — empty if we don't have it
    "tracklets_file": "labels/tracklets.parquet",
    "ego_poses_file": "labels/ego_poses.parquet",
    "calibrations_file": "calibrations.json",
}


# Edge index pairs for an 8-corner 3D box (rendered as a wireframe).
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]

# RGB color per Waymo type for the wireframe overlay.
_TYPE_COLOR = {
    1: (0, 255, 0),     # vehicle: green
    2: (0, 0, 255),     # pedestrian: red (BGR)
    3: (255, 255, 0),   # sign: cyan
    4: (255, 0, 255),   # cyclist: magenta
}


def _draw_wireframe(canvas: np.ndarray, corners_px: np.ndarray, color: tuple[int, int, int]):
    pts = corners_px
    for a, b in _BOX_EDGES:
        if not (np.isfinite(pts[a]).all() and np.isfinite(pts[b]).all()):
            continue
        p0 = (int(pts[a, 0]), int(pts[a, 1]))
        p1 = (int(pts[b, 0]), int(pts[b, 1]))
        cv2.line(canvas, p0, p1, color, thickness=2, lineType=cv2.LINE_AA)


def render_wireframe_frames(
    intermediate_dir: Path,
    camera: str,
    out_dir: Path,
) -> int:
    """Render one PNG per frame containing only the 3D-box wireframes, on a black canvas.

    These are the conditioning frames Cosmos consumes. Returns frame count.
    """
    calib = load_calibration(intermediate_dir, camera)
    tracklets = pq.read_table(intermediate_dir / "tracklets.parquet").to_pandas()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Number of frames = max frame_idx in tracklets, but really we want one wireframe
    # per RGB frame — derive count from the RGB dir.
    rgb_dir = intermediate_dir / "frames" / camera
    rgb_frames = sorted(rgb_dir.glob("*.png"))
    if not rgb_frames:
        raise FileNotFoundError(f"no RGB frames in {rgb_dir}")

    by_frame = tracklets.groupby("frame_idx")
    for rgb_path in tqdm(rgb_frames, desc=f"wireframe {camera}"):
        frame_idx = int(rgb_path.stem)
        canvas = np.zeros((calib.height, calib.width, 3), dtype=np.uint8)
        if frame_idx in by_frame.groups:
            for row in by_frame.get_group(frame_idx).itertuples(index=False):
                corners = box_corners_ego(row.cx, row.cy, row.cz,
                                            row.length, row.width, row.height,
                                            row.heading)
                px = project_corners_to_image(corners, calib)
                color = _TYPE_COLOR.get(int(row.type), (200, 200, 200))
                _draw_wireframe(canvas, px, color)
        cv2.imwrite(str(out_dir / f"{frame_idx:05d}.png"), canvas)

    return len(rgb_frames)


def build_rds_hq_dir(
    intermediate_dir: Path,
    out_dir: Path,
    cameras: list[str] | None = None,
) -> Path:
    """Assemble the RDS-HQ-shaped directory Cosmos will consume."""
    cameras = cameras or [config.CANONICAL_CAMERA]
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(intermediate_dir / "manifest.json") as f:
        manifest = json.load(f)

    # 1. Copy RGB.
    for cam in cameras:
        src = intermediate_dir / "frames" / cam
        dst = out_dir / DEFAULT_LAYOUT["rgb_dir"].format(camera=cam)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    # 2. Render wireframe conditioning per camera.
    for cam in cameras:
        wf_dir = out_dir / DEFAULT_LAYOUT["wireframe_dir"].format(camera=cam)
        render_wireframe_frames(intermediate_dir, cam, wf_dir)

    # 3. Copy labels + calibrations.
    (out_dir / "labels").mkdir(exist_ok=True)
    shutil.copy2(intermediate_dir / "tracklets.parquet",
                 out_dir / DEFAULT_LAYOUT["tracklets_file"])
    shutil.copy2(intermediate_dir / "ego_poses.parquet",
                 out_dir / DEFAULT_LAYOUT["ego_poses_file"])
    shutil.copy2(intermediate_dir / "calibrations.json",
                 out_dir / DEFAULT_LAYOUT["calibrations_file"])

    # 4. DreamLoop-controlled manifest so cosmos_runner can read paths back without
    # re-parsing the directory tree.
    rds_manifest = {
        "source_segment": manifest.get("segment_name"),
        "num_frames": manifest.get("num_frames"),
        "frame_rate_hz": manifest.get("frame_rate_hz"),
        "cameras": cameras,
        "layout": DEFAULT_LAYOUT,
    }
    with open(out_dir / "_rds_hq_manifest.json", "w") as f:
        json.dump(rds_manifest, f, indent=2)

    # 5. Carry forward the perturbation audit if present (so a downstream YOLO
    # eval knows exactly what was changed).
    pert = intermediate_dir / "perturbation.json"
    if pert.exists():
        shutil.copy2(pert, out_dir / "perturbation.json")

    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("intermediate_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cameras", type=str, default=config.CANONICAL_CAMERA)
    args = ap.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    out = build_rds_hq_dir(args.intermediate_dir, args.out, cameras)
    print(f"wrote RDS-HQ dir at {out}")


if __name__ == "__main__":
    main()

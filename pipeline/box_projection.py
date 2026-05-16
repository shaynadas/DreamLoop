"""Project 3D ego-frame bounding boxes into 2D image coordinates.

Used by:
  - yolo_eval.py to build ground-truth 2D boxes from tracklets for IoU matching
  - rds_writer.py to render wireframe conditioning overlays
  - perturb verification (sanity-check that injected boxes land on-image)

Waymo conventions:
  - Ego (vehicle) frame: +x forward, +y left, +z up
  - Camera extrinsic = vehicle→camera transform (4x4)
  - Waymo camera frame: +x forward (out of the lens), +y left, +z up
    To go to standard pinhole (+z forward, +x right, +y down), apply the
    axis swap below.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from schema import CameraCalibration


# Waymo camera frame -> OpenCV pinhole frame.
# Waymo: x forward, y left, z up.  OpenCV: x right, y down, z forward.
_WAYMO_CAM_TO_OPENCV = np.array([
    [0, -1, 0, 0],
    [0,  0, -1, 0],
    [1,  0, 0, 0],
    [0,  0, 0, 1],
], dtype=np.float64)


def load_calibration(intermediate_dir: Path, camera: str) -> CameraCalibration:
    with open(intermediate_dir / "calibrations.json") as f:
        all_calibs = json.load(f)
    if camera not in all_calibs:
        raise KeyError(f"camera {camera!r} not in calibrations; have {list(all_calibs)}")
    c = all_calibs[camera]
    return CameraCalibration(
        name=camera,
        width=c["width"],
        height=c["height"],
        intrinsic=np.array(c["intrinsic"], dtype=np.float64),
        extrinsic=np.array(c["extrinsic"], dtype=np.float64),
    )


def box_corners_ego(cx, cy, cz, length, width, height, heading) -> np.ndarray:
    """Return 8x3 array of 3D box corner positions in ego frame."""
    # Corners in box-local frame.
    dx, dy, dz = length / 2, width / 2, height / 2
    local = np.array([
        [ dx,  dy, -dz], [ dx, -dy, -dz], [-dx, -dy, -dz], [-dx,  dy, -dz],
        [ dx,  dy,  dz], [ dx, -dy,  dz], [-dx, -dy,  dz], [-dx,  dy,  dz],
    ], dtype=np.float64)
    # Rotate by heading around +z.
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    rotated = local @ R.T
    return rotated + np.array([cx, cy, cz], dtype=np.float64)


def project_corners_to_image(corners_ego: np.ndarray,
                              calib: CameraCalibration) -> np.ndarray:
    """Project Nx3 ego-frame points to Nx2 image pixels. Returns NaN for points behind camera."""
    N = corners_ego.shape[0]
    homo = np.hstack([corners_ego, np.ones((N, 1))])  # Nx4

    # ego -> waymo camera frame
    cam_w = (calib.extrinsic @ homo.T).T  # Nx4 in Waymo camera frame
    # waymo camera -> opencv camera
    cam = (_WAYMO_CAM_TO_OPENCV @ cam_w.T).T  # Nx4

    xyz = cam[:, :3]
    # Points behind camera (z <= 0) → NaN.
    behind = xyz[:, 2] <= 1e-3
    K = calib.K()
    proj = (K @ xyz.T).T  # Nx3
    px = proj[:, :2] / proj[:, 2:3]
    px[behind] = np.nan
    return px


def box_to_2d_aabb(corners_px: np.ndarray, img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    """Take 8 projected corners → axis-aligned bbox (x1, y1, x2, y2) clipped to image.

    Returns None if box is fully off-screen or fully behind camera.
    """
    finite = corners_px[np.isfinite(corners_px).all(axis=1)]
    if len(finite) == 0:
        return None
    x1, y1 = finite.min(axis=0)
    x2, y2 = finite.max(axis=0)
    x1 = max(0.0, min(img_w - 1.0, x1))
    x2 = max(0.0, min(img_w - 1.0, x2))
    y1 = max(0.0, min(img_h - 1.0, y1))
    y2 = max(0.0, min(img_h - 1.0, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (float(x1), float(y1), float(x2), float(y2))


def project_tracklets_to_2d(
    tracklets: pd.DataFrame,
    calib: CameraCalibration,
) -> pd.DataFrame:
    """Per-row 2D AABB for each tracklet entry. Drops rows that project off-screen."""
    out_rows = []
    for row in tracklets.itertuples(index=False):
        corners = box_corners_ego(row.cx, row.cy, row.cz,
                                    row.length, row.width, row.height,
                                    row.heading)
        px = project_corners_to_image(corners, calib)
        aabb = box_to_2d_aabb(px, calib.width, calib.height)
        if aabb is None:
            continue
        x1, y1, x2, y2 = aabb
        out_rows.append({
            "object_id": row.object_id,
            "frame_idx": row.frame_idx,
            "type": row.type,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "perturbed": row.perturbed,
        })
    return pd.DataFrame(out_rows)

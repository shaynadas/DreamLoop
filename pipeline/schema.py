"""Shared dataclasses + parquet schemas.

Single source of truth for what flows between waymo_loader → tracklet_perturb → rds_writer.
Anything that crosses a script boundary should be expressible here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import pyarrow as pa


class WaymoLabelType(IntEnum):
    UNKNOWN = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    SIGN = 3
    CYCLIST = 4


@dataclass
class BBox3D:
    """3D bounding box in ego-vehicle frame at one timestep.

    Right-handed coordinate system, Waymo convention:
      +x forward, +y left, +z up. heading is yaw around +z (radians).
    """
    cx: float
    cy: float
    cz: float
    length: float  # extent along heading
    width: float   # extent perpendicular to heading, in xy plane
    height: float  # extent along z
    heading: float

    def as_array(self) -> np.ndarray:
        return np.array([self.cx, self.cy, self.cz,
                         self.length, self.width, self.height,
                         self.heading], dtype=np.float64)


# Tracklet parquet schema. One row per (object_id, frame_idx).
TRACKLET_SCHEMA = pa.schema([
    ("object_id", pa.string()),       # stable Waymo track ID within a segment
    ("frame_idx", pa.int32()),
    ("type", pa.int32()),             # WaymoLabelType
    ("cx", pa.float64()),             # 3D box center, ego frame
    ("cy", pa.float64()),
    ("cz", pa.float64()),
    ("length", pa.float64()),
    ("width", pa.float64()),
    ("height", pa.float64()),
    ("heading", pa.float64()),        # radians, yaw
    ("speed_x", pa.float64()),        # ego-frame velocity from Waymo, nullable
    ("speed_y", pa.float64()),
    ("num_lidar_points", pa.int32()), # confidence proxy; rows with 0 are weak
    ("perturbed", pa.bool_()),        # set by tracklet_perturb
    ("perturb_kind", pa.string()),    # null if not perturbed
])


# Ego pose parquet schema. One row per frame_idx.
EGO_POSE_SCHEMA = pa.schema([
    ("frame_idx", pa.int32()),
    ("timestamp_us", pa.int64()),
    # 4x4 world->ego transform, row-major flattened.
    ("transform", pa.list_(pa.float64(), 16)),
])


@dataclass
class CameraCalibration:
    """Per-camera intrinsics + extrinsics in Waymo convention."""
    name: str              # e.g. "FRONT"
    width: int
    height: int
    # Pinhole intrinsics: f_x, f_y, c_x, c_y, k1, k2, p1, p2, k3
    intrinsic: np.ndarray  # shape (9,)
    # 4x4 vehicle->camera transform
    extrinsic: np.ndarray  # shape (4, 4)

    def K(self) -> np.ndarray:
        """3x3 pinhole intrinsic matrix (no distortion)."""
        fx, fy, cx, cy = self.intrinsic[:4]
        return np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype=np.float64)


@dataclass
class SegmentManifest:
    """Top-level metadata for one extracted segment."""
    segment_name: str
    num_frames: int
    frame_rate_hz: float
    cameras: list[str]
    intermediate_dir: str  # path to the on-disk layout

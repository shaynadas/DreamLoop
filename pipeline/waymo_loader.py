"""Waymo Open Dataset → DreamLoop intermediate format.

Reads one .tfrecord segment, writes a clean on-disk layout:

    <intermediate_dir>/
        manifest.json
        calibrations.json          (per-camera intrinsics + extrinsics)
        ego_poses.parquet          (one row per frame)
        tracklets.parquet          (one row per (object_id, frame_idx))
        frames/<CAMERA_NAME>/00000.png ... NNNNN.png

Usage:
    python waymo_loader.py /path/to/segment.tfrecord [--out DIR] [--cameras FRONT,FRONT_LEFT,...]

This script is heavy on imports (tensorflow + waymo SDK). On the Mac dev machine
it's fine for parsing; the real downloads + decode loop should run on the GX10.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

import config
from schema import (
    CameraCalibration,
    EGO_POSE_SCHEMA,
    SegmentManifest,
    TRACKLET_SCHEMA,
    WaymoLabelType,
)

# Heavy imports — deferred so the module is importable for typing/docs without TF.
def _load_waymo_proto():
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2
    return tf, dataset_pb2


def _camera_name_str(name_enum: int) -> str:
    # Mirrors waymo_open_dataset.dataset_pb2.CameraName.Name values.
    return {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT",
            4: "SIDE_LEFT", 5: "SIDE_RIGHT"}.get(name_enum, f"CAM_{name_enum}")


def _extract_calibrations(frame) -> dict[str, CameraCalibration]:
    out = {}
    for cam_calib in frame.context.camera_calibrations:
        name = _camera_name_str(cam_calib.name)
        out[name] = CameraCalibration(
            name=name,
            width=cam_calib.width,
            height=cam_calib.height,
            intrinsic=np.array(cam_calib.intrinsic, dtype=np.float64),
            extrinsic=np.array(cam_calib.extrinsic.transform, dtype=np.float64).reshape(4, 4),
        )
    return out


def _calibrations_to_json(calibs: dict[str, CameraCalibration]) -> dict:
    return {
        name: {
            "width": c.width,
            "height": c.height,
            "intrinsic": c.intrinsic.tolist(),
            "extrinsic": c.extrinsic.tolist(),
        }
        for name, c in calibs.items()
    }


def extract_segment(
    tfrecord_path: Path,
    out_dir: Path,
    cameras: list[str] | None = None,
    max_frames: int | None = None,
) -> SegmentManifest:
    tf, dataset_pb2 = _load_waymo_proto()
    cameras = cameras or [config.CANONICAL_CAMERA]
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam in cameras:
        (out_dir / "frames" / cam).mkdir(parents=True, exist_ok=True)

    tracklet_rows: list[dict] = []
    ego_rows: list[dict] = []
    calibrations: dict[str, CameraCalibration] = {}
    timestamps_us: list[int] = []

    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")
    frame_idx = 0
    for raw in tqdm(dataset, desc=f"decoding {tfrecord_path.name}"):
        if max_frames is not None and frame_idx >= max_frames:
            break
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(raw.numpy()))

        if frame_idx == 0:
            calibrations = _extract_calibrations(frame)

        # Cameras → PNGs.
        for cam_image in frame.images:
            cam_name = _camera_name_str(cam_image.name)
            if cam_name not in cameras:
                continue
            img_bytes = cam_image.image
            # Waymo stores JPEGs; decode and rewrite as PNG so downstream tools (Cosmos
            # conditioning, OpenCV) don't deal with mixed formats.
            img = tf.io.decode_jpeg(img_bytes).numpy()
            Image.fromarray(img).save(out_dir / "frames" / cam_name / f"{frame_idx:05d}.png")

        # Ego pose.
        pose = np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)
        ego_rows.append({
            "frame_idx": frame_idx,
            "timestamp_us": frame.timestamp_micros,
            "transform": pose.flatten().tolist(),
        })
        timestamps_us.append(frame.timestamp_micros)

        # 3D labels → tracklet rows.
        for lbl in frame.laser_labels:
            box = lbl.box
            speed_x = lbl.metadata.speed_x if lbl.HasField("metadata") else float("nan")
            speed_y = lbl.metadata.speed_y if lbl.HasField("metadata") else float("nan")
            tracklet_rows.append({
                "object_id": lbl.id,
                "frame_idx": frame_idx,
                "type": int(lbl.type),
                "cx": box.center_x,
                "cy": box.center_y,
                "cz": box.center_z,
                "length": box.length,
                "width": box.width,
                "height": box.height,
                "heading": box.heading,
                "speed_x": speed_x,
                "speed_y": speed_y,
                "num_lidar_points": lbl.num_lidar_points_in_box,
                "perturbed": False,
                "perturb_kind": None,
            })

        frame_idx += 1

    # Write parquets.
    tracklets_df = pd.DataFrame(tracklet_rows)
    tracklets_table = pa.Table.from_pandas(tracklets_df, schema=TRACKLET_SCHEMA, preserve_index=False)
    pq.write_table(tracklets_table, out_dir / "tracklets.parquet")

    ego_df = pd.DataFrame(ego_rows)
    ego_table = pa.Table.from_pandas(ego_df, schema=EGO_POSE_SCHEMA, preserve_index=False)
    pq.write_table(ego_table, out_dir / "ego_poses.parquet")

    # Calibrations JSON.
    with open(out_dir / "calibrations.json", "w") as f:
        json.dump(_calibrations_to_json(calibrations), f, indent=2)

    # Frame rate from timestamps (Waymo is nominally 10Hz).
    if len(timestamps_us) >= 2:
        dt_s = (timestamps_us[-1] - timestamps_us[0]) / 1e6 / (len(timestamps_us) - 1)
        hz = 1.0 / dt_s if dt_s > 0 else 10.0
    else:
        hz = 10.0

    manifest = SegmentManifest(
        segment_name=tfrecord_path.stem,
        num_frames=frame_idx,
        frame_rate_hz=hz,
        cameras=cameras,
        intermediate_dir=str(out_dir),
    )
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest.__dict__, f, indent=2)

    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tfrecord", type=Path, help="Path to a single Waymo .tfrecord segment file")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"Output dir (default: {config.WAYMO_INTERMEDIATE}/<segment_name>)")
    ap.add_argument("--cameras", type=str, default=config.CANONICAL_CAMERA,
                    help="Comma-separated camera names to extract (default: FRONT)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap frames extracted (useful for quick iteration)")
    args = ap.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    out_dir = args.out or (config.WAYMO_INTERMEDIATE / args.tfrecord.stem)
    manifest = extract_segment(args.tfrecord, out_dir, cameras, args.max_frames)
    print(json.dumps(manifest.__dict__, indent=2))


if __name__ == "__main__":
    main()

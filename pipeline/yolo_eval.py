"""Honest YOLO evaluation against tracklet-derived ground truth.

Workflow:
  1. Load a frame directory (PNGs) — typically extracted from a Cosmos MP4 or
     from the original Waymo intermediate dir.
  2. Load tracklets.parquet + calibrations.json from the intermediate dir that
     describes the *ground truth* for that sequence.
  3. Project 3D boxes → 2D AABBs per frame.
  4. Run YOLOv8 on each frame.
  5. Match predictions ↔ ground truth by IoU + class compatibility.
  6. Report mAP@0.5, per-class AP, and per-frame detection rate.

We treat this as a perception-degradation test, not a leaderboard run. The
numbers are meaningful for *comparisons* between sequences (sunny → blizzard),
not as absolute statements about YOLO quality.

Usage:
    # PNG dir mode
    python yolo_eval.py \
        --frames-dir <PNG dir>                  \
        --intermediate-dir <waymo_loader out>    \
        --camera FRONT                            \
        --out report.json

    # MP4 mode (Cosmos outputs MP4) — frames are extracted to a temp dir first.
    python yolo_eval.py \
        --mp4 <path/to/cosmos_video.mp4>          \
        --intermediate-dir <waymo_loader out>     \
        --out report.json

Note on ground truth:
    Pass the *perturbed* intermediate dir when evaluating Cosmos output of a
    perturbed scene — the GT must reflect the trajectories Cosmos was conditioned
    on, not the original Waymo trajectories.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

import config
from box_projection import load_calibration, project_tracklets_to_2d


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def _waymo_type_for_coco_class(coco_name: str) -> int | None:
    return config.COCO_TO_WAYMO.get(coco_name)


def _run_yolo(frames: list[Path], weights: Path | str, conf: float = 0.25):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    # Run sequentially so we can stream per-frame predictions out.
    for f in tqdm(frames, desc="yolo"):
        results = model.predict(source=str(f), conf=conf, verbose=False)
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
        cls_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros((0,), dtype=int)
        names = [r.names[i] for i in cls_ids]
        yield f, boxes, scores, names


def _ap_per_class(matches_by_class: dict[int, list[tuple[float, int]]],
                  num_gt_by_class: dict[int, int]) -> dict[int, float]:
    """11-point interpolated AP per class. matches: list of (score, tp_flag)."""
    out = {}
    for cls, recs in matches_by_class.items():
        n_gt = num_gt_by_class.get(cls, 0)
        if n_gt == 0:
            continue
        recs_sorted = sorted(recs, key=lambda r: -r[0])
        tp = np.array([r[1] for r in recs_sorted], dtype=np.float64)
        fp = 1.0 - tp
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)
        recall = tp_cum / (n_gt + 1e-9)
        # 11-point interpolation.
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            mask = recall >= r
            p = precision[mask].max() if mask.any() else 0.0
            ap += p / 11.0
        out[cls] = float(ap)
    return out


def evaluate(
    frames_dir: Path,
    intermediate_dir: Path,
    camera: str = config.CANONICAL_CAMERA,
    weights: Path | str | None = None,
    iou_threshold: float = 0.5,
    conf: float = 0.25,
) -> dict:
    weights = weights or config.YOLO_WEIGHTS
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no PNGs in {frames_dir}")

    calib = load_calibration(intermediate_dir, camera)
    tracklets = pq.read_table(intermediate_dir / "tracklets.parquet").to_pandas()
    gt_2d = project_tracklets_to_2d(tracklets, calib)
    gt_by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in gt_2d.itertuples(index=False):
        gt_by_frame[int(row.frame_idx)].append({
            "type": int(row.type),
            "box": np.array([row.x1, row.y1, row.x2, row.y2], dtype=np.float64),
            "matched": False,
        })

    num_gt_by_class: dict[int, int] = defaultdict(int)
    for entries in gt_by_frame.values():
        for e in entries:
            num_gt_by_class[e["type"]] += 1

    matches_by_class: dict[int, list[tuple[float, int]]] = defaultdict(list)
    per_frame: list[dict] = []

    for f, boxes, scores, names in _run_yolo(frames, weights, conf):
        frame_idx = int(f.stem)
        gts = gt_by_frame.get(frame_idx, [])
        # Reset matched flags for this frame.
        for g in gts:
            g["matched"] = False
        n_tp = 0
        # Sort predictions high→low confidence and greedily match.
        order = np.argsort(-scores)
        for i in order:
            waymo_type = _waymo_type_for_coco_class(names[i])
            if waymo_type is None:
                continue
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if g["matched"] or g["type"] != waymo_type:
                    continue
                iou = iou_xyxy(boxes[i], g["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            tp = 1 if (best_iou >= iou_threshold and best_j >= 0) else 0
            if tp:
                gts[best_j]["matched"] = True
                n_tp += 1
            matches_by_class[waymo_type].append((float(scores[i]), tp))

        per_frame.append({
            "frame_idx": frame_idx,
            "n_gt": len(gts),
            "n_tp": n_tp,
            "n_pred": int(len(boxes)),
            "detection_rate": (n_tp / len(gts)) if gts else None,
        })

    ap_per_class = _ap_per_class(matches_by_class, num_gt_by_class)
    map50 = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0

    return {
        "frames_dir": str(frames_dir),
        "intermediate_dir": str(intermediate_dir),
        "camera": camera,
        "iou_threshold": iou_threshold,
        "num_frames": len(frames),
        "num_gt_total": sum(num_gt_by_class.values()),
        "num_gt_by_class": dict(num_gt_by_class),
        "ap_per_class": ap_per_class,
        "map50": map50,
        "per_frame": per_frame,
    }


def extract_frames_from_mp4(mp4_path: Path, out_dir: Path) -> Path:
    """Dump every frame of an MP4 as a zero-padded PNG. Returns out_dir."""
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {mp4_path}")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # cv2 reads BGR; the rest of the pipeline assumes RGB-on-disk PNGs.
        # Save as-is (BGR) since YOLO + cv2.imread both round-trip consistently
        # in BGR — but ground-truth projection doesn't care about color.
        cv2.imwrite(str(out_dir / f"{i:05d}.png"), frame)
        i += 1
    cap.release()
    if i == 0:
        raise RuntimeError(f"{mp4_path} produced 0 frames — check codec")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames-dir", type=Path,
                     help="Directory of PNG frames to evaluate")
    src.add_argument("--mp4", type=Path,
                     help="MP4 to evaluate (frames extracted to a sibling tmp dir)")
    ap.add_argument("--intermediate-dir", type=Path, required=True,
                    help="waymo_loader output dir for ground-truth tracklets+calibrations. "
                         "Pass the PERTURBED intermediate dir when evaluating Cosmos output "
                         "of a perturbed scene.")
    ap.add_argument("--camera", type=str, default=config.CANONICAL_CAMERA)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.mp4:
        frames_dir = args.mp4.with_suffix("") / "frames"
        extract_frames_from_mp4(args.mp4, frames_dir)
    else:
        frames_dir = args.frames_dir

    report = evaluate(frames_dir, args.intermediate_dir,
                       camera=args.camera, weights=args.weights,
                       iou_threshold=args.iou, conf=args.conf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"mAP@{args.iou}: {report['map50']:.3f}  (n_frames={report['num_frames']}, "
          f"n_gt={report['num_gt_total']})")
    print(f"report → {args.out}")


if __name__ == "__main__":
    main()

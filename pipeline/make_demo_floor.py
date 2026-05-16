"""DreamLoop demo floor: one-shot generator for the four files Kayla's UI reads.

No Cosmos. No Helios. No GX10. No fine-tuned weights. Just:

    Nexar dashcam .mp4
       ├── OpenCV blizzard filter ───────► outputs/dreamloop_web.mp4
       ├── YOLOv8n on clean      ────────► outputs/baseline_yolo_web.mp4
       └── YOLOv8n on filtered   ────────► outputs/finetuned_yolo_web.mp4

The four output files plus outputs/metrics.json are exactly what `app.py`
on Kayla's branch expects. Drop this in, run once, the UI lights up.

What the numbers actually mean (tell judges this honestly):
    clean detections from off-the-shelf YOLO are treated as pseudo-ground-truth.
    "baseline_map"  = F1 of YOLO detections on the filtered video against
                      the clean-frame detections (i.e. how much perception
                      is retained under the weather perturbation).
    "finetuned_map" = projected post-fine-tune retention (midpoint of
                      baseline and clean); Shyam's actual fine-tune
                      pipeline overwrites this when his weights land.
    The provenance block in metrics.json flags both of these.

Usage (from DreamLoop/ repo root):

    python pipeline/make_demo_floor.py
    # or with explicit paths
    python pipeline/make_demo_floor.py \
        --input nexar_videos/train/negative/01040.mp4 \
        --out   outputs \
        --intensity 0.65
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# COCO class ids we care about for road-scene perception.
COCO_VEHICLE_CLASSES = {0, 1, 2, 3, 5, 7}  # person, bicycle, car, motorcycle, bus, truck

# BGR colors for box overlays.
_CLASS_COLORS = {
    0: (0, 200, 255),    # person — orange
    1: (255, 100, 0),    # bicycle — blue
    2: (0, 255, 0),      # car — green
    3: (255, 100, 0),    # motorcycle — blue
    5: (0, 255, 0),      # bus — green
    7: (0, 255, 0),      # truck — green
}


# ── Weather filter ─────────────────────────────────────────────────────────

def blizzard_filter(frame_bgr: np.ndarray, intensity: float = 0.6,
                    seed: int | None = None) -> np.ndarray:
    """Apply a blizzard-like overlay to one BGR frame.

    intensity: 0 = no effect, 1 = whiteout. 0.6 is a "judges still recognize
    the scene but YOLO struggles" sweet spot.
    """
    h, w = frame_bgr.shape[:2]
    rng = np.random.default_rng(seed)

    # 1. Mild contrast crush + blue tint (cold light).
    out = frame_bgr.astype(np.float32)
    out = out * (1.0 - intensity * 0.25) + 70.0 * intensity
    out[..., 0] *= (1.0 + intensity * 0.12)   # B up
    out[..., 2] *= (1.0 - intensity * 0.08)   # R down
    out = np.clip(out, 0, 255).astype(np.uint8)

    # 2. Snowflakes — vectorized for speed.
    num_flakes = int(intensity * 700 * (h * w) / (1920 * 1080))
    if num_flakes > 0:
        xs = rng.integers(0, w, size=num_flakes)
        ys = rng.integers(0, h, size=num_flakes)
        sizes = rng.integers(1, max(2, int(3 * intensity) + 1), size=num_flakes)
        for x, y, s in zip(xs, ys, sizes):
            cv2.circle(out, (int(x), int(y)), int(s), (255, 255, 255), -1)

    # 3. Fog overlay (additive white blend).
    fog_alpha = intensity * 0.18
    out = cv2.addWeighted(out, 1.0 - fog_alpha,
                          np.full_like(out, 255), fog_alpha, 0)

    # 4. Mild blur — "low visibility".
    blur_k = 3 if intensity < 0.7 else 5
    out = cv2.GaussianBlur(out, (blur_k, blur_k), 0)

    return out


# ── Drawing ────────────────────────────────────────────────────────────────

def draw_boxes(frame_bgr: np.ndarray, boxes: np.ndarray, scores: np.ndarray,
               classes: np.ndarray, names: dict) -> np.ndarray:
    out = frame_bgr.copy()
    for xyxy, conf, cls in zip(boxes, scores, classes):
        cls_int = int(cls)
        if cls_int not in COCO_VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        color = _CLASS_COLORS.get(cls_int, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{names[cls_int]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# ── Metric ─────────────────────────────────────────────────────────────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def retention_f1(clean_boxes: np.ndarray, weather_boxes: np.ndarray,
                 iou_threshold: float = 0.5) -> float:
    """Treat clean detections as ground truth, compute F1 of weather detections."""
    n_c, n_w = len(clean_boxes), len(weather_boxes)
    if n_c == 0:
        return 1.0 if n_w == 0 else 0.0
    if n_w == 0:
        return 0.0

    matched = set()
    tp = 0
    # Greedy match: each weather box claims its best unused clean box.
    for w_box in weather_boxes:
        best_iou, best_j = 0.0, -1
        for j in range(n_c):
            if j in matched:
                continue
            iou = _iou(w_box, clean_boxes[j])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold:
            matched.add(best_j)
            tp += 1

    fp = n_w - tp
    fn = n_c - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ── Pipeline ───────────────────────────────────────────────────────────────

def process_video(input_path: Path, output_dir: Path,
                   weights: str = "yolov8n.pt",
                   intensity: float = 0.6,
                   limit_frames: int | None = None,
                   conf: float = 0.25,
                   iou_threshold: float = 0.5) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input video not found: {input_path}\n"
            f"  - Kayla's branch ships a clip at "
            f"nexar_videos/train/negative/01040.mp4\n"
            f"  - `git checkout origin/kayla -- nexar_videos/` will pull it"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Defer the import so the script is at least parse-checkable without YOLO.
    from ultralytics import YOLO
    print(f"[floor] loading YOLO weights: {weights}")
    model = YOLO(weights)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if limit_frames is not None:
        total = min(total, limit_frames)
    print(f"[floor] {input_path.name}: {w}x{h} @ {fps:.1f}fps, {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # browser-playable; ffmpeg can re-mux to h264 if needed
    writers = {
        "dream":   cv2.VideoWriter(str(output_dir / "dreamloop_web.mp4"),     fourcc, fps, (w, h)),
        "base":    cv2.VideoWriter(str(output_dir / "baseline_yolo_web.mp4"),  fourcc, fps, (w, h)),
        "ft":      cv2.VideoWriter(str(output_dir / "finetuned_yolo_web.mp4"), fourcc, fps, (w, h)),
    }
    for name, wr in writers.items():
        if not wr.isOpened():
            raise RuntimeError(f"could not open VideoWriter for {name}")

    per_frame_retention: list[float] = []
    avg_clean_dets, avg_weather_dets = 0, 0
    t0 = time.time()
    n_frames = 0

    pbar = tqdm(total=total, desc="frames", unit="f")
    while True:
        ok, frame = cap.read()
        if not ok or (limit_frames is not None and n_frames >= limit_frames):
            break

        weather_frame = blizzard_filter(frame, intensity=intensity, seed=n_frames)

        # YOLO predict expects RGB; cv2 gives BGR. ultralytics handles this internally
        # when source is a numpy array, but we pass BGR explicitly via `source=frame`
        # which is what its training data assumes (it converts internally).
        clean_r = model.predict(source=frame, conf=conf, verbose=False)[0]
        weath_r = model.predict(source=weather_frame, conf=conf, verbose=False)[0]

        clean_boxes  = (clean_r.boxes.xyxy.cpu().numpy() if clean_r.boxes is not None else np.zeros((0, 4)))
        clean_scores = (clean_r.boxes.conf.cpu().numpy() if clean_r.boxes is not None else np.zeros((0,)))
        clean_cls    = (clean_r.boxes.cls.cpu().numpy()  if clean_r.boxes is not None else np.zeros((0,), dtype=int)).astype(int)

        weath_boxes  = (weath_r.boxes.xyxy.cpu().numpy() if weath_r.boxes is not None else np.zeros((0, 4)))
        weath_scores = (weath_r.boxes.conf.cpu().numpy() if weath_r.boxes is not None else np.zeros((0,)))
        weath_cls    = (weath_r.boxes.cls.cpu().numpy()  if weath_r.boxes is not None else np.zeros((0,), dtype=int)).astype(int)

        # Keep only vehicle/person classes for fair comparison.
        clean_mask = np.array([c in COCO_VEHICLE_CLASSES for c in clean_cls], dtype=bool)
        weath_mask = np.array([c in COCO_VEHICLE_CLASSES for c in weath_cls], dtype=bool)
        cb, cw = clean_boxes[clean_mask], weath_boxes[weath_mask]

        per_frame_retention.append(retention_f1(cb, cw, iou_threshold=iou_threshold))
        avg_clean_dets   += int(clean_mask.sum())
        avg_weather_dets += int(weath_mask.sum())

        writers["dream"].write(weather_frame)
        writers["base"].write(draw_boxes(frame, clean_boxes, clean_scores, clean_cls, clean_r.names))
        writers["ft"].write(draw_boxes(weather_frame, weath_boxes, weath_scores, weath_cls, weath_r.names))

        n_frames += 1
        pbar.update(1)

    cap.release()
    for wr in writers.values():
        wr.release()
    pbar.close()
    elapsed = time.time() - t0

    if n_frames == 0:
        raise RuntimeError("processed 0 frames — check the input video")

    baseline_map = float(np.mean(per_frame_retention))
    clear_weather_map = 1.0  # clean detections are pseudo-GT by construction
    finetuned_map_projected = round((baseline_map + clear_weather_map) / 2, 3)

    metrics = {
        "condition": "OpenCV blizzard filter (demo floor)",
        "baseline_map":      round(baseline_map, 3),
        "finetuned_map":     finetuned_map_projected,
        "clear_weather_map": clear_weather_map,
        "latency_ms":        int(round(1000.0 * elapsed / max(n_frames, 1) / 2)),  # per-frame, single pass
        "_provenance": {
            "renderer":          "opencv_blizzard_floor",
            "yolo_weights":      str(weights),
            "input_video":       str(input_path),
            "frames_processed":  n_frames,
            "blizzard_intensity": intensity,
            "iou_threshold":     iou_threshold,
            "avg_clean_dets_per_frame":   round(avg_clean_dets / n_frames, 2),
            "avg_weather_dets_per_frame": round(avg_weather_dets / n_frames, 2),
            "is_floor":              True,
            "finetuned_is_projected": True,
            "elapsed_seconds":   round(elapsed, 1),
        },
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[floor] done in {elapsed:.1f}s")
    print(f"[floor] wrote:")
    for k, name in [("dream", "dreamloop_web.mp4"),
                    ("base", "baseline_yolo_web.mp4"),
                    ("ft", "finetuned_yolo_web.mp4")]:
        print(f"          {output_dir / name}")
    print(f"          {metrics_path}")
    print(f"[floor] baseline_map (retention under blizzard): {baseline_map:.3f}")
    print(f"[floor] finetuned_map (projected):                {finetuned_map_projected:.3f}")

    return metrics


def main():
    ap = argparse.ArgumentParser(description="DreamLoop demo floor")
    ap.add_argument("--input", type=Path,
                    default=Path("nexar_videos/train/negative/01040.mp4"),
                    help="Path to baseline dashcam video (default: Kayla's Nexar clip)")
    ap.add_argument("--out", type=Path, default=Path("outputs"),
                    help="Output directory (must match Kayla's UI expectations)")
    ap.add_argument("--weights", type=str, default="yolov8n.pt",
                    help="YOLO weights (ultralytics auto-downloads on first run)")
    ap.add_argument("--intensity", type=float, default=0.6,
                    help="Blizzard intensity 0..1. Higher = worse weather. "
                         "0.6 is the sweet spot for visible YOLO degradation.")
    ap.add_argument("--limit-frames", type=int, default=None,
                    help="Cap frames for fast iteration")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="YOLO confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU threshold for retention matching")
    args = ap.parse_args()

    try:
        process_video(
            input_path=args.input,
            output_dir=args.out,
            weights=args.weights,
            intensity=args.intensity,
            limit_frames=args.limit_frames,
            conf=args.conf,
            iou_threshold=args.iou,
        )
    except FileNotFoundError as e:
        print(f"\n[floor] {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

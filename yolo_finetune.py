"""
yolo_finetune.py — Fine-tune YOLO on synthetic frames + boxes.json

Real training (TODO — swap placeholder block for Ultralytics train):
    # 1. Extract frames: ffmpeg -i outputs/<clip>/dreamloop_web.mp4 synthetic/<clip>/frames/%05d.jpg
    # 2. Label frames → synthetic/<clip>/boxes.json  (or use boxes.example.json schema)
    # 3. prepare_yolo_dataset.py  → datasets/person_blizzard/
    # 4. model = YOLO("yolov8n.pt"); model.train(data="...", time=120, ...)
    # 5. python yolo_eval.py --model runs/.../best.pt --output outputs/<clip>/finetuned_yolo.mp4 --web

Placeholder (UI demo — no GPU / no real train):
    python yolo_finetune.py --clip-id positive_01 --placeholder

    # After panels 2–3 exist:
    python render_clip.py --clip-id positive_01 --skip-sim --skip-baseline --placeholder-finetune
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parent

# BGR
C_BOX = (80, 220, 80)
C_HIT_BANNER = (40, 40, 40)
C_HIT_TEXT = (80, 220, 80)


def load_boxes_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Could not read {path}: {exc}", file=sys.stderr)
        return {}


def resolve_boxes_path(clip_id: str, boxes_arg: str | None) -> Path:
    if boxes_arg:
        p = Path(boxes_arg)
        return p if p.is_absolute() else ROOT / p
    for candidate in (
        ROOT / "synthetic" / clip_id / "boxes.json",
        ROOT / "boxes.json",
        ROOT / "boxes.example.json",
    ):
        if candidate.is_file():
            return candidate
    return ROOT / "boxes.example.json"


def default_pedestrian_box(w: int, h: int) -> tuple[int, int, int, int, float]:
    """Center-lower third — typical Nexar pedestrian region."""
    x1 = int(w * 0.44)
    y1 = int(h * 0.38)
    x2 = int(w * 0.56)
    y2 = int(h * 0.72)
    return (x1, y1, x2, y2, 0.91)


def box_from_annotations(
    data: dict[str, Any], frame_idx: int, w: int, h: int
) -> tuple[int, int, int, int, float]:
    annotations = data.get("annotations") or []
    if not annotations:
        return default_pedestrian_box(w, h)

    # Prefer exact frame_index match, else first annotation.
    chosen = None
    for ann in annotations:
        if ann.get("frame_index") == frame_idx:
            chosen = ann
            break
    if chosen is None:
        chosen = annotations[0]

    boxes = chosen.get("boxes") or []
    if not boxes:
        return default_pedestrian_box(w, h)

    b = boxes[0]
    ref_w = int(data.get("reference_width") or w)
    ref_h = int(data.get("reference_height") or h)
    sx = w / ref_w if ref_w else 1.0
    sy = h / ref_h if ref_h else 1.0
    x1 = int(b["x1"] * sx)
    y1 = int(b["y1"] * sy)
    x2 = int(b["x2"] * sx)
    y2 = int(b["y2"] * sy)
    score = float(b.get("score", 0.91))
    return (x1, y1, x2, y2, score)


def pseudo_train_steps(clip_id: str, boxes_path: Path, train_seconds: float) -> None:
    """Print rubric-aligned steps; real train replaces this block."""
    print("\n-- [PLACEHOLDER] Fine-tune pipeline (not running Ultralytics) --")
    print(f"  1. Load synthetic frames  -> synthetic/{clip_id}/frames/  (optional)")
    print(f"  2. Load labels            -> {boxes_path.relative_to(ROOT)}")
    print("  3. Convert boxes.json     -> YOLO labels/*.txt  (prepare_yolo_dataset.py -- TODO)")
    print("  4. YOLO.train(")
    print('       data="datasets/person_blizzard/data.yaml",')
    print(f"       time={int(train_seconds)},  # ~2 minute session")
    print('       model="yolov8n.pt",')
    print("     )  <- SKIPPED in --placeholder mode")
    print("  5. Render finetuned eval  -> finetuned_yolo_web.mp4 + hit screenshot")
    for remaining in (train_seconds, 0, -1):
        if remaining > 0:
            print(f"  ... simulating train wait {remaining:.0f}s ...")
            time.sleep(min(1.0, remaining))
        break
    print("  [OK] Placeholder train complete (no weights written)\n")


def draw_hit_box(frame, box: tuple[int, int, int, int, float], placeholder: bool) -> None:
    x1, y1, x2, y2, score = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_BOX, 2)
    tag = "placeholder" if placeholder else "fine-tuned"
    label = f"person {score:.2f} ({tag})"
    cv2.putText(
        frame, label, (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_BOX, 2, cv2.LINE_AA,
    )
    banner = "PERSON TRACKED (fine-tuned YOLO — PLACEHOLDER)" if placeholder else "PERSON TRACKED (fine-tuned YOLO)"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), C_HIT_BANNER, -1)
    cv2.putText(
        frame, banner, (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_HIT_TEXT, 2, cv2.LINE_AA,
    )


def render_placeholder_eval(
    video_path: Path,
    out_dir: Path,
    boxes_data: dict[str, Any],
    skip_frames: int,
    screenshot: bool,
) -> Path:
    """Write finetuned_yolo.mp4 with synthetic 'hit' boxes from boxes.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_mp4 = out_dir / "finetuned_yolo.mp4"
    web_mp4 = out_dir / "finetuned_yolo_web.mp4"
    hit_png = out_dir / "finetuned_hit_pedestrian.png"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(
        str(raw_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise SystemExit(f"[ERROR] Cannot write: {raw_mp4}")

    saved_shot = False
    frame_idx = 0
    print(f"[INFO] Placeholder eval -> {raw_mp4}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated = frame.copy()
        if frame_idx >= skip_frames:
            box = box_from_annotations(boxes_data, frame_idx, w, h)
            draw_hit_box(annotated, box, placeholder=True)
            if screenshot and not saved_shot:
                cv2.imwrite(str(hit_png), annotated)
                saved_shot = True
                print(f"[INFO] Hit screenshot (placeholder): {hit_png} (frame {frame_idx})")
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    from yolo_eval import try_ffmpeg_h264

    if try_ffmpeg_h264(raw_mp4, web_mp4):
        print(f"[INFO] Browser-ready: {web_mp4}")
    else:
        print("[WARN] ffmpeg missing — copy finetuned_yolo.mp4 or run ffmpeg for Streamlit.")
        if raw_mp4.is_file() and not web_mp4.is_file():
            shutil.copy2(raw_mp4, web_mp4)

    return web_mp4


def write_placeholder_metrics(out_dir: Path) -> None:
    example = ROOT / "metrics.example.json"
    dst = out_dir / "metrics.json"
    if example.is_file():
        shutil.copy2(example, dst)
        data = json.loads(dst.read_text(encoding="utf-8"))
        data["placeholder_metrics"] = True
        data["condition"] = "Snowy blizzard (edge case) — placeholder fine-tune"
        dst.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"[INFO] Metrics (placeholder): {dst}")
    else:
        dst.write_text(
            json.dumps(
                {
                    "baseline_map": 0.42,
                    "finetuned_map": 0.89,
                    "clear_weather_map": 0.70,
                    "latency_ms": 32,
                    "placeholder_metrics": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def run_placeholder(
    clip_id: str,
    video_path: Path | None,
    boxes_path: Path,
    train_seconds: float,
    skip_frames: int,
) -> int:
    out_dir = ROOT / "outputs" / clip_id
    if video_path is None:
        video_path = out_dir / "dreamloop_web.mp4"
    elif not video_path.is_absolute():
        video_path = ROOT / video_path

    if not video_path.is_file():
        print(
            f"[ERROR] Need blizzard clip at {video_path}\n"
            "  Run: python render_clip.py --clip-id "
            f"{clip_id} --video <nexar.mp4>",
            file=sys.stderr,
        )
        return 1

    boxes_data = load_boxes_json(boxes_path)
    if not boxes_data:
        print(f"[WARN] No boxes file; using default pedestrian box.")
    else:
        print(f"[INFO] Labels: {boxes_path}")

    pseudo_train_steps(clip_id, boxes_path, train_seconds)
    render_placeholder_eval(
        video_path=video_path,
        out_dir=out_dir,
        boxes_data=boxes_data,
        skip_frames=skip_frames,
        screenshot=True,
    )
    write_placeholder_metrics(out_dir)

    print("\n-- Placeholder fine-tune done --")
    print(f"  Panel 4 video : outputs/{clip_id}/finetuned_yolo_web.mp4")
    print(f"  Hit screenshot: outputs/{clip_id}/finetuned_hit_pedestrian.png")
    print(f"  Metrics       : outputs/{clip_id}/metrics.json")
    print("  Streamlit     : streamlit run app.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO on synthetic frames + boxes.json (or --placeholder for UI demo)",
    )
    parser.add_argument("--clip-id", default="positive_01")
    parser.add_argument(
        "--video", default=None,
        help="Eval video (default: outputs/<clip-id>/dreamloop_web.mp4)",
    )
    parser.add_argument(
        "--boxes", default=None,
        help="boxes.json path (default: synthetic/<clip>/boxes.json → boxes.example.json)",
    )
    parser.add_argument(
        "--placeholder", action="store_true",
        help="Skip real training; render placeholder panel-4 artifacts for Streamlit",
    )
    parser.add_argument(
        "--train-seconds", type=float, default=2.0,
        help="Simulated train duration in placeholder mode (default: 2)",
    )
    parser.add_argument("--skip-frames", type=int, default=15)
    args = parser.parse_args()

    if not args.placeholder:
        print(
            "[ERROR] Real fine-tuning is not implemented yet.\n"
            "  Use --placeholder to generate UI demo artifacts:\n"
            f"    python yolo_finetune.py --clip-id {args.clip_id} --placeholder",
            file=sys.stderr,
        )
        return 1

    boxes_path = resolve_boxes_path(args.clip_id, args.boxes)
    video_path = Path(args.video) if args.video else None
    return run_placeholder(
        clip_id=args.clip_id,
        video_path=video_path,
        boxes_path=boxes_path,
        train_seconds=args.train_seconds,
        skip_frames=args.skip_frames,
    )


if __name__ == "__main__":
    raise SystemExit(main())

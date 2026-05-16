"""
yolo_eval.py — Off-the-shelf YOLOv8 person detection for baseline / miss screenshots.

USAGE (positive collision + blizzard processed clip):
    python dreamloop_sim.py --video nexar_videos/train/positive/XXXX.mp4 \\
        --output outputs/positive_01/dreamloop_output.mp4 --no-display
    ffmpeg -y -i outputs/positive_01/dreamloop_output.mp4 -c:v libx264 \\
        -pix_fmt yuv420p -movflags +faststart outputs/positive_01/dreamloop_web.mp4

    python yolo_eval.py --video outputs/positive_01/dreamloop_web.mp4 \\
        --clip-id positive_01 --screenshot --output outputs/positive_01/baseline_yolo.mp4

    ffmpeg -y -i outputs/positive_01/baseline_yolo.mp4 -c:v libx264 \\
        -pix_fmt yuv420p -movflags +faststart outputs/positive_01/baseline_yolo_web.mp4

Or run on raw Nexar (clearer pedestrian, no snow):
    python yolo_eval.py --video nexar_videos/train/positive/XXXX.mp4 --clip-id positive_01 --screenshot
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
PERSON_CLASS = 0

# BGR
C_BOX = (80, 220, 80)
C_MISS = (60, 60, 230)
C_LABEL_BG = (40, 40, 40)


def detect_persons(model, frame, conf: float) -> list[tuple[int, int, int, int, float]]:
    results = model(frame, verbose=False)[0]
    boxes: list[tuple[int, int, int, int, float]] = []
    if results.boxes is None:
        return boxes
    for box in results.boxes:
        if int(box.cls[0]) != PERSON_CLASS:
            continue
        score = float(box.conf[0])
        if score < conf:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes.append((x1, y1, x2, y2, score))
    return boxes


def draw_boxes(frame, boxes: list[tuple[int, int, int, int, float]]) -> None:
    for x1, y1, x2, y2, score in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_BOX, 2)
        label = f"person {score:.2f}"
        cv2.putText(
            frame, label, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_BOX, 2, cv2.LINE_AA,
        )


def draw_miss_banner(frame) -> None:
    text = "NO PERSON DETECTED (baseline YOLOv8n)"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), C_LABEL_BG, -1)
    cv2.putText(
        frame, text, (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_MISS, 2, cv2.LINE_AA,
    )


def try_ffmpeg_h264(src: Path, dst: Path) -> bool:
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(dst),
            ],
            check=True,
            capture_output=True,
        )
        return dst.is_file()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_eval(
    video_path: Path,
    model_path: str,
    conf: float,
    output_path: Path | None,
    clip_id: str,
    screenshot: bool,
    save_all_misses: bool,
    skip_frames: int,
    display: bool,
    web_output: bool,
) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics", file=sys.stderr)
        return 1

    if not video_path.is_file():
        print(f"[ERROR] Video not found: {video_path}", file=sys.stderr)
        return 1

    out_dir = ROOT / "outputs" / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = out_dir / "baseline_yolo.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}", file=sys.stderr)
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        print(f"[ERROR] Cannot write: {output_path}", file=sys.stderr)
        return 1

    model = YOLO(model_path)
    print(f"[INFO] Model   : {model_path} (off-the-shelf, person class only)")
    print(f"[INFO] Video   : {video_path}")
    print(f"[INFO] Conf    : {conf}")
    print(f"[INFO] Output  : {output_path}")
    print(f"[INFO] Frames  : {total} @ {fps:.1f} fps")

    miss_dir = out_dir / "screenshots" / "misses"
    if save_all_misses:
        miss_dir.mkdir(parents=True, exist_ok=True)

    screenshot_path = out_dir / "baseline_miss_pedestrian.png"
    saved_screenshot = False
    miss_count = 0
    hit_count = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        boxes = detect_persons(model, frame, conf)
        annotated = frame.copy()

        if boxes:
            hit_count += 1
            draw_boxes(annotated, boxes)
        else:
            miss_count += 1
            draw_miss_banner(annotated)
            if frame_idx >= skip_frames:
                if save_all_misses:
                    cv2.imwrite(str(miss_dir / f"miss_{frame_idx:05d}.png"), annotated)
                if screenshot and not saved_screenshot:
                    cv2.imwrite(str(screenshot_path), annotated)
                    saved_screenshot = True
                    print(f"[INFO] Miss screenshot saved: {screenshot_path} (frame {frame_idx})")

        writer.write(annotated)

        if display:
            cv2.imshow("YOLO baseline eval (person)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

        frame_idx += 1
        if frame_idx % 60 == 0 and total > 0:
            print(f"[INFO] Processed {frame_idx}/{total}…")

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    print(f"[INFO] Person detected: {hit_count} frames | Miss: {miss_count} frames")
    if screenshot and not saved_screenshot:
        print(
            "[WARN] No miss frame saved after skip_frames="
            f"{skip_frames}. Try lower --conf or another clip."
        )
    elif screenshot:
        print(f"[INFO] Use screenshot for report: {screenshot_path}")

    if web_output:
        web_path = (
            output_path
            if output_path.name.endswith("_web.mp4")
            else output_path.parent / "baseline_yolo_web.mp4"
        )
        if try_ffmpeg_h264(output_path, web_path):
            print(f"[INFO] Browser-ready: {web_path}")
        else:
            print("[WARN] ffmpeg not found; run manually for Streamlit playback.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Off-the-shelf YOLOv8n person detection — baseline miss screenshots",
    )
    parser.add_argument(
        "--video", default=None,
        help="Input .mp4. If omitted, uses outputs/<clip-id>/dreamloop_web.mp4 (blizzard clip for panels 3–4)",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Weights (default: off-the-shelf yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Person confidence threshold")
    parser.add_argument("--clip-id", default="positive_01", help="outputs/<clip-id>/ folder name")
    parser.add_argument("--output", default=None, help="Annotated .mp4 path (default: outputs/<clip-id>/baseline_yolo.mp4)")
    parser.add_argument(
        "--screenshot", action="store_true",
        help="Save first miss frame to outputs/<clip-id>/baseline_miss_pedestrian.png",
    )
    parser.add_argument(
        "--save-all-misses", action="store_true",
        help="Save every miss frame under outputs/<clip-id>/screenshots/misses/",
    )
    parser.add_argument(
        "--skip-frames", type=int, default=15,
        help="Ignore misses in first N frames (intro/black)",
    )
    parser.add_argument("--display", action="store_true", help="Live preview window")
    parser.add_argument(
        "--web", action="store_true",
        help="Also write baseline_yolo_web.mp4 via ffmpeg when done",
    )
    args = parser.parse_args()

    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = ROOT / video_path
    else:
        video_path = ROOT / "outputs" / args.clip_id / "dreamloop_web.mp4"
        print(f"[INFO] --video not set; using blizzard clip: {video_path}")

    output_path = Path(args.output) if args.output else None
    if output_path and not output_path.is_absolute():
        output_path = ROOT / output_path

    return run_eval(
        video_path=video_path,
        model_path=args.model,
        conf=args.conf,
        output_path=output_path,
        clip_id=args.clip_id,
        screenshot=args.screenshot,
        save_all_misses=args.save_all_misses,
        skip_frames=args.skip_frames,
        display=args.display,
        web_output=args.web,
    )


if __name__ == "__main__":
    raise SystemExit(main())

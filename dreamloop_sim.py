"""
DreamLoop AV Simulator  (Cosmos-Drive-Dreams Edition)
==========================================================
Plays a driving video (e.g. Cosmos-Drive-Dreams snowy assets) with a
HUD overlay showing:

  - Bounding box when a car is "detected" ahead
  - Speed: 20 mph (BRAKE) when car detected, 60 mph (MAINTAIN) otherwise
  - Procedural falling-snow particle overlay
  - Smooth confidence readout (rolling average)
  - Flash transition on BRAKE ↔ MAINTAIN state change
  - "COSMOS SCENE" synthetic-data badge

USAGE:
    # Point at a Cosmos-Drive-Dreams downloaded clip:
    python dreamloop_sim_v2.py --video cosmos_snowy_scene.mp4

    # Use webcam instead:
    python dreamloop_sim_v2.py --video 0

    # Use real YOLOv8 (pip install ultralytics):
    python dreamloop_sim_v2.py --video cosmos_snowy_scene.mp4 --model yolo

    Optional flags:
      --model  mock | yolo      Detector backend   (default: mock)
      --conf   0.0–1.0          YOLO conf threshold (default: 0.4)

DEPS (mock mode):   pip install opencv-python
DEPS (yolo mode):   pip install opencv-python ultralytics
"""

import cv2
import argparse
import time
import math
import random
from collections import deque


# ─────────────────────────────────────────────────────────────
# Mock detector
# Simulates a bounding box that drifts and cycles detect/clear.
# ─────────────────────────────────────────────────────────────
class MockDetector:
    """Simulates intermittent car detection for demo purposes."""

    def __init__(self, w, h):
        self.w = w
        self.h = h

    def detect(self, frame, frame_idx):
        """Returns list of (x1, y1, x2, y2, confidence) or []."""
        # Cycle: detected 5 s, clear 3 s  (at ~30 fps → 150 / 90 frames)
        cycle = frame_idx % 240
        if cycle < 150:
            drift = int(20 * math.sin(frame_idx * 0.05))
            cx = self.w // 2 + drift
            cy = int(self.h * 0.45)
            bw  = int(self.w  * 0.18)
            bh  = int(self.h  * 0.15)
            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(self.w, cx + bw // 2)
            y2 = min(self.h, cy + bh // 2)
            conf = 0.72 + 0.10 * math.sin(frame_idx * 0.1)
            return [(x1, y1, x2, y2, round(conf, 2))]
        return []


# ─────────────────────────────────────────────────────────────
# YOLO detector  (optional — requires `pip install ultralytics`)
# ─────────────────────────────────────────────────────────────
class YOLODetector:
    def __init__(self, conf_threshold=0.4):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Run: pip install ultralytics")
        self.model = YOLO("yolov8n.pt")          # downloads on first run
        self.conf  = conf_threshold
        self.vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def detect(self, frame, frame_idx):
        results = self.model(frame, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in self.vehicle_classes:
                conf = float(box.conf[0])
                if conf >= self.conf:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2, round(conf, 2)))
        return boxes


# ─────────────────────────────────────────────────────────────
# Snow particle system
# ─────────────────────────────────────────────────────────────
class SnowSystem:
    """Lightweight procedural snow overlay."""

    def __init__(self, w, h, count=180):
        self.w = w
        self.h = h
        self.particles = [self._new_particle(random.randint(0, h)) for _ in range(count)]

    def _new_particle(self, y_start=0):
        return {
            "x": random.uniform(0, self.w),
            "y": float(y_start),
            "speed": random.uniform(1.2, 3.5),
            "drift": random.uniform(-0.4, 0.4),
            "radius": random.randint(1, 3),
            "alpha": random.uniform(0.4, 0.9),
        }

    def update_and_draw(self, frame):
        overlay = frame.copy()
        for p in self.particles:
            p["y"] += p["speed"]
            p["x"] += p["drift"]
            if p["y"] > self.h or p["x"] < 0 or p["x"] > self.w:
                self.particles[self.particles.index(p)] = self._new_particle(0)
                continue
            cx, cy = int(p["x"]), int(p["y"])
            cv2.circle(overlay, (cx, cy), p["radius"], (230, 235, 240), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)


# ─────────────────────────────────────────────────────────────
# HUD helpers
# ─────────────────────────────────────────────────────────────
FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN

# BGR palette
C_GREEN    = (80, 220, 80)
C_RED      = (60,  60, 230)
C_YELLOW   = (30, 200, 220)
C_WHITE    = (240, 240, 240)
C_DARK     = (20,  20,  20)
C_CYAN     = (210, 200,  60)
C_BOX      = (40, 180, 255)
C_COSMOS   = (255, 140,  30)   # orange accent for Cosmos badge


def semi_rect(img, x1, y1, x2, y2, color, alpha=0.50):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_hud(frame, speed, detections, smooth_conf,
             fps, frame_idx, w, h,
             flash_frames, is_braking):
    """Render the full HUD onto `frame` in-place."""

    # ── Transition flash ──────────────────────────────────────
    # When state just changed, briefly pulse the full border.
    if flash_frames > 0:
        flash_color = C_RED if is_braking else C_GREEN
        t = flash_frames / 12.0           # 1.0 → 0.0
        overlay = frame.copy()
        thick = max(1, int(10 * t))
        cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), flash_color, thick)
        cv2.addWeighted(overlay, 0.6 * t, frame, 1 - 0.6 * t, 0, frame)

    # ── Speed panel  (bottom-left) ────────────────────────────
    pw, ph = 220, 110
    px, py = 20, h - ph - 20
    semi_rect(frame, px, py, px + pw, py + ph, C_DARK, 0.65)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), C_WHITE, 1)
    cv2.putText(frame, "SPEED",
                (px + 10, py + 28), FONT, 0.55, C_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{speed} mph",
                (px + 10, py + 70), FONT, 1.4, C_WHITE, 2, cv2.LINE_AA)

    # ── Status panel  (bottom-center) ────────────────────────
    status_color = C_RED   if is_braking else C_GREEN
    status_text  = "BRAKE" if is_braking else "MAINTAIN"
    sw, sh = 240, 70
    sx = w // 2 - sw // 2
    sy = h - sh - 20
    semi_rect(frame, sx, sy, sx + sw, sy + sh, C_DARK, 0.65)
    cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), status_color, 2)
    cv2.putText(frame, status_text,
                (sx + 20, sy + 46), FONT, 1.2, status_color, 2, cv2.LINE_AA)

    # ── Detection panel  (bottom-right) ──────────────────────
    dw, dh = 240, 110
    dx = w - dw - 20
    dy = h - dh - 20
    semi_rect(frame, dx, dy, dx + dw, dy + dh, C_DARK, 0.65)
    cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), C_WHITE, 1)
    det_label = "CAR DETECTED" if detections else "NO DETECTION"
    det_color  = C_RED         if detections else C_GREEN
    cv2.putText(frame, "AHEAD",
                (dx + 10, dy + 28), FONT, 0.55, C_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, det_label,
                (dx + 10, dy + 62), FONT, 0.62, det_color, 1, cv2.LINE_AA)
    if detections and smooth_conf > 0:
        cv2.putText(frame, f"conf: {smooth_conf:.2f}",
                    (dx + 10, dy + 90), FONT_MONO, 1.2, C_WHITE, 1, cv2.LINE_AA)

    # ── FPS + frame (top-right) ───────────────────────────────
    cv2.putText(frame, f"FPS: {fps:.1f}  |  frame: {frame_idx}",
                (w - 290, 28), FONT_MONO, 1.3, C_WHITE, 1, cv2.LINE_AA)

    # ── DreamLoop branding (top-left) ─────────────────────────
    cv2.putText(frame, "DreamLoop AV Sim",
                (20, 32), FONT, 0.7, C_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, "Snowy Edge-Case Demo",
                (20, 56), FONT_MONO, 1.2, C_WHITE, 1, cv2.LINE_AA)

    # ── COSMOS badge (top-left, below branding) ───────────────
    badge_txt = "COSMOS SCENE  |  SYNTHETIC"
    (bw2, bh2), _ = cv2.getTextSize(badge_txt, FONT_MONO, 1.0, 1)
    bx, by = 20, 66
    semi_rect(frame, bx - 4, by, bx + bw2 + 8, by + bh2 + 6, C_COSMOS, 0.7)
    cv2.putText(frame, badge_txt,
                (bx, by + bh2 + 2), FONT_MONO, 1.0, C_DARK, 1, cv2.LINE_AA)

    # ── Bounding boxes ────────────────────────────────────────
    for (x1, y1, x2, y2, conf) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_BOX, 2)
        corner = 14
        for cx2, cy2, ddx, ddy in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx2, cy2), (cx2 + ddx * corner, cy2), C_YELLOW, 2)
            cv2.line(frame, (cx2, cy2), (cx2, cy2 + ddy * corner), C_YELLOW, 2)
        label = f"CAR  {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, FONT_MONO, 1.2, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), C_BOX, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    FONT_MONO, 1.2, C_DARK, 1, cv2.LINE_AA)

    # ── Flashing BRAKE strip (bottom edge) ────────────────────
    if is_braking and (frame_idx // 15) % 2 == 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 8), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)


# ─────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────
def run_sim(video_path, detector_mode, conf_threshold):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video  : {w}×{h} @ {native_fps:.1f} fps  ({total} frames)")
    print(f"[INFO] Detector: {detector_mode.upper()}")
    print("[INFO] Keys   : SPACE = pause/resume   Q / ESC = quit")

    detector   = YOLODetector(conf_threshold) if detector_mode == "yolo" \
                 else MockDetector(w, h)
    snow       = SnowSystem(w, h, count=200)

    # Confidence smoother — rolling mean over last 8 detections
    conf_buf   = deque(maxlen=8)

    frame_idx  = 0
    paused     = False
    prev_time  = time.time()
    fps_disp   = native_fps
    prev_brake = False          # last frame's brake state
    flash_ctr  = 0              # countdown for transition flash

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            # ── Detection ──────────────────────────────────────
            detections = detector.detect(frame, frame_idx)

            # ── Confidence smoothing ───────────────────────────
            if detections:
                conf_buf.append(detections[0][4])
            smooth_conf = sum(conf_buf) / len(conf_buf) if conf_buf else 0.0

            # ── Speed / brake logic ────────────────────────────
            is_braking = bool(detections)
            speed      = 20 if is_braking else 60

            # ── State-change flash trigger ─────────────────────
            if is_braking != prev_brake:
                flash_ctr  = 12      # ~12 frames of flash at 30 fps ≈ 0.4 s
            prev_brake = is_braking
            if flash_ctr > 0:
                flash_ctr -= 1

            # ── FPS ────────────────────────────────────────────
            now       = time.time()
            fps_disp  = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # ── Snow overlay ───────────────────────────────────
            snow.update_and_draw(frame)

            # ── HUD ────────────────────────────────────────────
            draw_hud(frame, speed, detections, smooth_conf,
                     fps_disp, frame_idx, w, h,
                     flash_ctr, is_braking)

            frame_idx += 1

        cv2.imshow("DreamLoop AV Simulator  ·  Cosmos Edition", frame)

        key = cv2.waitKey(max(1, int(1000 / native_fps))) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
            print("[INFO]", "Paused" if paused else "Resumed")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Simulation ended.")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DreamLoop AV Simulator — Cosmos Edition")
    parser.add_argument("--video", default="0",
                        help="Path to video file or '0' for webcam (default: webcam)")
    parser.add_argument("--model", default="mock", choices=["mock", "yolo"],
                        help="Detector backend: 'mock' or 'yolo' (default: mock)")
    parser.add_argument("--conf",  type=float, default=0.4,
                        help="YOLO confidence threshold 0–1 (default: 0.4)")
    args = parser.parse_args()

    source = int(args.video) if args.video.isdigit() else args.video
    run_sim(source, args.model, args.conf)

"""
render_clip.py — Offline pipeline for one dashboard clip.

  Panel 1: raw Nexar (already on disk)
  Panel 2: dreamloop_sim --model yolo (cars + HUD + snow, no pedestrians)
  Panel 3: yolo_eval on dreamloop_web.mp4 (person boxes + miss screenshot)
  Panel 4: yolo_finetune.py --placeholder (hit video + screenshot; no real train)

USAGE:
    python render_clip.py --clip-id positive_01 \\
        --video nexar_videos/train/positive/00000.mp4

    python render_clip.py --clip-id positive_01
        # raw path from clips.json or auto-discovery
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from clips_catalog import clip_by_id, load_clips, paths_for_clip


def run(cmd: list[str], label: str) -> None:
    print(f"\n── {label} ──")
    print(" ", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"[ERROR] Failed: {label} (exit {result.returncode})")


def ffmpeg_h264(src: Path, dst: Path) -> None:
    run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(dst),
        ],
        f"ffmpeg → {dst.name}",
    )


def resolve_raw_video(clip_id: str, video_arg: str | None) -> Path:
    if video_arg:
        p = Path(video_arg)
        return p if p.is_absolute() else ROOT / p

    clips = load_clips(ROOT)
    clip = clip_by_id(clips, clip_id)
    if clip is None:
        raise SystemExit(
            f"[ERROR] Unknown clip-id '{clip_id}'. "
            "Pass --video or add clips.json / nexar mp4 files."
        )
    raw = ROOT / clip["raw"]
    if not raw.is_file():
        raise SystemExit(f"[ERROR] Raw video not found: {raw}")
    return raw


def render_clip(
    clip_id: str,
    raw_video: Path,
    sim_conf: float,
    yolo_conf: float,
    skip_sim: bool,
    skip_baseline: bool,
    placeholder_finetune: bool,
    display_sim: bool,
) -> None:
    paths = paths_for_clip(ROOT, {"id": clip_id, "raw": str(raw_video.relative_to(ROOT)).replace("\\", "/")})
    out_dir = paths["processed"].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Clip id     : {clip_id}")
    print(f"[INFO] Raw video   : {raw_video}")
    print(f"[INFO] Output dir  : {out_dir}")

    if not skip_sim:
        sim_cmd = [
            sys.executable,
            str(ROOT / "dreamloop_sim.py"),
            "--video", str(raw_video),
            "--model", "yolo",
            "--conf", str(sim_conf),
            "--output", str(paths["processed_raw"]),
        ]
        if not display_sim:
            sim_cmd.append("--no-display")
        run(sim_cmd, "Panel 2: dreamloop_sim (cars + HUD + snow)")

        if not paths["processed_raw"].is_file():
            raise SystemExit(f"[ERROR] Missing {paths['processed_raw']}")

        ffmpeg_h264(paths["processed_raw"], paths["processed"])
    else:
        print("[INFO] Skipping dreamloop_sim (--skip-sim)")
        if not paths["processed"].is_file():
            raise SystemExit(
                f"[ERROR] Need {paths['processed']} — run without --skip-sim first."
            )

    if not skip_baseline:
        run(
            [
                sys.executable,
                str(ROOT / "yolo_eval.py"),
                "--video", str(paths["processed"]),
                "--clip-id", clip_id,
                "--conf", str(yolo_conf),
                "--screenshot",
                "--web",
            ],
            "Panel 3: yolo_eval on blizzard clip (person + screenshot)",
        )
    else:
        print("[INFO] Skipping yolo_eval (--skip-baseline)")

    if placeholder_finetune:
        run(
            [
                sys.executable,
                str(ROOT / "yolo_finetune.py"),
                "--clip-id", clip_id,
                "--placeholder",
            ],
            "Panel 4: placeholder fine-tune (UI demo)",
        )

    print("\n── Done ─────────────────────────────────────")
    print(f"  Panel 2 video : {paths['processed']}")
    print(f"  Panel 3 video : {paths['baseline_video']}")
    print(f"  Screenshot    : {paths['baseline_screenshot']}")
    if placeholder_finetune:
        print(f"  Panel 4 video : {paths['finetuned_video']}")
        print(f"  Hit screenshot  : {paths['finetuned_screenshot']}")
    print("  Streamlit     : streamlit run app.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render panel 2 + 3 artifacts for one clip")
    parser.add_argument("--clip-id", required=True, help="e.g. positive_01")
    parser.add_argument(
        "--video", default=None,
        help="Raw Nexar .mp4 (default: from clips.json / discovery)",
    )
    parser.add_argument("--sim-conf", type=float, default=0.3, help="YOLO conf for dreamloop_sim (cars)")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="Person conf for yolo_eval")
    parser.add_argument("--skip-sim", action="store_true", help="Only run yolo_eval (dreamloop_web exists)")
    parser.add_argument("--skip-baseline", action="store_true", help="Only run dreamloop_sim + ffmpeg")
    parser.add_argument(
        "--placeholder-finetune", action="store_true",
        help="Run yolo_finetune.py --placeholder for panel 4 (no real training)",
    )
    parser.add_argument("--display-sim", action="store_true", help="Show live dreamloop window")
    args = parser.parse_args()

    raw = resolve_raw_video(args.clip_id, args.video)
    render_clip(
        clip_id=args.clip_id,
        raw_video=raw,
        sim_conf=args.sim_conf,
        yolo_conf=args.yolo_conf,
        skip_sim=args.skip_sim,
        skip_baseline=args.skip_baseline,
        placeholder_finetune=args.placeholder_finetune,
        display_sim=args.display_sim,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

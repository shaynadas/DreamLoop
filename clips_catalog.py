"""Clip catalog: Nexar sources and per-clip pre-rendered outputs under outputs/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CLIPS_FILE = "clips.json"


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def discover_clips(root: Path) -> list[dict[str, Any]]:
    """Build clip entries from nexar_videos/train/positive and .../negative."""
    clips: list[dict[str, Any]] = []
    pos_dir = root / "nexar_videos" / "train" / "positive"
    neg_dir = root / "nexar_videos" / "train" / "negative"

    if pos_dir.is_dir():
        for i, mp4 in enumerate(sorted(pos_dir.glob("*.mp4")), start=1):
            clips.append(
                {
                    "id": f"positive_{i:02d}",
                    "label": f"Positive collision #{i} (pedestrian)",
                    "raw": _rel(root, mp4),
                    "kind": "positive",
                }
            )

    if neg_dir.is_dir():
        for i, mp4 in enumerate(sorted(neg_dir.glob("*.mp4")), start=1):
            clip_id = "negative_rainy" if i == 1 else f"negative_{i:02d}"
            label = "Negative (rainy)" if i == 1 else f"Negative #{i}"
            clips.append(
                {
                    "id": clip_id,
                    "label": label,
                    "raw": _rel(root, mp4),
                    "kind": "negative",
                }
            )

    return clips


def load_clips(root: Path) -> list[dict[str, Any]]:
    """Load clips.json if present, else auto-discover from nexar folders."""
    config_path = root / CLIPS_FILE
    if config_path.is_file():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            clips = data.get("clips", data if isinstance(data, list) else [])
            if clips:
                return clips
        except (json.JSONDecodeError, OSError):
            pass
    return discover_clips(root)


def output_dir(root: Path, clip_id: str) -> Path:
    return root / "outputs" / clip_id


def paths_for_clip(root: Path, clip: dict[str, Any]) -> dict[str, Path]:
    """Standard pre-rendered artifact paths for dashboard and YOLO eval."""
    clip_id = clip["id"]
    out = output_dir(root, clip_id)
    raw = root / clip["raw"]
    return {
        "raw": raw,
        "processed": out / "dreamloop_web.mp4",
        "processed_raw": out / "dreamloop_output.mp4",
        "baseline_video": out / "baseline_yolo_web.mp4",
        "baseline_video_raw": out / "baseline_yolo.mp4",
        "baseline_screenshot": out / "baseline_miss_pedestrian.png",
        "finetuned_video": out / "finetuned_yolo_web.mp4",
        "finetuned_screenshot": out / "finetuned_hit_pedestrian.png",
        "metrics": out / "metrics.json",
    }


def clip_by_id(clips: list[dict[str, Any]], clip_id: str) -> dict[str, Any] | None:
    for clip in clips:
        if clip["id"] == clip_id:
            return clip
    return None

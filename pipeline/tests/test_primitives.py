"""Primitive correctness — IoU math, MP4 extraction, cosmos_runner shape.

Catch the kind of bug that silently destroys the headline mAP number.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolo_eval import iou_xyxy, extract_frames_from_mp4  # noqa: E402
import cosmos_runner  # noqa: E402


_results = []


def check(name, fn):
    try:
        fn()
        _results.append((name, "PASS"))
        print(f"  PASS  {name}")
    except Exception as e:
        _results.append((name, f"FAIL: {e}"))
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")


def t_iou_identical():
    a = np.array([10, 10, 20, 20], dtype=float)
    assert abs(iou_xyxy(a, a) - 1.0) < 1e-9


def t_iou_disjoint():
    a = np.array([0, 0, 10, 10], dtype=float)
    b = np.array([100, 100, 110, 110], dtype=float)
    assert iou_xyxy(a, b) == 0.0


def t_iou_half_overlap():
    a = np.array([0, 0, 10, 10], dtype=float)
    b = np.array([5, 0, 15, 10], dtype=float)
    # intersection = 5*10 = 50, union = 100+100-50 = 150
    assert abs(iou_xyxy(a, b) - 50 / 150) < 1e-6


def t_iou_contained():
    a = np.array([0, 0, 10, 10], dtype=float)
    b = np.array([2, 2, 8, 8], dtype=float)
    # intersection = 36, union = 100+36-36 = 100
    assert abs(iou_xyxy(a, b) - 0.36) < 1e-6


def t_iou_touching_zero_area():
    a = np.array([0, 0, 10, 10], dtype=float)
    b = np.array([10, 0, 20, 10], dtype=float)
    # Edge-touching → intersection has 0 area → IoU = 0.
    assert iou_xyxy(a, b) == 0.0


def t_mp4_extract_roundtrip(tmp: Path):
    """Synth a 5-frame MP4, extract it, verify 5 PNGs come out."""
    mp4 = tmp / "synth.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4), fourcc, 10.0, (64, 64))
    assert writer.isOpened(), "could not open mp4 writer"
    for i in range(5):
        canvas = np.full((64, 64, 3), i * 40, dtype=np.uint8)
        writer.write(canvas)
    writer.release()

    out = extract_frames_from_mp4(mp4, tmp / "frames")
    pngs = sorted(out.glob("*.png"))
    assert len(pngs) == 5, f"expected 5 PNGs, got {len(pngs)}"


def t_cosmos_runner_missing_repo_raises():
    """If COSMOS_REPO doesn't exist, _check_repo should raise FileNotFoundError."""
    saved = cosmos_runner.config.COSMOS_REPO
    cosmos_runner.config.COSMOS_REPO = Path("/this/path/definitely/does/not/exist")
    try:
        try:
            cosmos_runner._check_repo()
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError")
    finally:
        cosmos_runner.config.COSMOS_REPO = saved


def main():
    tmp = Path(tempfile.mkdtemp(prefix="dreamloop_prim_"))
    print(f"working in {tmp}")
    try:
        cases = [
            ("iou: identical → 1.0", t_iou_identical),
            ("iou: disjoint → 0.0", t_iou_disjoint),
            ("iou: half-overlap → 1/3", t_iou_half_overlap),
            ("iou: contained → 0.36", t_iou_contained),
            ("iou: edge-touching → 0", t_iou_touching_zero_area),
            ("mp4: extract round-trip", lambda: t_mp4_extract_roundtrip(tmp)),
            ("cosmos_runner: missing repo raises", t_cosmos_runner_missing_repo_raises),
        ]
        for name, fn in cases:
            print(f"\n[{name}]")
            check(name, fn)
    finally:
        n_fail = sum(1 for _, s in _results if s.startswith("FAIL"))
        if n_fail == 0:
            shutil.rmtree(tmp, ignore_errors=True)

    n_pass = sum(1 for _, s in _results if s == "PASS")
    n_fail = sum(1 for _, s in _results if s.startswith("FAIL"))
    print(f"\n{'='*60}\n{n_pass} passed, {n_fail} failed")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()

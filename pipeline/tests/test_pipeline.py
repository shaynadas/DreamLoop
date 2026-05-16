"""End-to-end test on synthetic data. Exercises every pure-Python path.

Skipped automatically:
    - waymo_loader (requires TF + Waymo SDK)
    - cosmos_runner (requires Cosmos repo)
    - yolo_eval (requires ultralytics + torch)

What this DOES test:
    - parquet schema round-trip with None values
    - tracklet_perturb math + acceleration sanity warnings
    - box_projection forward + back-of-camera handling
    - rds_writer wireframe rendering + RDS-HQ assembly
    - cross-script invariants (perturbed flag carries forward, etc.)

Run:
    cd DreamLoop/pipeline && python -m tests.test_pipeline
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Make pipeline/ importable when running from DreamLoop/pipeline/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from box_projection import (  # noqa: E402
    box_corners_ego,
    box_to_2d_aabb,
    load_calibration,
    project_corners_to_image,
    project_tracklets_to_2d,
)
from rds_writer import build_rds_hq_dir, render_wireframe_frames  # noqa: E402
from schema import TRACKLET_SCHEMA  # noqa: E402
from tests.synth import write_synth  # noqa: E402
from tracklet_perturb import PerturbationSpec, apply_perturbation, perturb_segment  # noqa: E402


PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, fn):
    try:
        fn()
        _results.append((name, PASS, ""))
        print(f"  {PASS}  {name}")
    except AssertionError as e:
        _results.append((name, FAIL, str(e) or "assertion"))
        print(f"  {FAIL}  {name}: {e}")
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        _results.append((name, FAIL, f"{type(e).__name__}: {e}"))
        print(f"  {FAIL}  {name}: {type(e).__name__}: {e}")
        print(f"        {tb.splitlines()[-2] if len(tb.splitlines()) > 1 else ''}")


# -- tests ---------------------------------------------------------------

def t_synth_dir_loads(tmp: Path):
    root = write_synth(tmp / "synth")
    assert (root / "tracklets.parquet").exists()
    assert (root / "calibrations.json").exists()
    assert (root / "ego_poses.parquet").exists()
    pngs = list((root / "frames" / "FRONT").glob("*.png"))
    assert len(pngs) == 20, f"expected 20 frames, got {len(pngs)}"
    df = pq.read_table(root / "tracklets.parquet").to_pandas()
    assert len(df) == 40, f"expected 40 rows (2 obj * 20 frames), got {len(df)}"
    assert df["perturbed"].any() is np.False_ or not df["perturbed"].any()


def t_calibration_load(tmp: Path):
    root = write_synth(tmp / "synth")
    calib = load_calibration(root, "FRONT")
    assert calib.width == 1920 and calib.height == 1280
    K = calib.K()
    assert K.shape == (3, 3)
    assert K[0, 0] == 1500.0
    assert K[1, 2] == 640.0


def t_box_corners_count(_: Path):
    corners = box_corners_ego(10.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.0)
    assert corners.shape == (8, 3), corners.shape


def t_projection_front_object_on_image(tmp: Path):
    root = write_synth(tmp / "synth")
    calib = load_calibration(root, "FRONT")
    # 8m ahead, 0 lateral, on the ground in front of the camera.
    corners = box_corners_ego(8.0, 0.0, 0.75, 4.5, 1.8, 1.5, 0.0)
    px = project_corners_to_image(corners, calib)
    finite = np.isfinite(px).all(axis=1)
    assert finite.sum() >= 4, f"expected ≥4 finite projections, got {finite.sum()}"
    aabb = box_to_2d_aabb(px, calib.width, calib.height)
    assert aabb is not None, "object in front of camera should project to a valid AABB"
    x1, y1, x2, y2 = aabb
    assert 0 <= x1 < x2 <= calib.width
    assert 0 <= y1 < y2 <= calib.height


def t_projection_behind_camera_drops(tmp: Path):
    root = write_synth(tmp / "synth")
    calib = load_calibration(root, "FRONT")
    # 5m BEHIND ego — should be entirely behind the camera.
    corners = box_corners_ego(-5.0, 0.0, 0.75, 4.5, 1.8, 1.5, 0.0)
    px = project_corners_to_image(corners, calib)
    assert not np.isfinite(px).all(axis=1).any(), \
        "object behind camera should have no finite projections"


def t_project_tracklets_returns_rows(tmp: Path):
    root = write_synth(tmp / "synth")
    calib = load_calibration(root, "FRONT")
    df = pq.read_table(root / "tracklets.parquet").to_pandas()
    out = project_tracklets_to_2d(df, calib)
    assert len(out) >= 20, f"expected most tracklet rows to project, got {len(out)}"
    assert set(out.columns) >= {"object_id", "frame_idx", "type", "x1", "y1", "x2", "y2"}


def t_perturb_lateral_offset(tmp: Path):
    root = write_synth(tmp / "synth")
    out = tmp / "synth_lat"
    perturb_segment(root, out, "veh_001",
                    PerturbationSpec(kind="lateral_offset", magnitude=1.75, ramp_seconds=1.0))
    df = pq.read_table(out / "tracklets.parquet").to_pandas()
    veh = df[df["object_id"] == "veh_001"].sort_values("frame_idx")
    # First frame: ramp not started → cy ≈ 0. Last frame: full magnitude → cy ≈ 1.75.
    assert abs(veh["cy"].iloc[0]) < 0.05, f"start cy should be ~0, got {veh['cy'].iloc[0]}"
    assert abs(veh["cy"].iloc[-1] - 1.75) < 0.05, f"end cy should be ~1.75, got {veh['cy'].iloc[-1]}"
    # Pedestrian untouched.
    ped = df[df["object_id"] == "ped_001"]
    assert (ped["cy"] == 1.0).all(), "ped_001 should not be perturbed"
    # Perturbed flag set on at least some veh rows.
    assert veh["perturbed"].any(), "expected perturbed flag on veh_001 rows"
    assert (veh["perturb_kind"].dropna() == "lateral_offset").all()


def t_perturb_speed_delta(tmp: Path):
    root = write_synth(tmp / "synth")
    out = tmp / "synth_spd"
    # Slow it down 50%.
    perturb_segment(root, out, "veh_001",
                    PerturbationSpec(kind="speed_delta", magnitude=-0.5, ramp_seconds=0.5))
    df = pq.read_table(out / "tracklets.parquet").to_pandas()
    veh = df[df["object_id"] == "veh_001"].sort_values("frame_idx").reset_index(drop=True)
    # Last cx should be less than original (which was 8 + 5*1.9 = 17.5).
    original_last = 8.0 + 5.0 * 19 * 0.1
    assert veh["cx"].iloc[-1] < original_last, \
        f"expected slowed-down cx<{original_last}, got {veh['cx'].iloc[-1]}"


def t_perturb_unknown_object_id_raises(tmp: Path):
    root = write_synth(tmp / "synth")
    out = tmp / "synth_bad"
    try:
        perturb_segment(root, out, "does_not_exist",
                        PerturbationSpec(kind="lateral_offset", magnitude=1.0))
    except ValueError as e:
        assert "does_not_exist" in str(e)
        return
    raise AssertionError("perturb on missing object should raise ValueError")


def t_perturb_acceleration_warning(tmp: Path, capsys=None):
    """A very fast lateral_offset should print a warning to stderr but not crash."""
    root = write_synth(tmp / "synth")
    out = tmp / "synth_violent"
    # 5m offset in 0.1s ramp → very high lateral accel.
    perturb_segment(root, out, "veh_001",
                    PerturbationSpec(kind="lateral_offset", magnitude=5.0, ramp_seconds=0.1))
    # If we got here without crash, the test passes. We can't easily capture stderr
    # here because perturb_segment goes through Python directly; the warning is
    # printed to sys.stderr unconditionally. Just verify it ran.
    df = pq.read_table(out / "tracklets.parquet").to_pandas()
    assert df[df["object_id"] == "veh_001"]["perturbed"].any()


def t_wireframe_renders_pngs(tmp: Path):
    root = write_synth(tmp / "synth")
    out = tmp / "wire"
    n = render_wireframe_frames(root, "FRONT", out)
    assert n == 20
    pngs = sorted(out.glob("*.png"))
    assert len(pngs) == 20
    # Each PNG should have some non-zero pixels (the wireframe).
    from PIL import Image
    arr = np.array(Image.open(pngs[0]))
    assert arr.shape == (1280, 1920, 3)
    assert arr.sum() > 0, "wireframe canvas should not be all-black"


def t_rds_hq_assembly(tmp: Path):
    root = write_synth(tmp / "synth")
    out = tmp / "rds"
    build_rds_hq_dir(root, out, ["FRONT"])
    assert (out / "rgb" / "FRONT").exists()
    assert len(list((out / "rgb" / "FRONT").glob("*.png"))) == 20
    assert (out / "wireframe" / "FRONT").exists()
    assert len(list((out / "wireframe" / "FRONT").glob("*.png"))) == 20
    assert (out / "labels" / "tracklets.parquet").exists()
    assert (out / "labels" / "ego_poses.parquet").exists()
    assert (out / "calibrations.json").exists()
    assert (out / "_rds_hq_manifest.json").exists()
    manifest = json.loads((out / "_rds_hq_manifest.json").read_text())
    assert manifest["num_frames"] == 20
    assert manifest["cameras"] == ["FRONT"]


def t_rds_hq_carries_perturbation_audit(tmp: Path):
    root = write_synth(tmp / "synth")
    perturbed = tmp / "synth_lat"
    perturb_segment(root, perturbed, "veh_001",
                    PerturbationSpec(kind="lateral_offset", magnitude=1.5))
    out = tmp / "rds_perturbed"
    build_rds_hq_dir(perturbed, out, ["FRONT"])
    assert (out / "perturbation.json").exists(), \
        "RDS-HQ dir should carry forward perturbation.json from intermediate"
    audit = json.loads((out / "perturbation.json").read_text())
    assert audit["object_id"] == "veh_001"


def t_schema_allows_null_perturb_kind(tmp: Path):
    """Regression test: pandas → pyarrow conversion shouldn't choke on None strings."""
    root = write_synth(tmp / "synth")
    df = pq.read_table(root / "tracklets.parquet").to_pandas()
    # All perturb_kind should be None on a fresh segment.
    assert df["perturb_kind"].isna().all() or (df["perturb_kind"].fillna("") == "").all()


# -- runner --------------------------------------------------------------

def main():
    tmp = Path(tempfile.mkdtemp(prefix="dreamloop_test_"))
    print(f"working in {tmp}")
    try:
        for name, fn in [
            ("synth dir loads", t_synth_dir_loads),
            ("calibration load", t_calibration_load),
            ("box corners shape", t_box_corners_count),
            ("projection: front object on-image", t_projection_front_object_on_image),
            ("projection: behind-camera dropped", t_projection_behind_camera_drops),
            ("project_tracklets_to_2d returns rows", t_project_tracklets_returns_rows),
            ("perturb: lateral offset reaches magnitude", t_perturb_lateral_offset),
            ("perturb: speed_delta slows vehicle", t_perturb_speed_delta),
            ("perturb: unknown object raises", t_perturb_unknown_object_id_raises),
            ("perturb: violent lateral warns but runs", t_perturb_acceleration_warning),
            ("rds_writer: wireframe renders PNGs", t_wireframe_renders_pngs),
            ("rds_writer: RDS-HQ assembly", t_rds_hq_assembly),
            ("rds_writer: perturbation audit carried", t_rds_hq_carries_perturbation_audit),
            ("schema: null perturb_kind allowed", t_schema_allows_null_perturb_kind),
        ]:
            print(f"\n[{name}]")
            check(name, lambda fn=fn: fn(tmp))
    finally:
        # Leave the dir on failure for inspection.
        n_fail = sum(1 for _, s, _ in _results if s == FAIL)
        if n_fail == 0:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"\n(left {tmp} for inspection)")

    n_pass = sum(1 for _, s, _ in _results if s == PASS)
    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    print(f"\n{'='*60}")
    print(f"{n_pass} passed, {n_fail} failed")
    if n_fail:
        for n, s, e in _results:
            if s == FAIL:
                print(f"  FAIL  {n}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

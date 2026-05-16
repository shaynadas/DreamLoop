"""Adversarial tests. Each one tries to break a specific module.

If these pass, the pipeline survives the most common 'hackathon Monday morning'
class of bugs. If they fail, the bug is real and fixable now.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from box_projection import (  # noqa: E402
    box_corners_ego,
    box_to_2d_aabb,
    load_calibration,
    project_corners_to_image,
    project_tracklets_to_2d,
)
from schema import TRACKLET_SCHEMA  # noqa: E402
from tests.synth import write_synth  # noqa: E402
from tracklet_perturb import PerturbationSpec, apply_perturbation, perturb_segment  # noqa: E402


_results = []


def check(name, fn):
    try:
        fn()
        _results.append((name, "PASS", ""))
        print(f"  PASS  {name}")
    except AssertionError as e:
        _results.append((name, "FAIL", str(e) or "assertion"))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        _results.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        tb = traceback.format_exc(limit=3)
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        for line in tb.splitlines()[-4:]:
            print(f"        {line}")


# -- adversarial cases -----------------------------------------------------

def t_perturb_two_frame_object(tmp: Path):
    """Object existing for only 2 frames — np.gradient barely works."""
    root = write_synth(tmp / "synth")
    df = pq.read_table(root / "tracklets.parquet").to_pandas()
    short = df[(df["object_id"] == "veh_001") & (df["frame_idx"].isin([0, 1]))].copy()
    short = pd.concat([short, df[df["object_id"] == "ped_001"]], ignore_index=True)
    short["perturb_kind"] = short["perturb_kind"].astype("object")
    tbl = pa.Table.from_pandas(short, schema=TRACKLET_SCHEMA, preserve_index=False)
    pq.write_table(tbl, root / "tracklets.parquet")
    apply_perturbation(short, "veh_001",
                       PerturbationSpec(kind="lateral_offset", magnitude=1.0, ramp_seconds=0.5),
                       frame_rate_hz=10.0)


def t_perturb_one_frame_object(tmp: Path):
    """Object existing for only 1 frame — np.gradient with length-1 explodes."""
    root = write_synth(tmp / "synth")
    df = pq.read_table(root / "tracklets.parquet").to_pandas()
    one = df[(df["object_id"] == "veh_001") & (df["frame_idx"] == 0)].copy()
    one = pd.concat([one, df[df["object_id"] == "ped_001"]], ignore_index=True)
    one["perturb_kind"] = one["perturb_kind"].astype("object")
    # Try BOTH lateral and speed_delta on a 1-frame tracklet — speed_delta is the
    # one most likely to crash because it does gradient twice.
    apply_perturbation(one, "veh_001",
                       PerturbationSpec(kind="lateral_offset", magnitude=1.0, ramp_seconds=0.5),
                       frame_rate_hz=10.0)
    apply_perturbation(one, "veh_001",
                       PerturbationSpec(kind="speed_delta", magnitude=-0.3, ramp_seconds=0.5),
                       frame_rate_hz=10.0)


def t_perturb_invalid_range_silently_does_nothing(tmp: Path):
    """If start_frame > end_frame, perturbation should warn (or no-op safely)."""
    root = write_synth(tmp / "synth")
    df = pq.read_table(root / "tracklets.parquet").to_pandas()
    out = apply_perturbation(
        df, "veh_001",
        PerturbationSpec(kind="lateral_offset", magnitude=2.0,
                         start_frame=15, end_frame=5, ramp_seconds=0.5),
        frame_rate_hz=10.0,
    )
    veh = out[out["object_id"] == "veh_001"]
    # cy should be unchanged from original (0.0).
    assert (veh["cy"].abs() < 1e-6).all(), \
        f"invalid range should leave cy unchanged, got {veh['cy'].tolist()[:5]}"
    # And no rows should be marked perturbed.
    assert not veh["perturbed"].any(), "no rows should be flagged perturbed for invalid range"


def t_perturb_round_trip_parquet(tmp: Path):
    """Write perturbed parquet, read it back, verify floats survive."""
    root = write_synth(tmp / "synth")
    out = tmp / "synth_rt"
    perturb_segment(root, out, "veh_001",
                    PerturbationSpec(kind="lateral_offset", magnitude=1.5, ramp_seconds=1.0))
    reread = pq.read_table(out / "tracklets.parquet").to_pandas()
    # Compare against fresh apply.
    fresh_in = pq.read_table(root / "tracklets.parquet").to_pandas()
    fresh_out = apply_perturbation(fresh_in, "veh_001",
                                    PerturbationSpec(kind="lateral_offset", magnitude=1.5, ramp_seconds=1.0),
                                    frame_rate_hz=10.0)
    # Sort both and compare cy values.
    a = reread.sort_values(["object_id", "frame_idx"]).reset_index(drop=True)
    b = fresh_out.sort_values(["object_id", "frame_idx"]).reset_index(drop=True)
    assert np.allclose(a["cy"], b["cy"]), "cy values changed after parquet round-trip"


def t_perturb_perturbed_flag_carries_in_parquet(tmp: Path):
    """Regression: perturbed (bool) and perturb_kind (string|None) must survive round-trip."""
    root = write_synth(tmp / "synth")
    out = tmp / "synth_flag"
    perturb_segment(root, out, "veh_001",
                    PerturbationSpec(kind="lateral_offset", magnitude=1.0, ramp_seconds=1.0))
    reread = pq.read_table(out / "tracklets.parquet").to_pandas()
    veh = reread[reread["object_id"] == "veh_001"]
    assert veh["perturbed"].any(), "perturbed flag lost in round-trip"
    nonnull_kinds = veh["perturb_kind"].dropna()
    assert (nonnull_kinds == "lateral_offset").all(), \
        f"perturb_kind values lost or wrong: {nonnull_kinds.unique()}"


def t_projection_object_off_left_edge(tmp: Path):
    """Object 8m forward, 10m to the left — should mostly project off-image."""
    root = write_synth(tmp / "synth")
    calib = load_calibration(root, "FRONT")
    corners = box_corners_ego(8.0, 10.0, 0.75, 4.5, 1.8, 1.5, 0.0)
    px = project_corners_to_image(corners, calib)
    aabb = box_to_2d_aabb(px, calib.width, calib.height)
    # Either None (fully off-screen) or a thin sliver near x=0.
    if aabb is not None:
        x1, _, x2, _ = aabb
        assert x2 - x1 < calib.width / 2, \
            f"far-left object should yield narrow AABB, got width {x2-x1}"


def t_projection_yaw_pi_does_not_crash(tmp: Path):
    """Heading near pi — rotation matrix shouldn't NaN out."""
    root = write_synth(tmp / "synth")
    calib = load_calibration(root, "FRONT")
    corners = box_corners_ego(8.0, 0.0, 0.75, 4.5, 1.8, 1.5, np.pi - 0.001)
    px = project_corners_to_image(corners, calib)
    # At least some corners should project to finite values.
    assert np.isfinite(px).any(), "near-pi heading produced all-NaN projection"


def t_project_tracklets_empty_df(tmp: Path):
    """Empty tracklet df should produce empty output, not crash."""
    root = write_synth(tmp / "synth")
    calib = load_calibration(root, "FRONT")
    empty = pd.DataFrame(columns=["object_id", "frame_idx", "type",
                                    "cx", "cy", "cz", "length", "width", "height", "heading",
                                    "perturbed"])
    out = project_tracklets_to_2d(empty, calib)
    assert len(out) == 0
    # Even with no rows, the function should not crash. Columns should exist for
    # downstream code that does .x1, .y1 etc.
    # (We accept either no-columns or correct-columns — just no exception.)


def t_perturb_yaw_bias_changes_heading(tmp: Path):
    root = write_synth(tmp / "synth")
    out = tmp / "synth_yaw"
    perturb_segment(root, out, "veh_001",
                    PerturbationSpec(kind="yaw_bias", magnitude=0.3, ramp_seconds=0.5))
    df = pq.read_table(out / "tracklets.parquet").to_pandas()
    veh = df[df["object_id"] == "veh_001"].sort_values("frame_idx")
    # Last heading should be ~0.3 (started at 0).
    assert abs(veh["heading"].iloc[-1] - 0.3) < 0.05, \
        f"yaw_bias final heading wrong: {veh['heading'].iloc[-1]}"
    # First heading should be ~0.
    assert abs(veh["heading"].iloc[0]) < 0.05


def t_perturb_unknown_kind_raises(tmp: Path):
    root = write_synth(tmp / "synth")
    df = pq.read_table(root / "tracklets.parquet").to_pandas()
    try:
        apply_perturbation(df, "veh_001",
                           PerturbationSpec(kind="wiggle", magnitude=1.0),
                           frame_rate_hz=10.0)
    except ValueError as e:
        assert "wiggle" in str(e).lower() or "unknown" in str(e).lower()
        return
    raise AssertionError("expected ValueError for unknown kind")


def t_aabb_size_filter(tmp: Path):
    """A 1-pixel-tall AABB should be rejected (would create false positives in IoU)."""
    px = np.array([[100, 100], [101, 100], [101, 101], [100, 101],
                   [100, 100], [101, 100], [101, 101], [100, 101]], dtype=np.float64)
    aabb = box_to_2d_aabb(px, 1920, 1280)
    # 1x1 box → rejected by min size check (currently min is 2 pixels).
    assert aabb is None, f"tiny box should be rejected, got {aabb}"


# -- runner ----------------------------------------------------------------

def main():
    tmp = Path(tempfile.mkdtemp(prefix="dreamloop_edge_"))
    print(f"working in {tmp}")
    try:
        cases = [
            ("perturb: two-frame object survives", t_perturb_two_frame_object),
            ("perturb: one-frame object survives", t_perturb_one_frame_object),
            ("perturb: invalid frame range is a no-op", t_perturb_invalid_range_silently_does_nothing),
            ("perturb: parquet round-trip preserves cy", t_perturb_round_trip_parquet),
            ("perturb: perturbed flag + kind survive parquet", t_perturb_perturbed_flag_carries_in_parquet),
            ("perturb: yaw_bias updates heading", t_perturb_yaw_bias_changes_heading),
            ("perturb: unknown kind raises", t_perturb_unknown_kind_raises),
            ("project: far-left object yields narrow box", t_projection_object_off_left_edge),
            ("project: heading near pi does not NaN", t_projection_yaw_pi_does_not_crash),
            ("project: empty df does not crash", t_project_tracklets_empty_df),
            ("project: tiny AABB rejected", t_aabb_size_filter),
        ]
        for name, fn in cases:
            print(f"\n[{name}]")
            check(name, lambda fn=fn: fn(tmp))
    finally:
        n_fail = sum(1 for _, s, _ in _results if s == "FAIL")
        if n_fail == 0:
            shutil.rmtree(tmp, ignore_errors=True)

    n_pass = sum(1 for _, s, _ in _results if s == "PASS")
    n_fail = sum(1 for _, s, _ in _results if s == "FAIL")
    print(f"\n{'='*60}\n{n_pass} passed, {n_fail} failed")
    if n_fail:
        for n, s, e in _results:
            if s == "FAIL":
                print(f"  FAIL  {n}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

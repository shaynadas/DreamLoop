"""Trajectory perturbation primitives.

Operates on the tracklets.parquet produced by waymo_loader.py and writes
a new tracklets.parquet with the perturbed rows. Pure numpy/pandas — no GPU,
no TF, no Cosmos. Both pivots (data-factory + discovery search) call this.

Perturbations supported:
    lateral_offset    — smooth Y-axis drift (meters), ramped over a duration
    speed_delta       — multiplicative change to longitudinal speed (X-axis)
    yaw_bias          — adds a small heading offset (radians)

All perturbations:
  - are applied in the ego-vehicle frame at the *first frame the object appears*
  - ramp smoothly from 0 → target with a cosine schedule so the rendered video
    doesn't show a teleport
  - respect a max-acceleration sanity bound and warn (don't crash) if exceeded
  - mark every affected row with perturbed=True and perturb_kind=<name>

Usage:
    python tracklet_perturb.py <intermediate_dir> \
        --object-id <waymo_track_id>            \
        --kind lateral_offset --magnitude 1.75   \
        --ramp-seconds 2.0                        \
        --out <intermediate_dir_perturbed>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from schema import TRACKLET_SCHEMA


MAX_LATERAL_ACCEL_MPS2 = 6.0   # ~0.6g — beyond this is implausible for a road vehicle
MAX_LONG_ACCEL_MPS2 = 4.0


@dataclass
class PerturbationSpec:
    kind: str             # "lateral_offset" | "speed_delta" | "yaw_bias"
    magnitude: float      # meters | unitless multiplier | radians
    ramp_seconds: float = 2.0
    start_frame: int | None = None  # default: first frame the object appears
    end_frame: int | None = None    # default: last frame the object appears


def _cosine_ramp(t_norm: np.ndarray) -> np.ndarray:
    """Smooth 0→1 ramp on [0,1]. Returns 1.0 once t_norm >= 1."""
    out = np.clip(t_norm, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * out))


def _check_lateral_accel(offsets: np.ndarray, dt: float) -> tuple[float, bool]:
    """Returns (peak |a|, ok). offsets is per-frame lateral displacement (m)."""
    if len(offsets) < 3:
        return (0.0, True)
    v = np.gradient(offsets, dt)
    a = np.gradient(v, dt)
    peak = float(np.max(np.abs(a)))
    return (peak, peak <= MAX_LATERAL_ACCEL_MPS2)


def apply_perturbation(
    tracklets: pd.DataFrame,
    object_id: str,
    spec: PerturbationSpec,
    frame_rate_hz: float,
) -> pd.DataFrame:
    """Return a new DataFrame with the perturbation applied to one object's tracklet."""
    df = tracklets.copy()
    mask = df["object_id"] == object_id
    if not mask.any():
        raise ValueError(f"object_id {object_id!r} not present in tracklets")

    obj = df[mask].sort_values("frame_idx").reset_index()
    frames = obj["frame_idx"].to_numpy()
    start = spec.start_frame if spec.start_frame is not None else int(frames.min())
    end = spec.end_frame if spec.end_frame is not None else int(frames.max())

    dt = 1.0 / frame_rate_hz
    ramp_frames = max(1, int(round(spec.ramp_seconds * frame_rate_hz)))
    # Build per-row ramp weight aligned with frames.
    t_norm = (frames - start) / ramp_frames
    weights = _cosine_ramp(t_norm)
    # Zero out anything outside [start, end].
    weights = np.where((frames >= start) & (frames <= end), weights, 0.0)

    if spec.kind == "lateral_offset":
        offsets = weights * spec.magnitude
        peak_a, ok = _check_lateral_accel(offsets, dt)
        if not ok:
            print(f"[warn] lateral accel peak {peak_a:.2f} m/s^2 exceeds "
                  f"{MAX_LATERAL_ACCEL_MPS2}; consider increasing ramp_seconds",
                  file=sys.stderr)
        df.loc[obj["index"], "cy"] = obj["cy"].to_numpy() + offsets

    elif spec.kind == "speed_delta":
        # Re-integrate longitudinal position from per-frame velocity.
        # Approach: estimate per-frame dx from consecutive cx samples, scale dx
        # by (1 + magnitude*weight), then re-accumulate from the first frame.
        cx = obj["cx"].to_numpy()
        if len(cx) < 2:
            # Single-frame tracklet — speed has no meaning here. Skip rather than
            # crash, since perturb may be applied in batched discovery loops.
            print(f"[warn] speed_delta on {object_id!r} skipped: only {len(cx)} frame(s)",
                  file=sys.stderr)
        else:
            dx = np.diff(cx, prepend=cx[0])
            scaled_dx = dx * (1.0 + spec.magnitude * weights)
            new_cx = np.cumsum(scaled_dx) + (cx[0] - scaled_dx[0])
            # Long-accel sanity check needs ≥3 points for np.gradient.
            if len(new_cx) >= 3:
                v = np.gradient(new_cx, dt)
                a = np.gradient(v, dt)
                peak_la = float(np.max(np.abs(a)))
                if peak_la > MAX_LONG_ACCEL_MPS2:
                    print(f"[warn] longitudinal accel peak {peak_la:.2f} m/s^2 exceeds "
                          f"{MAX_LONG_ACCEL_MPS2}; consider smaller magnitude or longer ramp",
                          file=sys.stderr)
            df.loc[obj["index"], "cx"] = new_cx

    elif spec.kind == "yaw_bias":
        yaw_offsets = weights * spec.magnitude
        df.loc[obj["index"], "heading"] = obj["heading"].to_numpy() + yaw_offsets

    else:
        raise ValueError(f"unknown perturbation kind: {spec.kind!r}")

    # Mark perturbed rows. Use the index from the original df, only where weight > 0.
    perturbed_mask = mask & df["frame_idx"].isin(frames[weights > 0])
    df.loc[perturbed_mask, "perturbed"] = True
    df.loc[perturbed_mask, "perturb_kind"] = spec.kind
    return df


def _load_manifest(intermediate_dir: Path) -> dict:
    with open(intermediate_dir / "manifest.json") as f:
        return json.load(f)


def perturb_segment(
    intermediate_dir: Path,
    out_dir: Path,
    object_id: str,
    spec: PerturbationSpec,
) -> Path:
    """Copy a segment dir, overwrite tracklets.parquet with the perturbed version.

    Frames and calibrations are unchanged — Cosmos uses tracklets+calibs to build
    the conditioning wireframe; the *original* RGB frames are still the source.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Copy everything except tracklets.parquet (we'll overwrite that).
    for entry in intermediate_dir.iterdir():
        if entry.name == "tracklets.parquet":
            continue
        dest = out_dir / entry.name
        if dest.exists():
            continue
        if entry.is_dir():
            shutil.copytree(entry, dest)
        else:
            shutil.copy2(entry, dest)

    manifest = _load_manifest(intermediate_dir)
    fps = manifest.get("frame_rate_hz", 10.0)

    tracklets = pq.read_table(intermediate_dir / "tracklets.parquet").to_pandas()
    perturbed = apply_perturbation(tracklets, object_id, spec, fps)

    table = pa.Table.from_pandas(perturbed, schema=TRACKLET_SCHEMA, preserve_index=False)
    pq.write_table(table, out_dir / "tracklets.parquet")

    # Drop a sidecar so we can audit what was changed.
    audit = {
        "source_segment": manifest.get("segment_name"),
        "object_id": object_id,
        "spec": spec.__dict__,
        "frame_rate_hz": fps,
    }
    with open(out_dir / "perturbation.json", "w") as f:
        json.dump(audit, f, indent=2)

    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("intermediate_dir", type=Path,
                    help="A waymo_loader output dir (contains tracklets.parquet)")
    ap.add_argument("--object-id", required=True,
                    help="Waymo track ID to perturb (string from laser_labels.id)")
    ap.add_argument("--kind", required=True,
                    choices=["lateral_offset", "speed_delta", "yaw_bias"])
    ap.add_argument("--magnitude", type=float, required=True)
    ap.add_argument("--ramp-seconds", type=float, default=2.0)
    ap.add_argument("--start-frame", type=int, default=None)
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = PerturbationSpec(
        kind=args.kind,
        magnitude=args.magnitude,
        ramp_seconds=args.ramp_seconds,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    out = perturb_segment(args.intermediate_dir, args.out, args.object_id, spec)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

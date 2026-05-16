"""Wrapper around Cosmos-Drive-Dreams' rendering scripts.

Two entry points the team will need to call:
  - render_video(rds_hq_dir, prompt, out_dir)        → MP4 (front camera, generated)
  - render_lidar(rds_hq_dir, out_dir)                → .pcd / .npz LiDAR (optional stretch)

Both shell out to scripts in the Cosmos-Drive-Dreams clone. We capture stdout/stderr
to a log file, return the produced artifact path. Failure surfaces as an exception
with the last 40 lines of log so callers can decide whether to fall back.

Configure the path to the clone via env var COSMOS_REPO or config.COSMOS_REPO.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import config


# Script names inside the Cosmos-Drive-Dreams repo. Confirm + adjust against
# the actual repo when it's cloned on the GX10.
VIDEO_SCRIPT = "scripts/render_from_rds_hq.py"
LIDAR_SCRIPT = "scripts/render_lidargen.py"          # stretch goal — Hour-17 gate


@dataclass
class CosmosResult:
    artifact_path: Path
    log_path: Path
    elapsed_seconds: float
    returncode: int


def _check_repo() -> Path:
    repo = Path(config.COSMOS_REPO)
    if not repo.exists():
        raise FileNotFoundError(
            f"Cosmos-Drive-Dreams repo not found at {repo}. "
            f"Clone it or set COSMOS_REPO env var."
        )
    return repo


def _run(cmd: list[str], log_path: Path, cwd: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as logf:
        logf.write(f"$ cd {cwd}\n$ {' '.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode, time.time() - t0


def _tail(path: Path, n: int = 40) -> str:
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return "(no log)"
    return "\n".join(lines[-n:])


def render_video(
    rds_hq_dir: Path,
    prompt: str,
    out_dir: Path,
    extra_args: list[str] | None = None,
) -> CosmosResult:
    repo = _check_repo()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "cosmos_video.log"

    cmd = [
        sys.executable, str(repo / VIDEO_SCRIPT),
        "--rds_hq_dir", str(rds_hq_dir),
        "--prompt", prompt,
        "--output_dir", str(out_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)

    rc, elapsed = _run(cmd, log_path, cwd=repo)
    if rc != 0:
        raise RuntimeError(
            f"cosmos video render failed (rc={rc}). Tail:\n{_tail(log_path)}"
        )
    # Find the produced MP4. Cosmos writes one per camera, take the front by default.
    mp4s = sorted(out_dir.glob("*.mp4"))
    if not mp4s:
        raise RuntimeError(
            f"cosmos exited 0 but produced no MP4 in {out_dir}. Tail:\n{_tail(log_path)}"
        )
    return CosmosResult(mp4s[0], log_path, elapsed, rc)


def render_lidar(
    rds_hq_dir: Path,
    out_dir: Path,
    extra_args: list[str] | None = None,
) -> CosmosResult:
    repo = _check_repo()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "cosmos_lidar.log"

    cmd = [
        sys.executable, str(repo / LIDAR_SCRIPT),
        "--rds_hq_dir", str(rds_hq_dir),
        "--output_dir", str(out_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)

    rc, elapsed = _run(cmd, log_path, cwd=repo)
    if rc != 0:
        raise RuntimeError(
            f"cosmos lidar render failed (rc={rc}). Tail:\n{_tail(log_path)}"
        )
    artifacts = sorted(out_dir.glob("*.pcd")) + sorted(out_dir.glob("*.npz"))
    if not artifacts:
        raise RuntimeError(f"lidar render produced no artifact in {out_dir}")
    return CosmosResult(artifacts[0], log_path, elapsed, rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rds_hq_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["video", "lidar"], default="video")
    ap.add_argument("--prompt", type=str, default="A clear sunny day on a city street")
    args = ap.parse_args()

    if args.mode == "video":
        result = render_video(args.rds_hq_dir, args.prompt, args.out)
    else:
        result = render_lidar(args.rds_hq_dir, args.out)
    print(json.dumps({
        "artifact": str(result.artifact_path),
        "log": str(result.log_path),
        "elapsed_seconds": result.elapsed_seconds,
    }, indent=2))


if __name__ == "__main__":
    main()

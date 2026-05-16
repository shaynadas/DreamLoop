# DreamLoop Pipeline — Integration Guide

How everyone else's code plugs into the pipeline modules. Read this first
before touching `streamlit_app.py`, the discovery loop, or anything that
calls into `pipeline/`.

---

## 1. Layout

```
DreamLoop/
├── pipeline/                      ← MERGE-SAFE: this is all one team's work
│   ├── config.py                  paths + constants
│   ├── schema.py                  parquet schemas + dataclasses
│   ├── waymo_loader.py            .tfrecord → intermediate dir
│   ├── tracklet_perturb.py        perturbation primitives
│   ├── box_projection.py          3D ego → 2D image pixels
│   ├── rds_writer.py              intermediate → RDS-HQ + wireframes
│   ├── cosmos_runner.py           subprocess wrapper for Cosmos scripts
│   ├── yolo_eval.py               mAP@0.5 vs tracklet-derived GT
│   ├── INTEGRATION.md             ← you are here
│   ├── README.md (will move up)   user-facing repo readme
│   ├── requirements.txt           Python deps
│   └── tests/
│       ├── synth.py               build a fake intermediate dir
│       ├── test_pipeline.py       happy-path end-to-end
│       ├── test_edge_cases.py     adversarial cases (short tracklets, etc.)
│       ├── test_primitives.py     IoU math, MP4 extraction, runner shape
│       └── run_all.sh             one-shot runner
```

All cross-file imports are flat — `from schema import ...`, `from config import ...`.
**Always run commands from `DreamLoop/pipeline/`.** Don't add a `pipeline.` prefix
to imports; if you need to import these modules from a script outside this folder,
prepend `pipeline/` to `sys.path` like `tests/*.py` does:

```python
import sys
sys.path.insert(0, "/path/to/DreamLoop/pipeline")
from waymo_loader import extract_segment
```

---

## 2. Install

### Mac dev (no GPU)

```bash
cd DreamLoop/pipeline
python3.11 -m venv .venv          # 3.13 / 3.14 also fine; 3.11 chosen for TF compat
source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, the `waymo-open-dataset-tf-2-12-0` wheel does NOT exist. You
can either (a) skip `waymo_loader` locally and run it on the GX10 only — which
is what we plan to do, or (b) install from source. The rest of the pipeline
runs fine on Mac without TF.

### GX10 (rendering + heavy work)

```bash
ssh gx10
# clone both
git clone https://github.com/<your-org>/DreamLoop.git
git clone https://github.com/nv-tlabs/Cosmos-Drive-Dreams.git
cd DreamLoop/pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export COSMOS_REPO=$HOME/Cosmos-Drive-Dreams
export DREAMLOOP_DATA=/scratch/dreamloop_data      # big-disk path
```

---

## 3. End-to-end pipeline (canonical command sequence)

All commands run from `DreamLoop/pipeline/` with `.venv` active.

```bash
# (0) Sanity: every fast test should pass.
bash tests/run_all.sh

# (1) Decode one Waymo segment → intermediate dir.
python waymo_loader.py /scratch/waymo/segment-XYZ.tfrecord \
    --cameras FRONT --max-frames 200
# → $DREAMLOOP_DATA/waymo_intermediate/segment-XYZ/

# (2) Perturb one tracklet's trajectory.
python tracklet_perturb.py $DREAMLOOP_DATA/waymo_intermediate/segment-XYZ \
    --object-id <waymo_track_id>                     \
    --kind lateral_offset --magnitude 1.75            \
    --ramp-seconds 2.0                                 \
    --out $DREAMLOOP_DATA/waymo_intermediate/segment-XYZ__lat175

# (3) Build the RDS-HQ-shaped dir Cosmos consumes (renders wireframes too).
python rds_writer.py $DREAMLOOP_DATA/waymo_intermediate/segment-XYZ__lat175 \
    --out $DREAMLOOP_DATA/rds_hq/segment-XYZ__lat175

# (4) Render with Cosmos (GX10 only — needs Blackwell + Cosmos weights).
python cosmos_runner.py $DREAMLOOP_DATA/rds_hq/segment-XYZ__lat175 \
    --mode video                                                     \
    --prompt "Heavy night blizzard, snow accumulating on the road"    \
    --out $DREAMLOOP_DATA/cosmos_outputs/segment-XYZ__lat175__blizzard

# (5) Evaluate YOLO on the Cosmos MP4 vs the PERTURBED ground truth.
python yolo_eval.py \
    --mp4 $DREAMLOOP_DATA/cosmos_outputs/segment-XYZ__lat175__blizzard/generated_front.mp4 \
    --intermediate-dir $DREAMLOOP_DATA/waymo_intermediate/segment-XYZ__lat175             \
    --out $DREAMLOOP_DATA/cosmos_outputs/segment-XYZ__lat175__blizzard/eval.json
```

**The single most common mistake**: passing the *original* intermediate dir to
yolo_eval when the rendered video came from a *perturbed* trajectory. GT must
reflect what Cosmos was conditioned on. The script's `--intermediate-dir` flag
docstring says this — heed it.

---

## 4. How each pivot plugs in

### Pivot A — Data Factory (one input → N weather variants)

A Streamlit (or any) front-end calls modules in this order, per user click:

```python
from waymo_loader import extract_segment          # one-time
from tracklet_perturb import perturb_segment, PerturbationSpec
from rds_writer import build_rds_hq_dir
from cosmos_runner import render_video

# preload at startup
INT_DIR = extract_segment(Path("seg.tfrecord"), Path("data/wi/seg"))

# per click
spec = PerturbationSpec(kind="lateral_offset", magnitude=1.75, ramp_seconds=2.0)
pert = perturb_segment(INT_DIR, Path("data/wi/seg__p1"), "veh_001", spec)
rds  = build_rds_hq_dir(pert, Path("data/rds/seg__p1"), cameras=["FRONT"])
out  = render_video(rds, "<weather prompt>", Path("data/cosmos/seg__p1__rain"))
```

For the live demo, **pre-render the 4 (or N) combinations offline** and have
the UI read from `data/cosmos/...` directories that already exist. Cosmos
inference is minutes per clip, not seconds.

### Pivot B — Discovery Engine (offline grid + ranked failure list)

A separate `discover.py` (not yet written) wraps the same modules:

```python
# 1. Define a grid of (perturbation, weather) pairs.
GRID = [
    ("veh_001", PerturbationSpec("lateral_offset", 1.0, 2.0),  "night blizzard"),
    ("veh_001", PerturbationSpec("lateral_offset", 1.5, 2.0),  "night blizzard"),
    ("veh_001", PerturbationSpec("lateral_offset", 2.0, 2.0),  "night blizzard"),
    ("veh_001", PerturbationSpec("speed_delta", -0.5, 1.0),    "dense fog"),
    ...
]

# 2. For each, build RDS-HQ, render, eval. Store mAP per cell.
results = []
for object_id, spec, prompt in GRID:
    pert = perturb_segment(INT_DIR, ..., object_id, spec)
    rds  = build_rds_hq_dir(pert, ...)
    mp4  = render_video(rds, prompt, ...).artifact_path
    report = evaluate(extract_frames_from_mp4(mp4, ...), pert)
    results.append({"spec": spec, "prompt": prompt, "map50": report["map50"]})

# 3. Sort by mAP ascending — top N are the "discovered" failure modes.
results.sort(key=lambda r: r["map50"])
```

The Streamlit UI then iterates through `results` and animates the search as
if it were live.

---

## 5. Schemas — what's in each parquet

### `tracklets.parquet` (per intermediate dir)

| column          | type    | meaning                                            |
|-----------------|---------|----------------------------------------------------|
| object_id       | string  | Waymo track ID, stable within a segment             |
| frame_idx       | int32   | 0-indexed frame                                     |
| type            | int32   | WaymoLabelType: 1=VEH, 2=PED, 3=SIGN, 4=CYCLIST     |
| cx, cy, cz      | float64 | 3D box center, ego frame (+x fwd, +y left, +z up)  |
| length          | float64 | extent along heading                                |
| width           | float64 | extent perpendicular to heading, xy plane           |
| height          | float64 | extent along z                                      |
| heading         | float64 | yaw around +z, radians                              |
| speed_x         | float64 | ego-frame velocity, NaN if Waymo didn't provide it  |
| speed_y         | float64 | same                                                |
| num_lidar_points| int32   | LiDAR-point confidence proxy; 0 = weak label        |
| perturbed       | bool    | set True by tracklet_perturb for changed rows       |
| perturb_kind    | string  | "lateral_offset" / "speed_delta" / "yaw_bias" / None|

### `ego_poses.parquet`

| column       | type           | meaning                                  |
|--------------|----------------|------------------------------------------|
| frame_idx    | int32          | 0-indexed frame                          |
| timestamp_us | int64          | microseconds, monotonic per segment      |
| transform    | list<f64, 16>  | 4x4 world→ego, row-major flattened       |

### `calibrations.json`

```json
{
  "FRONT": {
    "width": 1920, "height": 1280,
    "intrinsic": [fx, fy, cx, cy, k1, k2, p1, p2, k3],
    "extrinsic": [[4x4 row-major vehicle→camera transform]]
  },
  "FRONT_LEFT": { ... }
}
```

### RDS-HQ output (`rds_writer` writes this)

```
<rds_hq_dir>/
├── rgb/FRONT/00000.png ...           original RGB
├── wireframe/FRONT/00000.png ...     3D box wireframe on black canvas
├── labels/
│   ├── tracklets.parquet             possibly perturbed
│   └── ego_poses.parquet
├── calibrations.json
├── _rds_hq_manifest.json             DreamLoop-controlled metadata
└── perturbation.json                 (if intermediate was perturbed)
```

---

## 6. Known gaps you'll hit on the GX10

These are **not bugs** — they are deliberate stubs flagged for the first hour
on the GX10 once we can read the real Cosmos repo source.

1. **`rds_writer.DEFAULT_LAYOUT`** — best-guess directory/file naming. Check
   against `scripts/render_from_rds_hq.py` in the Cosmos repo. Search the
   repo for `rds_hq_dir` and adjust the layout dict. Anything in the
   `_rds_hq_manifest.json` is ours, not theirs — change at will.

2. **`cosmos_runner.VIDEO_SCRIPT` / `LIDAR_SCRIPT`** — script paths inside
   the Cosmos repo. We guessed `scripts/render_from_rds_hq.py` and
   `scripts/render_lidargen.py`. Confirm and patch before first GPU run.

3. **`cosmos_runner` CLI args** — we pass `--rds_hq_dir`, `--prompt`,
   `--output_dir`. Cosmos's real flags may be `--input-dir / --text-prompt /
   --output-path` or similar. First failed run will reveal it; the wrapper
   captures stdout to `cosmos_*.log` for inspection.

4. **YOLO ↔ Waymo class map** (`config.COCO_TO_WAYMO`) doesn't include every
   COCO vehicle subtype. Construction vehicles, trailers, and emergency
   vehicles are silently dropped. Fine for the demo. Flag in slide notes
   if asked.

5. **`waymo_loader` on Apple Silicon** — the Waymo SDK wheel is not built
   for arm64. Run loader on GX10 only.

6. **Cosmos video has no audio channel; YOLO doesn't care.** No action needed.

---

## 7. Running the tests

From `DreamLoop/pipeline/`:

```bash
bash tests/run_all.sh
# or individually:
python -m tests.test_pipeline       # 14 happy-path checks
python -m tests.test_edge_cases     # 11 adversarial checks
python -m tests.test_primitives     # 7 IoU + MP4 + runner-shape checks
```

These tests do NOT need:
- Waymo data (uses synthesized intermediate)
- A GPU (no Cosmos, no YOLO model load)
- The Cosmos repo (only verifies the wrapper raises on missing repo)

They DO catch:
- Parquet schema regressions (e.g. None vs NaN in `perturb_kind`)
- Single-frame tracklet edge case (regression already caught + fixed)
- Coordinate frame errors (Waymo→OpenCV axis swap)
- IoU math errors
- Perturbation flag round-trips

If you change anything in `pipeline/`, **run the full suite before pushing.**

---

## 8. Coordinate frames — read this before touching projection

Three frames, three conventions:

1. **Waymo ego (vehicle) frame** — `+x forward, +y left, +z up`. All
   tracklet rows live here.

2. **Waymo camera frame** — same axes as ego, but origin at the camera
   sensor. `extrinsic` in `calibrations.json` is `vehicle → waymo_camera`.

3. **OpenCV pinhole frame** — `+x right, +y down, +z forward`. Required
   for `K @ [x/z, y/z, 1]` to produce sensible image pixels.

The axis-swap matrix is in `box_projection._WAYMO_CAM_TO_OPENCV`. Don't
re-derive it casually — verify with a known-front-of-camera point first.
Tests `t_projection_front_object_on_image` and `t_projection_behind_camera_drops`
guard this; if either breaks, your swap is wrong.

---

## 9. Quick reference: what each module imports

| Module               | Imports from pipeline |
|----------------------|-----------------------|
| `config.py`          | (none)                |
| `schema.py`          | (none)                |
| `box_projection.py`  | `schema`              |
| `waymo_loader.py`    | `config`, `schema`    |
| `tracklet_perturb.py`| `schema`              |
| `rds_writer.py`      | `config`, `box_projection` |
| `cosmos_runner.py`   | `config`              |
| `yolo_eval.py`       | `config`, `box_projection` |

No cycles. Adding a new module? Match this — never import upward from a
"lower" module into a "higher" one.

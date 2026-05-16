# DreamLoop — Implementation Guide for ASUS Ascent GX10 (DGX Spark)

End-to-end setup of the `main` branch on a fresh ASUS Ascent GX10 (NVIDIA Grace Blackwell, aarch64 Ubuntu 22.04). Every step has a verification command. Every known error has a fix in Appendix A. Read this top-to-bottom before starting; it will save you the 17 errors we already debugged.

---

## 0. What's on this branch and what each file does

| File | Purpose | Inputs | Outputs |
|---|---|---|---|
| `inject.py` | Mathematically injects a synthetic hazard (pedestrian/collision/jaywalker/bike) into per-frame label JSONs. Renders a bird's-eye-view MP4 + YOLO-format labels. | A directory of `*.all_object_info.json` files (one per frame, from NVIDIA Cosmos data) | Modified JSONs + `boxes.json` + `<scenario>.mp4` + `<scenario>_labels.pt` |
| `gen.py` | Wraps a Helios-Distilled V2V model call to add weather (blizzard/rain) to a dashcam video. **Critical: shells out to `infer_helios.py` which does not exist in this repo.** | `--input_video` MP4 + scenario flags | Weather-altered MP4 (when `infer_helios.py` is provided) |
| `json_to_yolo.py` | Converts per-frame JSON labels to YOLO `.txt` files. | A `boxes.json`-style file with nested `{frame_id, boxes:[{class_id,x,y,w,h}]}` format | One `frame_NNNN.txt` per frame |
| `prepare_yolo_data.py` | Pairs frames from a "dream" video with YOLO labels into a training dataset. | A dream MP4 + a folder of baseline `.txt` labels | `images/train/` + `labels/train/` under `./dreamloop_dataset/` |
| `train_yolo.py` | Fine-tunes YOLOv8n on the prepared dataset. | `data.yaml` + the `./dreamloop_dataset/` produced above | `runs/train/exp/` with fine-tuned weights |
| `data.yaml` | YOLO dataset config; maps to COCO classes 0/1/2/3/5/7. | n/a | n/a |
| `master_pipeline.py` | Orchestrator. Chains all five stages as `subprocess.run` calls. | `--frames_dir` + `--baseline_video` + `--scenario` | All artifacts above, plus fine-tuned weights |
| `agent.py` | Demo console — runs `gen.py` in `--dry-run` mode and prints a scripted narrative. Not part of the production pipeline. | n/a | stdout only |
| `boxes.json` / `boxes_pedestrian.json` / `boxes_collision.json` | Pre-computed scenario trajectories. **Caveat: all three files are byte-identical despite the names; only the pedestrian scenario is actually exported.** | n/a | n/a |

**This branch does not include:** `requirements.txt`, `setup.py`, the Cosmos-Drive-Dreams renderer (separate repo), Helios model weights, fine-tuned YOLO weights, or any input video.

---

## 1. Hardware + OS prerequisites

- **Hardware**: ASUS Ascent GX10 (NVIDIA Grace Blackwell, aarch64). The Waymo Open Dataset SDK does not ship aarch64 wheels — this guide therefore does not use Waymo data; it uses NVIDIA Cosmos RDS-HQ data instead.
- **OS**: Ubuntu 22.04 LTS (default on the GX10).
- **Python**: 3.10 required. Python 3.11+ breaks the Waymo SDK and TensorFlow 2.12; 3.9 breaks newer NumPy/Pandas wheels. Use exactly 3.10.
- **Disk**: At least 100 GB free under `~/Desktop/`. Cosmos data, model weights, and intermediate frames are large.
- **Network**: Outbound HTTPS for Hugging Face, GitHub, and PyPI.
- **Accounts**:
  - GitHub (read access to `shaynadas/DreamLoop`)
  - Hugging Face (read access to `nvidia/Cosmos-Drive-Dreams` — public dataset, just needs `hf auth login`)

Verify hardware + OS:

```bash
uname -m                              # expect: aarch64
lsb_release -d                        # expect: Ubuntu 22.04.x LTS
nvidia-smi                            # expect: a Grace Blackwell GPU + driver info
python3.10 --version                  # expect: Python 3.10.x
df -h ~/Desktop                       # confirm ≥ 100 GB available
```

If any of these fail, fix them before proceeding. Don't attempt the rest of the guide on a wrong-arch or wrong-Python box.

---

## 2. System-level dependencies

Several Python wheels need C/C++ headers and X11 libraries that Ubuntu does not ship by default. Install all of them up front to avoid the build-failure detour we did the first time around:

```bash
sudo apt update
sudo apt install -y \
    python3.10 python3.10-venv python3.10-dev \
    build-essential \
    libx11-dev \
    libglu1-mesa-dev \
    libosmesa6-dev \
    ffmpeg \
    git \
    git-lfs \
    curl \
    wget
git lfs install --skip-repo
```

Why each one:
- `python3.10-dev` → provides `Python.h`. Without it, `pip install moderngl` fails with `fatal error: Python.h: No such file or directory`.
- `libx11-dev` → provides `X11/Xlib.h`. Without it, `pip install glcontext` (a `moderngl` dep) fails.
- `libosmesa6-dev` + `libglu1-mesa-dev` → offscreen OpenGL for `moderngl` to render without a display.
- `git-lfs` → required to fetch Cosmos's `assets/example/*.tar` files. Without it, those files are 130-byte text pointers, not actual tar archives.
- `ffmpeg` → re-encoding to H.264 for browser-playable MP4s.

Verify:

```bash
python3.10 -c "import sysconfig; print(sysconfig.get_path('include'))"
ls /usr/include/X11/Xlib.h            # must exist
git lfs version                       # must report a version
ffmpeg -version | head -1
```

---

## 3. Clone the repo + create the venv

```bash
mkdir -p ~/Desktop && cd ~/Desktop
git clone git@github.com:shaynadas/DreamLoop.git
cd DreamLoop
git checkout main
git lfs pull                          # in case any future LFS asset is added
```

Create an isolated Python 3.10 venv (do not share with Cosmos's venv — torch versions diverge):

```bash
python3.10 -m venv dreamloop_env
source dreamloop_env/bin/activate
pip install --upgrade pip wheel setuptools
```

Verify:

```bash
which python                          # must point inside dreamloop_env
python --version                      # must be 3.10.x
```

---

## 4. Python dependencies for the `main` pipeline

There is no `requirements.txt` on `main`. Install the exact set of packages each script imports:

```bash
# Core (all scripts)
pip install numpy opencv-python-headless tqdm pillow

# inject.py
pip install torch                      # CPU-only torch is fine on Mac dev;
                                       # on GX10, use CUDA torch (see below).

# json_to_yolo.py / prepare_yolo_data.py / train_yolo.py
pip install ultralytics                # pulls torchvision + YOLOv8 automatically

# master_pipeline.py / agent.py — stdlib only, no extras needed
```

**Important — torch on the GX10**: Grace Blackwell uses CUDA 13.x. The default PyPI `torch` wheels may be CPU-only on aarch64. For GPU acceleration, install NVIDIA's wheels first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

If that 404s (wheels not yet available for your CUDA version), fall back to:

```bash
pip install torch torchvision           # CPU fallback; YOLO training will be slow but functional
```

Verify everything imports without errors:

```bash
python -c "import numpy, cv2, tqdm, PIL, torch, ultralytics; print('all imports OK')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If CUDA is `False`, training works but is slow. If CUDA is `True`, you're good.

---

## 5. Acquire input data (NVIDIA Cosmos RDS-HQ clip)

The `inject.py` script reads `*.all_object_info.json` files. These come from NVIDIA's Cosmos-Drive-Dreams dataset on Hugging Face. **Do not try to use Waymo data** — the Waymo SDK has no aarch64 wheel.

### 5a. Clone the Cosmos-Drive-Dreams toolkit (in a separate directory)

```bash
cd ~/Desktop
git clone https://github.com/nv-tlabs/Cosmos-Drive-Dreams.git
cd Cosmos-Drive-Dreams
git lfs pull                          # essential — the example assets are LFS-managed
file assets/example/all_object_info/*.tar    # sanity check; must say "POSIX tar archive"
                                              # NOT "ASCII text" (the LFS pointer)
```

If `file` reports `ASCII text`, the LFS pull silently failed. Re-run `git lfs pull` and verify.

### 5b. Authenticate to Hugging Face

The Cosmos download script uses the modern `hf` CLI (NOT the deprecated `huggingface-cli`):

```bash
pip install --upgrade huggingface_hub
hf auth login
# Paste your HF token when prompted.
```

Verify:

```bash
hf auth whoami                        # must print your HF username
```

### 5c. Download exactly one clip

The dataset is ~5,843 clips, several terabytes. **Always use `--limit 1`** for hackathon work or you will fill the disk and waste hours.

```bash
cd ~/Desktop/Cosmos-Drive-Dreams/cosmos-drive-dreams-toolkits
python download.py \
    --odir ./data \
    --file_types all_object_info pose vehicle_pose pinhole_intrinsic ftheta_intrinsic 3d_lanes 3d_lanelines 3d_road_boundaries 3d_road_markings 3d_traffic_lights 3d_traffic_signs 3d_poles 3d_crosswalks 3d_wait_lines captions lidar_raw lidar_sensor_config car_mask_coarse \
    --workers 4 \
    --limit 1
```

This pulls ~10 GB. Takes 10–30 minutes depending on bandwidth.

Verify:

```bash
ls ~/Desktop/Cosmos-Drive-Dreams/cosmos-drive-dreams-toolkits/data/all_object_info/
# Must show one .tar file with a clip-id name like:
#   2d23a1f4-c269-46aa-8e7d-1bb595d1e421_2445376400000_2445396400000.tar
```

### 5d. Extract the per-frame JSON files

`inject.py` reads individual `.all_object_info.json` files, not the raw tar. The tar is a WebDataset shard — each member is a separate file. Extract:

```bash
cd ~/Desktop/Cosmos-Drive-Dreams/cosmos-drive-dreams-toolkits/data/all_object_info
mkdir -p extracted
tar -xf *.tar -C extracted/
ls extracted/ | head -5
# Expect filenames like:
#   <clip_id>.000000.all_object_info.json
#   <clip_id>.000001.all_object_info.json
```

The exact location of these extracted JSONs becomes your `--data_dir` argument in Step 6.

---

## 6. Patch the Cosmos toolkit (the `pycg.Isometry` fix)

NVIDIA's `cosmos-drive-dreams-toolkits/utils/bbox_utils.py` imports `from pycg import Isometry`. **The `pycg` package on public PyPI is name-squatted and unrelated** to NVIDIA's internal `pycg`. Installing PyPI's `pycg` makes the import succeed but `Isometry` doesn't exist there.

The fix is to replace the `Isometry` usage with a NumPy SVD-based orthogonalization. Apply this patch:

```bash
cd ~/Desktop/Cosmos-Drive-Dreams/cosmos-drive-dreams-toolkits/utils

# 1. Back up the original.
cp bbox_utils.py bbox_utils.py.orig

# 2. Add the helper function near the top imports (after `import numpy as np`):
python3 << 'EOF'
import re
src = open("bbox_utils.py").read()

helper = '''
def _ortho_matrix(m):
    """SVD-based orthogonalization. Replaces pycg.Isometry whose internal
    package is not the same as PyPI's pycg (which is name-squatted)."""
    import numpy as _np
    R = m[:3, :3]
    U, _, Vt = _np.linalg.svd(R)
    R_ortho = U @ Vt
    result = m.copy()
    result[:3, :3] = R_ortho
    return result

'''

# Inject the helper after the first numpy import.
src = re.sub(r"(import numpy as np\n)", r"\1" + helper, src, count=1)

# Replace pycg.Isometry(tfm).matrix with our helper.
src = src.replace("pycg.Isometry(tfm).matrix", "_ortho_matrix(tfm)")

# Remove the unused pycg import to prevent ImportError at module load.
src = re.sub(r"(^|\n)\s*from pycg import Isometry\s*\n", "\n", src)
src = re.sub(r"(^|\n)\s*import pycg\s*\n", "\n", src)

open("bbox_utils.py", "w").write(src)
print("patched bbox_utils.py")
EOF

# 3. Verify.
diff bbox_utils.py.orig bbox_utils.py | head -30
python3 -c "import sys; sys.path.insert(0, '.'); import bbox_utils; print('bbox_utils imports cleanly')"
```

Note: this is only needed if you plan to run Cosmos's renderer (`render_from_rds_hq.py`). The `main` branch's `inject.py` does **not** require this patch — it only modifies JSON labels.

---

## 7. Run `inject.py` standalone

This is the smallest unit you can verify. It does not need a GPU, Helios, YOLO, or Cosmos's renderer.

```bash
cd ~/Desktop/DreamLoop
source dreamloop_env/bin/activate

# Substitute your extracted JSON dir from Step 5d.
DATA_DIR=~/Desktop/Cosmos-Drive-Dreams/cosmos-drive-dreams-toolkits/data/all_object_info/extracted
OUT_DIR=./injected_pedestrian

python inject.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUT_DIR" \
    --scenario pedestrian_cross
```

Expected runtime: under 30 seconds for a single 297-frame clip.

Verify outputs:

```bash
ls "$OUT_DIR"
# Expect:
#   *.all_object_info.json    (one per input frame, with the pedestrian inserted)
#   boxes.json                (trajectory summary)
#   pedestrian_cross.mp4      (top-down bird's-eye visualization, ~10 sec at 30fps)
#   pedestrian_cross_labels.pt    (PyTorch tensor: frame, class_id, cx, cy, bw, bh)

# Inspect the MP4 (skip if no display):
ffprobe -v error -show_entries stream=width,height,nb_frames -of csv=p=0 "$OUT_DIR/pedestrian_cross.mp4"
# Expect: 800,800,297 (or similar — 800x800 canvas, ~297 frames)
```

Run the other three scenarios as smoke tests:

```bash
for scenario in collision jaywalker bike; do
    python inject.py \
        --data_dir "$DATA_DIR" \
        --output_dir "./injected_$scenario" \
        --scenario "$scenario"
done
```

If any scenario fails, the issue is in `inject.py`'s scenario configs (lines 132–167); the data path is already validated.

**Known data quality issue**: the three pre-shipped `boxes_*.json` files at the repo root are byte-identical (all contain the pedestrian scenario despite the names). If you want a real `boxes_collision.json`, generate it yourself with the command above and overwrite. This is filed as a bug but not blocking.

---

## 8. Wire up `gen.py` (Helios) — known incomplete

`gen.py` shells out to `infer_helios.py`, which **does not exist in this repo**. You have two options:

### Option A: Stub `infer_helios.py` to copy input → output

For demos where weather generation isn't on the critical path, create a passthrough:

```bash
cd ~/Desktop/DreamLoop
cat > infer_helios.py << 'EOF'
"""Stub. Real Helios integration TBD. Copies input to output unchanged."""
import argparse, shutil, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--input_video", required=True)
p.add_argument("--output_dir", required=True)
p.add_argument("--prompt", default="")
p.add_argument("--scenario", default="blizzard")
p.add_argument("--intensity", type=int, default=10)
p.add_argument("--time_of_day", default="night")
p.add_argument("--dry_run", action="store_true")
args = p.parse_args()

out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
dst = out / f"{args.scenario}.mp4"

if args.dry_run:
    print(f"[stub] would generate {dst} with prompt={args.prompt}")
    sys.exit(0)

shutil.copy(args.input_video, dst)
print(f"[stub] copied {args.input_video} → {dst}")
EOF
```

### Option B: Install the real Helios-Distilled model

This requires a working HF login (Step 5b), GPU memory, and a separate set of deps. Out of scope for the floor; pursue only if the floor demo is stable.

Verify the stub:

```bash
mkdir -p ./data && cp <any-mp4> ./data/baseline.mp4
python gen.py --scenario blizzard --intensity 10 --time_of_day night \
              --input_video ./data/baseline.mp4 \
              --output_dir ./outputs/dream_run \
              --dry-run
# Should print the stub's dry-run line and exit 0.
```

---

## 9. Run the full pipeline via `master_pipeline.py`

This is the orchestrator. It chains: `inject.py` → `gen.py` → `json_to_yolo.py` → `prepare_yolo_data.py` → `train_yolo.py`.

### 9a. Layout the expected directory structure

```bash
cd ~/Desktop/DreamLoop

# master_pipeline.py expects:
#   ./data/frames           (extracted Cosmos JSONs)
#   ./data/baseline.mp4     (input dashcam video)

mkdir -p ./data
ln -sf ~/Desktop/Cosmos-Drive-Dreams/cosmos-drive-dreams-toolkits/data/all_object_info/extracted ./data/frames
cp /path/to/your/baseline_dashcam.mp4 ./data/baseline.mp4
# OR if you don't have one yet:
#   curl <a-public-dashcam-mp4-url> -o ./data/baseline.mp4
```

### 9b. Run

```bash
python master_pipeline.py \
    --scenario pedestrian_cross \
    --frames_dir ./data/frames \
    --baseline_video ./data/baseline.mp4
```

This will sequentially:
1. Inject the scenario into the frame JSONs → `./shared_data/`
2. Run `gen.py` to produce a weather-altered MP4 → `./outputs/dream_run/{scenario}.mp4`
3. Convert `boxes.json` → YOLO `.txt` labels in `./data/injected_labels/`
4. Pair the dream MP4 frames with the labels → `./dreamloop_dataset/`
5. Train YOLOv8n for 30 epochs → `runs/train/exp/`

Expected runtime on the GX10:
- Steps 1–3: under 1 minute total
- Step 4: 2–5 minutes (frame extraction)
- Step 5: 15–40 minutes depending on dataset size and GPU availability

### 9c. Verify each stage

After the pipeline finishes:

```bash
# Stage 1: injection
ls ./shared_data/*.all_object_info.json | head -3
ls ./shared_data/boxes.json

# Stage 2: dream video
ls ./outputs/dream_run/*.mp4

# Stage 3: YOLO labels
ls ./data/injected_labels/frame_*.txt | head -3

# Stage 4: prepared dataset
find ./dreamloop_dataset -type f | head -10

# Stage 5: training
ls runs/train/exp/weights/best.pt    # the fine-tuned weights
cat runs/train/exp/results.csv | tail -5    # final epoch metrics
```

If any stage's outputs are missing, isolate that stage and run its script standalone with the same arguments `master_pipeline.py` used (see lines 38–81 of `master_pipeline.py`).

---

## 10. Validation

The pipeline succeeds only if all five stages produced their expected artifacts AND the fine-tuned YOLO weights are usable. Smoke-test the weights:

```bash
python -c "
from ultralytics import YOLO
model = YOLO('runs/train/exp/weights/best.pt')
results = model.predict(source='./data/baseline.mp4', save=True, conf=0.25)
print('predictions saved under runs/detect/')
"
```

If the predictions look reasonable (boxes on people/vehicles in the baseline video), the full pipeline is functional.

---

## Appendix A: Known errors → fixes

Every error we hit during initial setup, in the order we encountered them. If you see one of these, jump to the fix without burning time on debugging.

| # | Error message | Root cause | Fix |
|---|---|---|---|
| 1 | `conda: command not found` | Miniconda not installed. | Skip conda. Use `python3.10 -m venv` instead (this guide). |
| 2 | `cosmos-drive-dreams-toolkits/: No such file or directory` after clone | `git-lfs` not installed, clone partially incomplete. | `sudo apt install git-lfs && git lfs install && git lfs pull` |
| 3 | `git-lfs: command not found` during clone | LFS hooks fired before binary exists. | Install git-lfs first; re-clone. |
| 4 | `brew install git-lfs` builds LLVM for 1-2 hours (Mac only) | Homebrew dep resolution pulls swig → LLVM. | On Mac, skip git-lfs entirely if you only need the downloaded NVIDIA dataset (not the LFS-managed `assets/example`). |
| 5 | `wget: command not found` (Mac) | Mac doesn't ship wget. | `curl -O <url>` is equivalent. |
| 6 | `Warning: huggingface-cli is deprecated` | Old CLI name. | Use `hf auth login` (note: just `hf`, not `huggingface-cli`). |
| 7 | Download started pulling all 5,843 clips | No `--limit` flag passed. | Always pass `--limit 1` for development. |
| 8 | `assets/` folder is empty after clone | LFS objects never pulled. | `cd <repo>; git lfs pull` |
| 9 | `inject.py` FileNotFoundError on `./Cosmos-Drive-Dreams/data/all_object_info` | Wrong relative path. | Pass an absolute path via `--data_dir`. |
| 10 | `inject.py` ignored CLI args, used hardcoded `CLIP_ID` | Earlier version had `CLIP_ID = "0079aad7..."` hardcoded. | Already fixed on `main` — confirm `inject.py` auto-discovers JSON files. |
| 11 | `git add` failed: `not a git repository` | Files created outside the repo dir. | `cd` into the repo first, then `git add`. |
| 12 | `git add ../inject.py` rejected as outside repo | Paths above repo root cannot be staged. | Copy the file into the repo first, then add. |
| 13 | `ModuleNotFoundError: No module named 'pycg'` | NVIDIA's `bbox_utils.py` imports it. | `pip install pycg` (but see #14 — the PyPI package is wrong). |
| 14 | `ImportError: cannot import name 'Isometry' from 'pycg'` | PyPI's `pycg` (v0.0.8) is a different, name-squatted package without `Isometry`. | Apply the SVD-based patch in Step 6 of this guide. |
| 15 | `fatal error: Python.h: No such file or directory` during `pip install moderngl` | Python C headers missing. | `sudo apt install python3.10-dev` |
| 16 | `fatal error: X11/Xlib.h: No such file or directory` during `pip install glcontext` | X11 headers missing. | `sudo apt install libx11-dev` |
| 17 | `git: 'lfs' is not a git command` on the GX10 | git-lfs not installed. | `sudo apt install git-lfs && git lfs install` |
| 18 | `tar: This does not look like a tar archive` on `assets/example/*.tar` | The file is a 130-byte LFS pointer, not the real tar. | `git lfs pull` from the Cosmos repo root. |
| 19 | `AttributeError: 'Namespace' object has no attribute 'output_dir'` (historic) | Argparse declared `--output_dira` (typo) but code used `args.output_dir`. | Already fixed on `main` in commit `4dca0e3`. Confirm `inject.py` line 225 says `--output_dir` (no `a`). |
| 20 | `IndentationError` when running multi-line `python3 -c "..."` from a heredoc | Leading whitespace in the heredoc lines. | Use `python3 << 'EOF' ... EOF` form with no leading indent on Python lines. |

---

## Appendix B: File layout after a successful end-to-end run

```
~/Desktop/
├── DreamLoop/                                     ← this repo
│   ├── implementation.md                          ← you are reading this
│   ├── inject.py
│   ├── gen.py
│   ├── infer_helios.py                            ← stub created in Step 8
│   ├── json_to_yolo.py
│   ├── prepare_yolo_data.py
│   ├── train_yolo.py
│   ├── master_pipeline.py
│   ├── agent.py
│   ├── data.yaml
│   ├── boxes.json / boxes_pedestrian.json / boxes_collision.json
│   ├── dreamloop_env/                             ← venv
│   ├── data/
│   │   ├── frames -> .../Cosmos-Drive-Dreams/.../extracted    (symlink)
│   │   ├── baseline.mp4                           ← your input dashcam
│   │   └── injected_labels/frame_*.txt
│   ├── shared_data/
│   │   ├── *.all_object_info.json                 ← injected frame labels
│   │   └── boxes.json
│   ├── outputs/dream_run/<scenario>.mp4           ← Helios output (or stub)
│   ├── dreamloop_dataset/
│   │   ├── images/train/frame_*.jpg
│   │   └── labels/train/frame_*.txt
│   ├── runs/train/exp/                            ← YOLO training output
│   │   ├── weights/best.pt
│   │   ├── results.csv
│   │   └── confusion_matrix.png
│   └── injected_{pedestrian_cross,collision,jaywalker,bike}/
│       ├── *.all_object_info.json
│       ├── boxes.json
│       ├── <scenario>.mp4
│       └── <scenario>_labels.pt
│
└── Cosmos-Drive-Dreams/                           ← NVIDIA toolkit
    ├── cosmos-drive-dreams-toolkits/
    │   ├── render_from_rds_hq.py
    │   ├── scripts/generate_video_single_view.py
    │   ├── scripts/generate_video_multi_view.py
    │   ├── utils/bbox_utils.py                    ← PATCHED in Step 6
    │   ├── utils/bbox_utils.py.orig               ← backup of original
    │   └── data/
    │       ├── all_object_info/<clip_id>.tar
    │       ├── all_object_info/extracted/<clip_id>.<frame:06d>.all_object_info.json
    │       ├── pose/<clip_id>.tar
    │       └── ...
    └── assets/example/                            ← if LFS pulled (Step 5a)
```

---

## Appendix C: Estimated wall-clock timing on a fresh GX10

| Phase | Time | Bottleneck |
|---|---|---|
| Step 1 (verify hardware) | 1 min | n/a |
| Step 2 (apt install) | 5–10 min | apt download speed |
| Step 3 (git clone + venv) | 2 min | n/a |
| Step 4 (pip install) | 10–20 min | aarch64 wheel availability; CUDA torch builds |
| Step 5a (clone Cosmos repo) | 5–10 min | n/a |
| Step 5a (git lfs pull) | 15–45 min | network bandwidth; LFS files are large |
| Step 5c (download one clip) | 10–30 min | HF download |
| Step 5d (extract tar) | <1 min | n/a |
| Step 6 (pycg patch) | 1 min | n/a |
| Step 7 (inject.py smoke test) | <1 min | n/a |
| Step 8 (Helios stub) | 1 min | n/a |
| Step 9 (full master_pipeline) | 20–50 min | YOLO training (Step 5 of pipeline) |
| Step 10 (validation) | 1 min | n/a |
| **Total wall-clock** | **~1.5–3 hours** | mostly downloads + training |

If you exceed 4 hours, you have hit an undocumented error. Triage it against Appendix A first; if no match, capture `2>&1 | tee error.log` and add the new entry.

---

## Appendix D: What this guide deliberately does NOT cover

These belong on sibling branches or to future work, not on `main`:

- **`outputs/<clip_id>/*.mp4` per-clip subdir layout** — that's Kayla's `app.py` contract on the `kayla` branch.
- **`make_demo_floor.py` OpenCV blizzard demo** — that's on the `shreyas-python-scripts` branch.
- **Streamlit dashboard** — Kayla's branch.
- **Cosmos `render_from_rds_hq.py` end-to-end render** — partially blocked on the pycg patch + Cosmos venv setup; not part of this `main`-only path.
- **Helios real model weights** — Step 8 ships a stub; real integration is open work.
- **Waymo decode** — see top of this doc; the Waymo SDK has no aarch64 wheel. Use Cosmos's bundled RDS-HQ data.

When integrating sibling branches, add their setup steps below as new sections rather than mutating the main-only flow in §3–§10.

---

## Appendix E: Contacts / ownership

Update this section with team contacts as branches merge:

- **`main` (this branch)**: scenario injection + YOLO fine-tuning pipeline.
- **`kayla`**: Streamlit dashboard + Nexar ingest + UI metrics.
- **`shyam`**: YOLO training pipeline (merged into `main` as of `f2e5a29`).
- **`shreyas-python-scripts`**: Waymo-style perturbation library + tests + demo floor.

For setup questions, refer to this document first; if it doesn't cover your case, capture the exact error message and add it as a new row in Appendix A.

"""Central paths and constants. Override locally via config.local.py if needed."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Data lives outside git. data/ is gitignored.
DATA_ROOT = Path(os.environ.get("DREAMLOOP_DATA", REPO_ROOT / "data"))

# Raw Waymo segments dropped here.
WAYMO_RAW = DATA_ROOT / "waymo_raw"

# waymo_loader.py outputs here.
WAYMO_INTERMEDIATE = DATA_ROOT / "waymo_intermediate"

# rds_writer.py outputs here. This is what Cosmos-Drive-Dreams ingests.
RDS_HQ_ROOT = DATA_ROOT / "rds_hq"

# cosmos_runner.py outputs here.
COSMOS_OUTPUTS = DATA_ROOT / "cosmos_outputs"

# Cosmos-Drive-Dreams clone path. Override on GX10.
COSMOS_REPO = Path(os.environ.get("COSMOS_REPO", REPO_ROOT.parent / "Cosmos-Drive-Dreams"))

# Model weights.
MODELS_ROOT = Path(os.environ.get("DREAMLOOP_MODELS", DATA_ROOT / "models"))
YOLO_WEIGHTS = MODELS_ROOT / "yolov8n.pt"  # nano for speed; swap to yolov8x for stronger baseline.

# Which Waymo camera we treat as canonical. FRONT = enum 1 in Waymo proto.
CANONICAL_CAMERA = "FRONT"

# YOLO class names we treat as on-road objects, mapped to Waymo label types.
# Waymo types: 1=VEHICLE, 2=PEDESTRIAN, 3=SIGN, 4=CYCLIST
COCO_TO_WAYMO = {
    "car": 1, "truck": 1, "bus": 1, "motorcycle": 1,
    "person": 2,
    "bicycle": 4,
    "stop sign": 3,
}

# Try to load local overrides.
try:
    from config_local import *  # noqa: F401,F403
except ImportError:
    pass

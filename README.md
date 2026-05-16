# DreamLoop

Counterfactual AV data pipeline: Waymo → tracklet perturbation → RDS-HQ → Cosmos render → YOLO eval.

## Branch + layout

The pipeline scripts live in `pipeline/`, isolated from other team members'
work (Streamlit UI, discovery loop, slide deck) so branches merge without
collisions.

```
DreamLoop/
├── pipeline/        ← all pipeline modules + tests + integration doc
└── (other folders for UI, discovery, etc. as they land)
```

Start here: **[`pipeline/INTEGRATION.md`](pipeline/INTEGRATION.md)** — full
install + run + schema reference.

## Quick run

```bash
cd pipeline
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash tests/run_all.sh           # 32 tests, no Waymo / no GPU needed
```

Then follow the end-to-end command sequence in `pipeline/INTEGRATION.md` §3.

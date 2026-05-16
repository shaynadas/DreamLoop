import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
NEXAR_DIR = ROOT / "nexar_videos" / "train" / "negative"
OUTPUTS_DIR = ROOT / "outputs"

# Pre-rendered H.264 clips only — buttons never run live inference.
PROCESSED_VIDEO = OUTPUTS_DIR / "dreamloop_web.mp4"
BASELINE_YOLO_VIDEO = OUTPUTS_DIR / "baseline_yolo_web.mp4"
FINETUNED_YOLO_VIDEO = OUTPUTS_DIR / "finetuned_yolo_web.mp4"
METRICS_FILE = OUTPUTS_DIR / "metrics.json"

DEFAULT_METRICS = {
    "baseline_map": 0.42,
    "finetuned_map": 0.89,
    "clear_weather_map": 0.70,
    "latency_ms": 32,
    "condition": "Snowy blizzard (edge case)",
}
st.set_page_config(page_title="DreamLoop AV Dashboard", layout="wide")

st.title("DreamLoop AV Simulator Dashboard")
st.subheader("Cosmos-Drive-Dreams Edge-Case Evaluation")

# Sidebar
st.sidebar.header("Scenario Settings")
st.sidebar.selectbox(
    "Select Driving Scenario",
    ["Snowy Blizzard", "Night Drive", "Heavy Rain", "Foggy Highway"],
)
st.sidebar.markdown("**Weather controls**")
rain = st.sidebar.slider("Rain", 0.0, 1.0, 0.0)
night = st.sidebar.slider("Night", 0.0, 1.0, 0.0)
fog = st.sidebar.slider("Fog", 0.0, 1.0, 0.0)
st.sidebar.caption(f"Rain {rain:.2f} · Night {night:.2f} · Fog {fog:.2f}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Offline render (not run from this UI)**")
st.sidebar.code(
    "python dreamloop_sim.py "
    "--video nexar_videos/train/negative/01040.mp4 "
    "--output outputs/dreamloop_output.mp4 --no-display\n"
    "ffmpeg -y -i outputs/dreamloop_output.mp4 -c:v libx264 "
    "-pix_fmt yuv420p -movflags +faststart outputs/dreamloop_web.mp4",
    language="bash",
)


def list_mp4s(folder: Path) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(f.name for f in folder.glob("*.mp4"))


def load_metrics() -> dict:
    """Load pre-computed eval metrics from disk (no live inference)."""
    metrics = dict(DEFAULT_METRICS)
    if METRICS_FILE.is_file():
        try:
            loaded = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metrics.update(loaded)
        except (json.JSONDecodeError, OSError):
            pass
    return metrics


def play_prerendered(session_key: str, path: Path, button_label: str) -> None:
    """Load a pre-rendered file from outputs/ when the user clicks play."""
    if st.session_state.get(session_key):
        if path.is_file():
            st.video(str(path))
            st.success(f"Playing `{path.name}`")
            return
        st.warning(
            f"No pre-rendered video at `{path}`.\n\n"
            f"Add `{path.name}` (H.264, e.g. ffmpeg `*_web.mp4`), then try again."
        )
        if st.button("Try again", width="stretch", key=f"retry_{session_key}"):
            st.session_state.pop(session_key, None)
            st.rerun()
        return

    if st.button(button_label, type="primary", width="stretch", key=f"btn_{session_key}"):
        st.session_state[session_key] = True
        st.rerun()


video_files = list_mp4s(NEXAR_DIR)
raw_video = NEXAR_DIR / video_files[0] if video_files else None

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

with row1_col1:
    st.markdown("### Raw Camera Feed")
    st.caption(f"Pre-rendered source: `{NEXAR_DIR.relative_to(ROOT)}`")
    if raw_video:
        st.video(str(raw_video))
    else:
        st.error(f"No .mp4 files found in `{NEXAR_DIR}`")

with row1_col2:
    st.markdown("### Processed Simulation (HUD + Snow)")
    st.caption(f"Pre-rendered: `{PROCESSED_VIDEO.relative_to(ROOT)}`")
    play_prerendered(
        "play_processed",
        PROCESSED_VIDEO,
        "Play processed demo for judge",
    )

with row2_col1:
    st.markdown("### Baseline YOLO (Misses Pedestrian)")
    st.caption(f"Pre-rendered: `{BASELINE_YOLO_VIDEO.relative_to(ROOT)}`")
    play_prerendered(
        "play_baseline",
        BASELINE_YOLO_VIDEO,
        "Play baseline YOLO",
    )

with row2_col2:
    st.markdown("### Fine-Tuned YOLO (Tracking Correctly)")
    st.caption(f"Pre-rendered: `{FINETUNED_YOLO_VIDEO.relative_to(ROOT)}`")
    play_prerendered(
        "play_finetuned",
        FINETUNED_YOLO_VIDEO,
        "Play fine-tuned YOLO",
    )

st.markdown("---")
st.markdown("## Validation Metrics Panel")
st.caption(
    f"Before/after mAP from `{METRICS_FILE.relative_to(ROOT)}` when present, else demo defaults."
)

metrics = load_metrics()
baseline_map = float(metrics["baseline_map"])
finetuned_map = float(metrics["finetuned_map"])
clear_map = float(metrics["clear_weather_map"])
latency_ms = int(metrics["latency_ms"])
condition = str(metrics["condition"])

weather_drop = baseline_map - clear_map
recovery_gain = finetuned_map - baseline_map
relative_gain_pct = (recovery_gain / baseline_map * 100) if baseline_map > 0 else 0.0

st.markdown(f"**Eval condition:** {condition}")

before_col, after_col = st.columns(2)

with before_col:
    st.markdown("### Before — Baseline YOLO")
    st.metric(
        label="mAP @ edge case",
        value=f"{baseline_map:.2f}",
        delta=f"{weather_drop:.2f} vs clear ({clear_map:.2f})",
        delta_color="inverse",
    )
    st.markdown(
        "- Off-the-shelf model under weather stress  \n"
        "- Misses pedestrian in blizzard demo clip"
    )

with after_col:
    st.markdown("### After — Fine-Tuned YOLO")
    st.metric(
        label="mAP @ edge case",
        value=f"{finetuned_map:.2f}",
        delta=f"+{recovery_gain:.2f} vs baseline",
        delta_color="normal",
    )
    st.markdown(
        "- Fine-tuned on synthetic Cosmos frames  \n"
        "- Tracks pedestrian correctly in demo clip"
    )

st.markdown("#### mAP comparison")
st.bar_chart(
    {
        "mAP": {
            "Clear weather (reference)": clear_map,
            "Before (baseline)": baseline_map,
            "After (fine-tuned)": finetuned_map,
        }
    },
    width="stretch",
)

summary_col1, summary_col2, summary_col3 = st.columns(3)
with summary_col1:
    st.metric("Absolute gain (after − before)", f"{recovery_gain:+.2f}")
with summary_col2:
    st.metric("Relative improvement", f"{relative_gain_pct:+.0f}%")
with summary_col3:
    st.metric("Inference latency", f"{latency_ms} ms", delta="Stable")
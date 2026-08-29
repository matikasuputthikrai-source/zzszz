"""
Ehrlich-Z Web App
==================
AI screening tool for Ehrlichia canis in canine blood smear images.

Run locally:
    pip install streamlit requests opencv-python-headless pillow numpy
    streamlit run app.py

Deploy for free:
    1. Push this file + requirements.txt to a GitHub repo
    2. Go to share.streamlit.io -> connect GitHub -> select repo -> Deploy
    3. Get a public link instantly (free, no server needed)
"""

import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io

# ============================================================
# Roboflow settings (fill in your real values)
# ============================================================
ROBOFLOW_API_KEY = "2G5Lbz1TQC0doTcK4YiO"          # <-- your real API key
MODEL_ID = "my-first-project-cp4zt/1"                # <-- your real model_id (project/version)
TARGET_CLASS = "Ehrilchia canis"                     # <-- must match the exact label spelling in Roboflow
DEFAULT_CONFIDENCE = 0.5

st.set_page_config(page_title="Ehrlich-Z", page_icon="🩸", layout="wide")


# ============================================================
# Styling — Giemsa-stain theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --bg-deep: #1E1330;
    --bg-mid: #2E1E45;
    --paper: #F3EEE2;
    --ink: #211A2B;
    --blood: #C1443C;
    --sage: #6E8B5E;
    --gold: #C9A567;
    --muted: #7A7189;
}

.stApp {
    background: radial-gradient(circle at 15% 0%, var(--bg-mid) 0%, var(--bg-deep) 55%);
    background-attachment: fixed;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.18em;
    font-size: 0.72rem;
    color: var(--gold);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.wordmark {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.6rem;
    color: var(--paper);
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.wordmark .drop { color: var(--blood); }
.subhead {
    color: #C9C2DA;
    font-size: 0.98rem;
    margin-top: 0.35rem;
    margin-bottom: 2rem;
}

.paper-card {
    background: var(--paper);
    border-radius: 22px;
    padding: 1.9rem 2.1rem;
    box-shadow: 0 18px 40px rgba(10, 5, 20, 0.35);
    margin-bottom: 1.4rem;
}
.card-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--blood);
    margin-bottom: 0.15rem;
}
.card-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.5rem;
    color: var(--ink);
    margin: 0 0 0.3rem 0;
}
.card-desc { color: var(--muted); font-size: 0.9rem; margin-bottom: 1.1rem; }

[data-testid="stFileUploaderDropzone"] {
    background: #FBF8F1 !important;
    border: 1.5px dashed #C7BFA8 !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--muted) !important; }

div.stButton > button {
    background: linear-gradient(120deg, var(--blood), #D9695F);
    color: #FBF3EE;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    width: 100%;
    box-shadow: 0 8px 18px rgba(193, 68, 60, 0.35);
}
div.stButton > button:hover { background: linear-gradient(120deg, #A9382F, #C1443C); color: #FBF3EE; }

.stat-box {
    border-radius: 14px;
    padding: 1rem 1.1rem;
    height: 100%;
}
.stat-box .label { font-size: 0.78rem; color: var(--muted); margin-bottom: 0.5rem; }
.stat-box .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 600;
    color: var(--ink);
}
.stat-total { background: #E7E2F2; }
.stat-infected { background: #F3DEDB; }
.stat-percent { background: #E4EBDD; }

.result-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--gold);
}
.result-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: var(--muted); margin-top: 0.2rem;}
.result-level {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.1rem;
    margin: 0.15rem 0 0.6rem 0;
}
.level-negative { color: var(--sage); }
.level-mild { color: #B58A2E; }
.level-moderate { color: #C1443C; }
.level-severe { color: #8E1F1F; }

.score-track {
    background: #E4DED0;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-top: 0.6rem;
}
.score-fill { height: 100%; border-radius: 999px; }

.flow-strip {
    text-align: center;
    color: #C9C2DA;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    margin: 1.6rem 0 0.6rem 0;
}

.disclaimer {
    color: #B9AFCF;
    font-size: 0.8rem;
    text-align: center;
    margin-top: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Core functions
# ============================================================
def infer_image_bytes(image_bytes, confidence=0.5):
    """Send image bytes to the Roboflow model via REST API"""
    url = f"https://detect.roboflow.com/{MODEL_ID}"
    params = {
        "api_key": ROBOFLOW_API_KEY,
        "confidence": int(confidence * 100),
    }
    response = requests.post(url, params=params, files={"file": image_bytes})
    response.raise_for_status()
    return response.json()


def split_image_into_9(pil_image):
    """Split a PIL image into 9 tiles (3x3 grid)"""
    img = np.array(pil_image.convert("RGB"))
    h, w = img.shape[:2]
    tile_w, tile_h = w // 3, h // 3
    tiles = []
    for row in range(3):
        for col in range(3):
            x1, y1 = col * tile_w, row * tile_h
            x2 = w if col == 2 else x1 + tile_w
            y2 = h if row == 2 else y1 + tile_h
            tiles.append(Image.fromarray(img[y1:y2, x1:x2]))
    return tiles


def pil_to_bytes(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def draw_boxes(pil_image, predictions):
    img = np.array(pil_image.convert("RGB"))
    for pred in predictions:
        x, y = int(pred["x"]), int(pred["y"])
        w, h = int(pred["width"]), int(pred["height"])
        x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
        is_infected = pred["class"].lower() == TARGET_CLASS.lower()
        color = (193, 68, 60) if is_infected else (110, 139, 94)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return Image.fromarray(img)


def calculate_infection_rate(total_cells, infected_cells):
    if total_cells == 0:
        return 0.0, "no_data", "No white blood cells detected — try another image"
    percent = round((infected_cells / total_cells) * 100, 2)
    if percent == 0:
        return percent, "negative", "No infection detected"
    elif percent <= 2:
        return percent, "mild", "Mild"
    elif percent <= 5:
        return percent, "moderate", "Moderate"
    else:
        return percent, "severe", "Severe"


LEVEL_STYLE = {
    "no_data":  {"class": "level-negative", "color": "#6E8B5E"},
    "negative": {"class": "level-negative", "color": "#6E8B5E"},
    "mild":     {"class": "level-mild",     "color": "#B58A2E"},
    "moderate": {"class": "level-moderate", "color": "#C1443C"},
    "severe":   {"class": "level-severe",   "color": "#8E1F1F"},
}


# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="eyebrow">AI SCREENING · CANINE EHRLICHIOSIS</div>
<div class="wordmark">🩸 Ehrlich<span class="drop">-Z</span></div>
<div class="subhead">Spotting Ehrlichia canis morulae in blood smear images — faster than scanning by eye.</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, DEFAULT_CONFIDENCE, 0.05)
    st.markdown("---")
    st.markdown(
        "**How to use**\n"
        "1. Upload a blood smear image taken through a microscope\n"
        "2. Click \u201cAnalyze Image\u201d\n"
        "3. Review the infected-cell percentage and severity level"
    )
    st.markdown("---")
    st.caption("⚠️ This is a preliminary screening result, not a final diagnosis. A veterinarian should confirm with ELISA/PCR testing.")

# ============================================================
# Card: Start with your image (upload + analyze button)
# ============================================================
st.markdown('<div class="paper-card">', unsafe_allow_html=True)
st.markdown("""
<div class="card-eyebrow">STEP 1 — UPLOAD</div>
<div class="card-title">Start with your blood smear image</div>
<div class="card-desc">Upload one image to see the infected-cell ratio and severity level in seconds.</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag an image here, or click to browse (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)
run_clicked = st.button("✨ Analyze Image", disabled=uploaded_file is None)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# Defaults before analysis
# ============================================================
total_cells, infected_cells = 0, 0
percent, level_key, level_label = 0.0, "no_data", "Awaiting analysis"
annotated_tiles = []
has_result = False

# ============================================================
# Run real analysis
# ============================================================
if uploaded_file is not None and run_clicked:
    pil_image = Image.open(uploaded_file)

    with st.spinner("Splitting image into 9 tiles and running detection..."):
        tiles = split_image_into_9(pil_image)
        progress = st.progress(0)
        for i, tile in enumerate(tiles):
            try:
                result = infer_image_bytes(pil_to_bytes(tile), confidence=confidence)
                predictions = result.get("predictions", [])
                total_cells += len(predictions)
                infected_cells += sum(
                    1 for p in predictions if p["class"].lower() == TARGET_CLASS.lower()
                )
                annotated_tiles.append(draw_boxes(tile, predictions))
            except Exception as e:
                st.error(f"Error on tile {i+1}: {e}")
                annotated_tiles.append(tile)
            progress.progress((i + 1) / 9)

    percent, level_key, level_label = calculate_infection_rate(total_cells, infected_cells)
    has_result = True

# ============================================================
# Card: QUICK READ — overall results
# ============================================================
st.markdown('<div class="paper-card">', unsafe_allow_html=True)
st.markdown("""
<div class="card-eyebrow">QUICK READ</div>
<div class="card-title">Overall Results</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.markdown(f"""
<div class="stat-box stat-total">
    <div class="label">Total Cells</div>
    <div class="value">{total_cells if has_result else '\u2014'}</div>
</div>""", unsafe_allow_html=True)
c2.markdown(f"""
<div class="stat-box stat-infected">
    <div class="label">Infected Cells</div>
    <div class="value">{infected_cells if has_result else '\u2014'}</div>
</div>""", unsafe_allow_html=True)
c3.markdown(f"""
<div class="stat-box stat-percent">
    <div class="label">% Infected</div>
    <div class="value">{percent if has_result else '\u2014'}{'%' if has_result else ''}</div>
</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# Card: RESULT LEVEL — score out of 100
# ============================================================
style = LEVEL_STYLE[level_key]
score = min(100, round(percent * 15)) if has_result else 0

st.markdown(f"""
<div class="paper-card">
    <div class="result-eyebrow">RESULT LEVEL</div>
    <div class="result-title">Severity</div>
    <div class="result-level {style['class']}">{level_label if has_result else 'Awaiting analysis'}</div>
    <div class="card-desc" style="margin-bottom:0.4rem;">
        {"Upload an image and click Analyze to see results instantly." if not has_result else
         f"Found morulae in {infected_cells} of {total_cells} detected cells."}
    </div>
    <div style="display:flex; justify-content:space-between; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:{style['color']};">
        <span>SCORE</span><span>{score}/100</span>
    </div>
    <div class="score-track">
        <div class="score-fill" style="width:{score}%; background:{style['color']};"></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="flow-strip">Upload · Analyze · Review</div>', unsafe_allow_html=True)

# ============================================================
# Result images (shown after analysis)
# ============================================================
if has_result:
    st.markdown('<div class="paper-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="card-eyebrow">DETECTION MAP</div>
    <div class="card-title">Detected Regions</div>
    <div class="card-desc">Red box = suspected infected · Green box = normal cell</div>
    """, unsafe_allow_html=True)
    grid_cols = st.columns(3)
    for i, tile_img in enumerate(annotated_tiles):
        with grid_cols[i % 3]:
            st.image(tile_img, caption=f"Tile {i+1}", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">This is a preliminary screening result, not a final diagnosis — a veterinarian should confirm.</div>',
    unsafe_allow_html=True,
)

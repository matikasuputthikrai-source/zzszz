"""
Ehrlich-Z Web App
==================
ระบบตรวจหาเชื้อ Ehrlichia canis จากภาพสเมียร์เลือดสุนัข

วิธีรันในเครื่อง:
    pip install streamlit requests opencv-python-headless pillow numpy
    streamlit run app.py

วิธี deploy ให้คนอื่นใช้ได้ฟรี:
    1. อัปโหลดไฟล์นี้ + requirements.txt ขึ้น GitHub (repo แยกต่างหาก)
    2. ไปที่ share.streamlit.io -> เชื่อม GitHub -> เลือก repo -> Deploy
    3. ได้ลิงก์เว็บสาธารณะทันที (ฟรี ไม่ต้องมีเซิร์ฟเวอร์ของตัวเอง)
"""

import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io

# ============================================================
# ตั้งค่า Roboflow (แก้ให้ตรงกับของจริง)
# ============================================================
ROBOFLOW_API_KEY = "2G5Lbz1TQC0doTcK4YiO"          # <-- แก้เป็นของจริง
MODEL_ID = "my-first-project-cp4zt/1"                # <-- แก้เป็นของจริง (project/version)
TARGET_CLASS = "Ehrilchia canis"                     # <-- ต้องตรงกับที่ label ไว้ใน Roboflow เป๊ะๆ
DEFAULT_CONFIDENCE = 0.5

st.set_page_config(page_title="Ehrlich-Z", page_icon="🩸", layout="wide")


# ============================================================
# ดีไซน์ — ธีมสีย้อมเลือด (Giemsa stain) แทนธีม "Image Insight" เดิม
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=IBM+Plex+Sans+Thai:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

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

html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', sans-serif; }

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
# ฟังก์ชันหลัก
# ============================================================
def infer_image_bytes(image_bytes, confidence=0.5):
    """ส่งภาพ (bytes) เข้าโมเดล Roboflow ผ่าน REST API"""
    url = f"https://detect.roboflow.com/{MODEL_ID}"
    params = {
        "api_key": ROBOFLOW_API_KEY,
        "confidence": int(confidence * 100),
    }
    response = requests.post(url, params=params, files={"file": image_bytes})
    response.raise_for_status()
    return response.json()


def split_image_into_9(pil_image):
    """แบ่งภาพ PIL เป็น 9 ส่วน (3x3 grid)"""
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
        return 0.0, "no_data", "ไม่พบเซลล์เม็ดเลือดขาวในภาพ — ลองถ่ายภาพใหม่"
    percent = round((infected_cells / total_cells) * 100, 2)
    if percent == 0:
        return percent, "negative", "ไม่พบการติดเชื้อ"
    elif percent <= 2:
        return percent, "mild", "ระดับต่ำ"
    elif percent <= 5:
        return percent, "moderate", "ระดับปานกลาง"
    else:
        return percent, "severe", "ระดับสูง"


LEVEL_STYLE = {
    "no_data":  {"class": "level-negative", "color": "#6E8B5E"},
    "negative": {"class": "level-negative", "color": "#6E8B5E"},
    "mild":     {"class": "level-mild",     "color": "#B58A2E"},
    "moderate": {"class": "level-moderate", "color": "#C1443C"},
    "severe":   {"class": "level-severe",   "color": "#8E1F1F"},
}


# ============================================================
# หัวหน้าเว็บ
# ============================================================
st.markdown("""
<div class="eyebrow">AI SCREENING · CANINE EHRLICHIOSIS</div>
<div class="wordmark">🩸 Ehrlich<span class="drop">-Z</span></div>
<div class="subhead">มองหา morulae ของ Ehrlichia canis ในภาพสเมียร์เลือด — เร็วกว่าการนั่งส่องด้วยตาเปล่า</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ ตั้งค่า")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, DEFAULT_CONFIDENCE, 0.05)
    st.markdown("---")
    st.markdown(
        "**วิธีใช้**\n"
        "1. อัปโหลดภาพสเมียร์เลือดที่ถ่ายจากกล้องจุลทรรศน์\n"
        "2. กด \u201cวิเคราะห์รูปภาพ\u201d\n"
        "3. ดูผล % เซลล์ติดเชื้อและระดับความรุนแรง"
    )
    st.markdown("---")
    st.caption("⚠️ ผลลัพธ์เป็นการคัดกรองเบื้องต้นเท่านั้น ไม่ใช่การวินิจฉัยขั้นสุดท้าย ควรให้สัตวแพทย์ตรวจสอบซ้ำและยืนยันด้วย ELISA/PCR")

# ============================================================
# การ์ด: เริ่มจากรูปของคุณ (อัปโหลด + ปุ่มวิเคราะห์)
# ============================================================
st.markdown('<div class="paper-card">', unsafe_allow_html=True)
st.markdown("""
<div class="card-eyebrow">STEP 1 — UPLOAD</div>
<div class="card-title">เริ่มจากภาพสเมียร์เลือดของคุณ</div>
<div class="card-desc">อัปโหลดภาพหนึ่งภาพ เพื่อดูสัดส่วนเซลล์ติดเชื้อและระดับความรุนแรงอย่างรวดเร็ว</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "ลากรูปมาวางที่นี่ หรือคลิกเพื่อเลือกไฟล์ JPG, PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)
run_clicked = st.button("✨ วิเคราะห์รูปภาพ", disabled=uploaded_file is None)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# ค่าเริ่มต้นก่อนวิเคราะห์
# ============================================================
total_cells, infected_cells = 0, 0
percent, level_key, level_label = 0.0, "no_data", "รอวิเคราะห์"
annotated_tiles = []
has_result = False

# ============================================================
# รันวิเคราะห์จริง
# ============================================================
if uploaded_file is not None and run_clicked:
    pil_image = Image.open(uploaded_file)

    with st.spinner("กำลังแบ่งภาพเป็น 9 ส่วน และตรวจจับด้วย AI..."):
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
                st.error(f"เกิดข้อผิดพลาดที่ส่วนที่ {i+1}: {e}")
                annotated_tiles.append(tile)
            progress.progress((i + 1) / 9)

    percent, level_key, level_label = calculate_infection_rate(total_cells, infected_cells)
    has_result = True

# ============================================================
# การ์ด: QUICK READ — ผลลัพธ์โดยรวม
# ============================================================
st.markdown('<div class="paper-card">', unsafe_allow_html=True)
st.markdown("""
<div class="card-eyebrow">QUICK READ</div>
<div class="card-title">ผลลัพธ์โดยรวม</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.markdown(f"""
<div class="stat-box stat-total">
    <div class="label">เซลล์ทั้งหมด</div>
    <div class="value">{total_cells if has_result else '\u2014'}</div>
</div>""", unsafe_allow_html=True)
c2.markdown(f"""
<div class="stat-box stat-infected">
    <div class="label">เซลล์ติดเชื้อ</div>
    <div class="value">{infected_cells if has_result else '\u2014'}</div>
</div>""", unsafe_allow_html=True)
c3.markdown(f"""
<div class="stat-box stat-percent">
    <div class="label">% Infected</div>
    <div class="value">{percent if has_result else '\u2014'}{'%' if has_result else ''}</div>
</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# การ์ด: RESULT LEVEL — คะแนน /100
# ============================================================
style = LEVEL_STYLE[level_key]
score = min(100, round(percent * 15)) if has_result else 0

st.markdown(f"""
<div class="paper-card">
    <div class="result-eyebrow">RESULT LEVEL</div>
    <div class="result-title">ระดับผลของภาพ</div>
    <div class="result-level {style['class']}">{level_label if has_result else 'รอวิเคราะห์'}</div>
    <div class="card-desc" style="margin-bottom:0.4rem;">
        {"อัปโหลดภาพ แล้วกดวิเคราะห์เพื่อดูผลได้ทันที" if not has_result else
         f"พบเซลล์ที่มี morulae {infected_cells} เซลล์ จากทั้งหมด {total_cells} เซลล์ที่ตรวจพบ"}
    </div>
    <div style="display:flex; justify-content:space-between; font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:{style['color']};">
        <span>SCORE</span><span>{score}/100</span>
    </div>
    <div class="score-track">
        <div class="score-fill" style="width:{score}%; background:{style['color']};"></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="flow-strip">อัปโหลด · วิเคราะห์ · ดูภาพรวม</div>', unsafe_allow_html=True)

# ============================================================
# ภาพประกอบผล (แสดงเมื่อมีผลลัพธ์แล้ว)
# ============================================================
if has_result:
    st.markdown('<div class="paper-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="card-eyebrow">DETECTION MAP</div>
    <div class="card-title">ภาพที่ตรวจจับแล้ว</div>
    <div class="card-desc">กรอบสีแดง = สงสัยติดเชื้อ · กรอบสีเขียว = เซลล์ปกติ</div>
    """, unsafe_allow_html=True)
    grid_cols = st.columns(3)
    for i, tile_img in enumerate(annotated_tiles):
        with grid_cols[i % 3]:
            st.image(tile_img, caption=f"ส่วนที่ {i+1}", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">ผลลัพธ์เป็นการคัดกรองเบื้องต้น ไม่ใช่การวินิจฉัยขั้นสุดท้าย — ควรให้สัตวแพทย์ตรวจสอบซ้ำ</div>',
    unsafe_allow_html=True,
)

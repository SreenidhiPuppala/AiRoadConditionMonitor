import streamlit as st
from PIL import Image
import cv2
import os
import tempfile
import numpy as np
import pandas as pd
from collections import Counter
from ultralytics import YOLO
# ===== QUANTITATIVE ASSUMPTIONS (MODEL-ALIGNED) =====
PIXEL_TO_METER = 0.02

# COST per square meter (₹)
COST_PER_SQM = {
    "G00": 900,    # pothole (was D40)
    "D40": 150,    # crack (was D00/D10)
    "D00": 180,    # other cracks if any
    "D10": 200,
    "D20": 450,
}

# Repair thickness in meters
REPAIR_THICKNESS = {
    "G00": 0.08,   # pothole
    "D40": 0.01,   # crack sealing
    "D00": 0.01,
    "D10": 0.02,
    "D20": 0.05,
}

# Material density (kg/m³)
MATERIAL_DENSITY = {
    "G00": 2400,   # pothole asphalt mix
    "D40": 1100,   # crack sealant
    "D00": 1100,
    "D10": 1200,
    "D20": 2200,
}
# ====================================================

# ====================== CONFIG ======================
MODEL_PATH = "runs/detect/train3/weights/best.pt"  # Path to YOLOv8 model
OUTPUT_DIR = "outputs/frames"  # Directory to save results
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_frames = []       # [(frame_no, save_path)]
detection_summary = []   # [{Frame, Type, Confidence, Severity, Timestamp}]
# =====================================================
#yolo detect train model=yolov8n(dot)pt data=data.yaml epochs=20 imgsz=416 cache=True batch=16.  ---training command
# Load YOLO model
model = YOLO(MODEL_PATH)
model.info()
# Streamlit page setup
st.set_page_config(page_title="Road Damage Detector", layout="wide")
st.title("🛣️ Road Damage Detection")
st.markdown("""
Upload a **road image** or **video** to detect potholes, cracks.  
""")

# ================= HELPER FUNCTIONS ==================
def estimate_severity(bbox, frame_shape):
    """Estimate damage severity based on bounding box area."""
    frame_h, frame_w, _ = frame_shape
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    total_area = frame_w * frame_h
    ratio = area / total_area
    if ratio < 0.002:
        return "Minor"
    elif ratio < 0.01:
        return "Moderate"
    else:
        return "Severe"

def quantitative_repair_estimation(bbox, damage_type):
    x1, y1, x2, y2 = bbox
    pixel_area = (x2 - x1) * (y2 - y1)

    area_m2 = pixel_area * (PIXEL_TO_METER ** 2)
    cost = area_m2 * COST_PER_SQM.get(damage_type, 0)

    if cost < 500:
        priority = "Low"
    elif cost < 3000:
        priority = "Medium"
    else:
        priority = "High"

    return round(area_m2, 3), round(cost, 2), priority


def estimate_material_required(area_m2, damage_type):
    thickness = REPAIR_THICKNESS.get(damage_type, 0)
    density = MATERIAL_DENSITY.get(damage_type, 0)

    volume = area_m2 * thickness
    weight = volume * density

    return round(weight, 2)

def process_frame(frame_rgb, frame_num=None, fps=None, save_path=None):
    """Run YOLO on one frame and return resized annotated frame + detections."""
    # Optional: enhance details
    frame_rgb = cv2.detailEnhance(frame_rgb, sigma_s=10, sigma_r=0.15)

    # Resize for YOLO input (speed optimization)
    target_size = 416
    frame_resized = cv2.resize(frame_rgb, (target_size, target_size))

    results = model.predict(source=frame_resized, conf=0.1, iou=0.45, verbose=False)
    result_frame = results[0].plot()  # high-res annotated frame

    # Resize output frame for display and saving
    max_output_size = 400  # max width or height
    h, w, _ = result_frame.shape
    scale = min(max_output_size / w, max_output_size / h, 1.0)
    resized_frame = cv2.resize(result_frame, (int(w * scale), int(h * scale)))

    # Collect detections
    detections = []
    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls_id = box
        severity = estimate_severity((x1, y1, x2, y2), frame_rgb.shape)

        area, cost, priority = quantitative_repair_estimation(
            (x1, y1, x2, y2),
            model.names[int(cls_id)]
        )

        material_kg = estimate_material_required(
            area,
            model.names[int(cls_id)]
        )
        timestamp = round(frame_num / fps, 2) if fps else 0
        detections.append({
            "Frame": frame_num if frame_num is not None else 0,
            "Type": model.names[int(cls_id)],
            "Confidence": round(float(conf), 2),
            "Severity": severity,
            "Affected Area (m²)": area,
            "Material Required (kg)": material_kg,
            "Estimated Cost (₹)": cost,
            "Priority": priority,
            "Timestamp (s)": timestamp
        })

    # Save resized frame with compression
    if save_path:
        img = Image.fromarray(resized_frame)
        img.save(save_path, quality=75)  # reduce JPEG size

    return resized_frame, detections

# ============== SUMMARY & SUGGESTION FUNCTIONS ===============

def generate_detection_summary(detection_summary):
    """Display summary table and allow CSV download."""
    if not detection_summary:
        st.info("No detections available to generate a report.")
        return

    st.markdown("---")
    st.subheader("📊 Detection Summary Report")

    df = pd.DataFrame(detection_summary)
    df = df.drop_duplicates(ignore_index=True)

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📄 Download Detection Report (CSV)",
        data=csv,
        file_name="road_damage_report.csv",
        mime="text/csv"
    )




def generate_repair_suggestions(detection_summary):
    repair_suggestions = {
    "G00": "Fill pothole using hot-mix asphalt and compact properly.",
    "D40": "Seal surface cracks using bituminous crack filler.",
    "D00": "Apply longitudinal crack sealing.",
    "D10": "Apply transverse crack sealing.",
    "D20": "Perform full-depth patch repair.",
    "F00": "Repaint faded road markings."
}
    if not detection_summary:
        return

    st.markdown("---")
    st.subheader("🧰 Smart Repair Suggestions")

    
    detected_types = [d["Type"] for d in detection_summary]
    damage_count = Counter(detected_types)

    for cls, count in damage_count.items():
        suggestion = repair_suggestions.get(cls, "No repair suggestion available.")
        st.markdown(f"**{cls}** — {count} detections")
        st.info(f"🛠️ {suggestion}")



   


# ================= FILE UPLOAD =======================
uploaded_file = st.file_uploader(
    "Choose an image or video...", 
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov",'webp']
)

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()


    # IMAGE INPUT HANDLING
    if file_ext in ["jpg", "jpeg", "png",'webp']:
        st.image(uploaded_file, caption="🖼️ Uploaded Image", width=400)
        st.markdown("### Running detection on image...")

        # Resize uploaded image if too large
        max_input_dim = 800
        img = Image.open(uploaded_file).convert("RGB")
        ratio = min(max_input_dim / img.width, max_input_dim / img.height, 1.0)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))
        frame_rgb = np.array(img)

        save_path = os.path.join(OUTPUT_DIR, "image_result.jpg")
        result_frame, detections = process_frame(frame_rgb, frame_num=0, save_path=save_path)

        if detections:
            detection_summary.extend(detections)
            st.success(f"✅ {len(detections)} damages detected.")
            st.image(result_frame, caption="🧭 Detected Damages", width=400)
            output_frames = [(0, save_path)]  # only one output frame
        else:
            st.warning("⚠️ No damages detected in this image.")
            output_frames = []

  
    # VIDEO INPUT HANDLING
    elif file_ext in ["mp4", "avi", "mov"]:
        st.video(uploaded_file)
        st.markdown("### Processing video...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp:
            temp.write(uploaded_file.read())
            video_path = temp.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 5
        frame_num = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                save_path = os.path.join(OUTPUT_DIR, f"frame_{frame_num}.jpg")
                result_frame, detections = process_frame(frame_rgb, frame_num, fps, save_path)

                if detections:  # skip empty frames
                    output_frames.append((frame_num, save_path))
                    detection_summary.extend(detections)

            frame_num += 1
            progress_bar.progress(min(frame_num / total_frames, 1.0))

        cap.release()
        st.success(f"✅ Video processed! Frames with detections: {len(output_frames)}")
# =====================================================

# ================= GALLERY DISPLAY ==================
if output_frames:
    if len(output_frames) > 1:
        st.markdown("---")
        st.subheader("🖼️ Detected Frames Gallery")

        images_per_row = 6
        clicked = None

        for i in range(0, len(output_frames), images_per_row):
            row_images = output_frames[i:i + images_per_row]
            cols = st.columns(len(row_images))

            for idx, (col, (frame_no, img_path)) in enumerate(zip(cols, row_images)):
                img = Image.open(img_path)
                with col:
                    if st.button(f"🔍 View {frame_no}", key=f"btn_{frame_no}"):
                        clicked = img_path
                    st.image(img, caption=f"Frame {frame_no}", use_container_width=True)

        if clicked:
            st.markdown("---")
            st.subheader("🔍 Enlarged View")
            st.image(Image.open(clicked), use_column_width=True)
# =====================================================

# ================= REPORT & REPAIR ==================
generate_detection_summary(detection_summary)
generate_repair_suggestions(detection_summary)
# ====================================================


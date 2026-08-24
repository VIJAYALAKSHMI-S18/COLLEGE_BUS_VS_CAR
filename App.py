import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import tempfile
import os

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(
    page_title="College Bus vs Car - Smart Vehicle Monitoring",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        text-align: center;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏫 Smart Campus Vehicle Monitoring System 🚗🚍</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Vehicle Detection, Classification & Tracking for College Campus Traffic</div>', unsafe_allow_html=True)

# ----------------------------
# LOAD MODELS
# ----------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_classifier_model():
    h5_path = "college_bus_vs_car_model.h5"
    if os.path.exists(h5_path):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(h5_path)
            return model
        except Exception:
            return None
    return None

yolo_model = load_yolo_model()
classifier_model = load_classifier_model()
yolo_class_names = yolo_model.names

# ----------------------------
# SIDEBAR NAVIGATION & SETTINGS
# ----------------------------
st.sidebar.title("Navigation & Settings")

menu = st.sidebar.radio(
    "📌 Select Mode",
    [
        "📸 Single Image Detection",
        "📂 Batch Images Detection",
        "🎥 CCTV / Video Tracking",
        "🧠 Custom Classifier (Bus vs Car)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Detection Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.35,
    step=0.05
)

selected_classes = st.sidebar.multiselect(
    "Vehicles to Monitor",
    options=["car", "bus", "truck", "motorcycle"],
    default=["car", "bus"]
)

# Helper function to convert YOLO plot (BGR) to RGB for Streamlit
def bgr_to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# =========================================================
# 🔹 1. SINGLE IMAGE MODE
# =========================================================
if menu == "📸 Single Image Detection":
    st.subheader("📸 Single Image Vehicle Detection & Counting")
    st.write("Upload an image to detect and count vehicles in the campus.")

    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    with col_up2:
        use_sample = st.checkbox("Or use sample image (Traffic.jpeg)", value=False)

    image_to_process = None
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file).convert("RGB")
    elif use_sample and os.path.exists("Traffic.jpeg"):
        image_to_process = Image.open("Traffic.jpeg").convert("RGB")

    if image_to_process is not None:
        img_np = np.array(image_to_process)

        with st.spinner("Detecting vehicles..."):
            results = yolo_model.predict(img_np, conf=conf_threshold, verbose=False)
            res = results[0]
            annotated_bgr = res.plot()
            annotated_rgb = bgr_to_rgb(annotated_bgr)

            # Count detections
            counts = {cls_name: 0 for cls_name in selected_classes}
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    cls_id = int(box.cls[0].item())
                    label = yolo_class_names.get(cls_id, "")
                    if label in counts:
                        counts[label] += 1

        # Display images side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_to_process, caption="Original Image", use_container_width=True)
        with col2:
            st.image(annotated_rgb, caption="Detection Result", use_container_width=True)

        st.markdown("### 📊 Detection Summary")
        cols = st.columns(len(selected_classes) + 1)
        total_vehicles = sum(counts.values())

        icons = {"car": "🚗", "bus": "🚍", "truck": "🚚", "motorcycle": "🏍️"}
        for idx, cls_name in enumerate(selected_classes):
            with cols[idx]:
                icon = icons.get(cls_name, "🚘")
                st.metric(label=f"{icon} {cls_name.capitalize()}s", value=counts[cls_name])

        with cols[-1]:
            st.metric(label="🚦 Total Vehicles", value=total_vehicles)


# =========================================================
# 🔹 2. BATCH MODE
# =========================================================
elif menu == "📂 Batch Images Detection":
    st.subheader("📂 Multiple Images Detection & Analysis")
    st.write("Upload multiple images at once to get detection results and an aggregated vehicle count.")

    uploaded_files = st.file_uploader(
        "Upload Multiple Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        total_counts = {cls_name: 0 for cls_name in selected_classes}
        st.write(f"Processing {len(uploaded_files)} images...")

        progress_bar = st.progress(0)

        for i, file in enumerate(uploaded_files):
            image = Image.open(file).convert("RGB")
            img_np = np.array(image)

            results = yolo_model.predict(img_np, conf=conf_threshold, verbose=False)
            res = results[0]
            annotated_rgb = bgr_to_rgb(res.plot())

            # Current image counts
            current_counts = {cls_name: 0 for cls_name in selected_classes}
            if res.boxes is not None:
                for box in res.boxes:
                    cls_id = int(box.cls[0].item())
                    label = yolo_class_names.get(cls_id, "")
                    if label in current_counts:
                        current_counts[label] += 1
                        total_counts[label] += 1

            st.markdown(f"#### 🖼️ {file.name}")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(annotated_rgb, caption=f"Detection for {file.name}", use_container_width=True)
            with col2:
                for cls_name in selected_classes:
                    st.write(f"**{cls_name.capitalize()}s:** {current_counts[cls_name]}")
                st.write(f"**Total in image:** {sum(current_counts.values())}")

            st.markdown("---")
            progress_bar.progress((i + 1) / len(uploaded_files))

        st.markdown("## 📊 Batch Overall Summary")
        cols = st.columns(len(selected_classes) + 1)
        icons = {"car": "🚗", "bus": "🚍", "truck": "🚚", "motorcycle": "🏍️"}
        for idx, cls_name in enumerate(selected_classes):
            with cols[idx]:
                icon = icons.get(cls_name, "🚘")
                st.metric(label=f"{icon} Total {cls_name.capitalize()}s", value=total_counts[cls_name])

        with cols[-1]:
            st.metric(label="🚦 Grand Total", value=sum(total_counts.values()))


# =========================================================
# 🔹 3. CCTV / VIDEO TRACKING MODE
# =========================================================
elif menu == "🎥 CCTV / Video Tracking":
    st.subheader("🎥 Smart CCTV Vehicle Tracking & Unique Counting")
    st.write("Upload a CCTV surveillance video or test with sample traffic video.")

    col_v1, col_v2 = st.columns([2, 1])
    with col_v1:
        video_file = st.file_uploader("Upload CCTV Video", type=["mp4", "avi", "mov", "mkv"])
    with col_v2:
        use_sample_vid = st.checkbox("Or use sample video (Traffic.mp4)", value=False)

    temp_video_path = None

    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.close()
        temp_video_path = tfile.name
    elif use_sample_vid and os.path.exists("Traffic.mp4"):
        temp_video_path = "Traffic.mp4"

    if temp_video_path:
        start_btn = st.button("▶️ Start Video Processing", key="start_video")
        stop_placeholder = st.empty()

        if start_btn:
            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

            stframe = st.empty()
            progress_bar = st.progress(0)

            col_m1, col_m2, col_m3 = st.columns(3)
            metric_cars = col_m1.empty()
            metric_buses = col_m2.empty()
            metric_total = col_m3.empty()

            counted_ids = set()
            total_car = 0
            total_bus = 0
            frame_idx = 0

            stop_processing = stop_placeholder.button("⏹️ Stop Processing", key="stop_video")

            try:
                while cap.isOpened():
                    if stop_processing:
                        st.warning("Video processing stopped by user.")
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1

                    # YOLO Tracking with ByteTrack
                    results = yolo_model.track(
                        frame,
                        persist=True,
                        conf=conf_threshold,
                        verbose=False
                    )

                    annotated_frame = results[0].plot()

                    # Check track IDs
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        boxes = results[0].boxes
                        ids = boxes.id.int().cpu().tolist()
                        classes = boxes.cls.int().cpu().tolist()

                        for track_id, cls_id in zip(ids, classes):
                            label = yolo_class_names.get(cls_id, "")

                            if track_id not in counted_ids:
                                counted_ids.add(track_id)
                                if label == "car":
                                    total_car += 1
                                elif label == "bus":
                                    total_bus += 1

                    # Overlay vehicle metrics on frame
                    cv2.putText(
                        annotated_frame,
                        f"Unique Cars: {total_car} | Unique Buses: {total_bus}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

                    # Display Live Video in Streamlit (convert BGR to RGB)
                    stframe.image(
                        bgr_to_rgb(annotated_frame),
                        use_container_width=True
                    )

                    # Update UI metrics
                    metric_cars.metric("🚗 Unique Cars", total_car)
                    metric_buses.metric("🚍 Unique Buses", total_bus)
                    metric_total.metric("🚦 Total Vehicles Tracked", total_car + total_bus)

                    progress_bar.progress(min(1.0, frame_idx / total_frames))

            finally:
                cap.release()
                if video_file is not None and os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except Exception:
                        pass

            st.success("✅ Video Processing Completed!")
            st.markdown("### 📊 Final Cumulative Vehicle Counts")
            col1, col2, col3 = st.columns(3)
            col1.metric("🚗 Total Cars Counted", total_car)
            col2.metric("🚍 Total Buses Counted", total_bus)
            col3.metric("🚦 Grand Total", total_car + total_bus)


# =========================================================
# 🔹 4. CUSTOM CLASSIFIER (MobileNetV2: Bus vs Car)
# =========================================================
elif menu == "🧠 Custom Classifier (Bus vs Car)":
    st.subheader("🧠 Deep Learning Image Classifier (MobileNetV2)")
    st.write("Classifies whether an individual image is a **College Bus** or a **Car** using the custom trained Deep Learning model.")

    if classifier_model is None:
        st.info("ℹ️ The standalone MobileNetV2 classifier model (.h5) requires a Python 3.11/3.10 environment with TensorFlow. You can use the high-performance **Single Image**, **Batch Images**, or **CCTV Video Tracking** modes in the sidebar (powered by YOLOv8) for real-time detection & counting!")
    else:
        uploaded_file = st.file_uploader("Upload Image to Classify", type=["jpg", "jpeg", "png"], key="classifier_upload")
        use_sample_car = st.checkbox("Or use sample Car image (Car.jpeg)", value=False)

        img_to_classify = None
        if uploaded_file is not None:
            img_to_classify = Image.open(uploaded_file).convert("RGB")
        elif use_sample_car and os.path.exists("Car.jpeg"):
            img_to_classify = Image.open("Car.jpeg").convert("RGB")

        if img_to_classify is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(img_to_classify, caption="Uploaded Image", use_container_width=True)

            with col2:
                # Preprocess for MobileNetV2 (128x128)
                img_resized = img_to_classify.resize((128, 128))
                img_array = np.array(img_resized, dtype=np.float32)
                # Model trained with Rescaling(1./255) inside or outside?
                # In notebook: normalization_layer was tf.keras.layers.Rescaling(1./255) in dataset pipeline or model?
                # Let's check: dataset map had Rescaling(1./255), so input to model should be normalized or raw?
                # If dataset had Rescaling(1./255), img_array / 255.0 is required!
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                with st.spinner("Classifying image..."):
                    prediction = classifier_model.predict(img_array, verbose=0)[0][0]

                    # In dataset: alphabetical class_names = ['bus', 'car'] -> bus=0, car=1
                    if prediction >= 0.5:
                        label = "Car 🚗"
                        confidence = float(prediction) * 100
                    else:
                        label = "College Bus 🚍"
                        confidence = float(1.0 - prediction) * 100

                st.markdown("### 🎯 Classification Result")
                st.markdown(f"#### Predicted Class: **{label}**")
                st.markdown(f"#### Confidence: **{confidence:.2f}%**")
                st.progress(int(confidence))
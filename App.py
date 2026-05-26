import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
class_names = model.names

# ----------------------------
# UI CONFIG
# ----------------------------
st.set_page_config(page_title="Smart Campus Vehicle System", layout="wide")

st.title("🏫 Smart Campus Vehicle Monitoring System 🚗🚍")

# ----------------------------
# SIDEBAR MENU
# ----------------------------
menu = st.sidebar.radio(
    "📌 Select Mode",
    ["Single Image", "Batch Images", "CCTV / Video"]
)

# =========================================================
# 🔹 SINGLE IMAGE MODE
# =========================================================
if menu == "Single Image":

    st.subheader("📸 Single Image Detection")

    file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

    if file:

        image = Image.open(file)
        img = np.array(image)

        results = model(img)

        annotated = results[0].plot()

        car_count = 0
        bus_count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = class_names[cls]

                if label == "car":
                    car_count += 1
                elif label == "bus":
                    bus_count += 1

        st.image(img, caption="Input Image", use_container_width=True)
        st.image(annotated, caption="Detection Result", use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("🚗 Cars", car_count)
        col2.metric("🚍 Buses", bus_count)
        col3.metric("🚦 Total", car_count + bus_count)

# =========================================================
# 🔹 BATCH MODE
# =========================================================
elif menu == "Batch Images":

    st.subheader("📂 Multiple Image Detection")

    files = st.file_uploader(
        "Upload Multiple Images",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True
    )

    if files:

        total_car = 0
        total_bus = 0

        for file in files:

            image = Image.open(file)
            img = np.array(image)

            results = model(img)
            annotated = results[0].plot()

            car_count = 0
            bus_count = 0

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = class_names[cls]

                    if label == "car":
                        car_count += 1
                        total_car += 1
                    elif label == "bus":
                        bus_count += 1
                        total_bus += 1

            st.image(annotated, caption=file.name)

        st.markdown("## 📊 Batch Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("🚗 Total Cars", total_car)
        col2.metric("🚍 Total Buses", total_bus)
        col3.metric("🚦 Grand Total", total_car + total_bus)

# =========================================================
# 🔹 CCTV / VIDEO MODE
# =========================================================
# =========================================================
# 🔹 CCTV / VIDEO MODE
# =========================================================
elif menu == "CCTV / Video":

    st.subheader("🎥 Smart CCTV Vehicle Counting System")

    video_file = st.file_uploader(
        "Upload CCTV Video",
        type=["mp4", "avi", "mov"]
    )

    if video_file:

        # Save uploaded video
        with open("temp.mp4", "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture("temp.mp4")

        stframe = st.empty()

        # Total Counts
        total_car = 0
        total_bus = 0

        # Track IDs to avoid duplicate counting
        counted_ids = set()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            # YOLO Tracking
            results = model.track(
                frame,
                persist=True,
                conf=0.4
            )

            annotated_frame = results[0].plot()

            # Current frame counts
            frame_car = 0
            frame_bus = 0

            if results[0].boxes.id is not None:

                boxes = results[0].boxes
                ids = boxes.id.cpu().numpy().astype(int)
                classes = boxes.cls.cpu().numpy().astype(int)

                for track_id, cls in zip(ids, classes):

                    label = class_names[cls]

                    # Count only once
                    if track_id not in counted_ids:

                        counted_ids.add(track_id)

                        if label == "car":
                            total_car += 1

                        elif label == "bus":
                            total_bus += 1

                    # Live Frame Count
                    if label == "car":
                        frame_car += 1

                    elif label == "bus":
                        frame_bus += 1

            # Display Text on Video
            cv2.putText(
                annotated_frame,
                f"Cars: {total_car}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            cv2.putText(
                annotated_frame,
                f"Buses: {total_bus}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                2
            )

            cv2.putText(
                annotated_frame,
                f"Total Vehicles: {total_car + total_bus}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
                2
            )

            # Show Live Video
            stframe.image(
                annotated_frame,
                channels="BGR",
                use_container_width=True
            )

        cap.release()

        # Final Summary
        st.success("✅ CCTV Video Processing Completed")

        st.markdown("## 📊 Final Vehicle Count")

        col1, col2, col3 = st.columns(3)

        col1.metric("🚗 Cars", total_car)
        col2.metric("🚍 Buses", total_bus)
        col3.metric("🚦 Total", total_car + total_bus)
import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import pandas as pd
import calendar
import datetime
import pickle
from ultralytics import YOLO
from tensorflow.keras.models import load_model


st.set_page_config(page_title="Traffic Monitor", layout="wide")

# Increase Streamlit maximum upload size to 1 GB (value in MB)
#st.set_option('server.maxUploadSize', 1024)


@st.cache_resource(show_spinner=False)
def load_models(yolo_path: str = "yolov8n.pt", traffic_model_path: str = "traffic_nn_model.h5"):
    yolo = YOLO(yolo_path)
    traffic_model = load_model(traffic_model_path)
    return yolo, traffic_model


def create_video_writer(output_path, frame_width, frame_height, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


def process_video(input_path, output_path, yolo, traffic_model, conf_threshold=0.5, meters_per_pixel=0.05, interval_seconds=3):
    cap = cv2.VideoCapture(input_path)
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps * interval_seconds)
    writer = create_video_writer(output_path, video_width, video_height, fps)

    # Load encoders if present
    label_encoder_labels = None
    try:
        with open('label_encoder_labels.pkl', 'rb') as f:
            label_encoder_labels = pickle.load(f)
    except Exception:
        label_encoder_labels = None

    prev_centers = {}
    vehicle_speeds = {}
    frame_data_list = []
    frame_count = 0

    vehicle_classes = ['car', 'motorbike', 'bus', 'truck']

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = yolo.track(frame, persist=True, verbose=False)

        car_count = bike_count = bus_count = truck_count = 0

        if results and len(results) > 0:
            boxes = results[0].boxes
            ids = boxes.id
            names = results[0].names

            if ids is not None:
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    if conf < conf_threshold:
                        continue

                    cls_id = int(box.cls[0])
                    cls_name = names[cls_id]

                    if cls_name == "car":
                        car_count += 1
                    elif cls_name == "motorbike":
                        bike_count += 1
                    elif cls_name == "bus":
                        bus_count += 1
                    elif cls_name == "truck":
                        truck_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    obj_id = int(ids[i])

                    if obj_id in prev_centers:
                        prev_x, prev_y = prev_centers[obj_id]
                        pixel_distance = np.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)
                        real_distance_m = pixel_distance * meters_per_pixel
                        speed_m_s = real_distance_m * fps
                        speed_kmh = speed_m_s * 3.6
                        vehicle_speeds[obj_id] = speed_kmh

                    prev_centers[obj_id] = (cx, cy)

                    speed_text = f"{vehicle_speeds.get(obj_id, 0):.1f} km/h"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {obj_id} {cls_name}: {speed_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if frame_count % frame_interval == 0:
            now = datetime.datetime.now()
            total = car_count + bike_count + bus_count + truck_count
            X = pd.DataFrame([{
                "Time": now.strftime("%H:%M:%S"),
                "Date": now.strftime("%Y-%m-%d"),
                "DayOfWeek": calendar.day_name[now.weekday()],
                "CarCount": car_count,
                "BikeCount": bike_count,
                "BusCount": bus_count,
                "TruckCount": truck_count,
                "Total": total
            }])

            try:
                prediction = traffic_model.predict(X[["CarCount", "BikeCount", "BusCount", "TruckCount", "Total"]])[0]
                if label_encoder_labels is not None:
                    pred_label = label_encoder_labels.inverse_transform([np.argmax(prediction)])[0]
                else:
                    pred_label = str(np.argmax(prediction))
            except Exception:
                pred_label = "N/A"

            record = {
                "Time": now.strftime("%H:%M:%S"),
                "Date": now.strftime("%Y-%m-%d"),
                "DayOfWeek": calendar.day_name[now.weekday()],
                "CarCount": car_count,
                "BikeCount": bike_count,
                "BusCount": bus_count,
                "TruckCount": truck_count,
                "Total": total,
                "TrafficSituation": pred_label
            }
            frame_data_list.append(record)

        writer.write(frame)

    cap.release()
    writer.release()

    df = pd.DataFrame(frame_data_list)
    df.to_csv(output_path + '.csv', index=False)
    return output_path, output_path + '.csv'


def main():
    st.title("Traffic Monitoring — Video to Congestion Prediction")
    st.markdown("Upload a video or choose an existing one in the workspace. The app will run the YOLO tracker and the traffic prediction model and return an annotated video and a CSV with predictions.")

    with st.sidebar:
        st.header("Settings")

        # Small visual flow description with icons
        st.markdown("**Flow**")
        st.markdown("- 📤 **Upload / Select**: provide a video file (mp4, mov, avi).")
        st.markdown("- ⚙️ **Configure**: adjust confidence and prediction interval.")
        st.markdown("- ▶️ **Run**: start analysis; output is an annotated video and CSV.")
        st.caption("Tip: first run may take longer while models load.")

        # Allow choosing YOLO model
        model_choice = st.selectbox("📦 YOLO model", options=["yolov8n.pt", "yolov8s.pt"], index=0, help="Choose a detection model: smaller is faster, larger is more accurate.")

        st.markdown("---")

        conf_threshold = st.slider("🔎 Confidence threshold", 0.0, 1.0, 0.5, 0.05, help="Filter detections below this confidence.")
        interval_seconds = st.number_input("⏱️ Prediction interval (seconds)", min_value=1, max_value=300, value=3, help="How often (seconds) to aggregate counts and predict congestion.")
        run_button = st.button("▶️ Run analysis")

    uploaded_file = st.file_uploader("Upload a video file (mp4, mov, avi)", type=["mp4", "mov", "avi"], help="Max file size: 200 MB")
    sample_files = [f for f in os.listdir('.') if f.lower().endswith(('.mp4', '.mov', '.avi'))]
    selected = st.selectbox("Or select an existing video in this folder", options=["-- choose --"] + sample_files)

    st.info("Models are loaded once; first load may take some time.")
    yolo, traffic_model = load_models(yolo_path=model_choice)

    if run_button:
        if uploaded_file is None and (selected == "-- choose --"):
            st.warning("Please upload a video or select one from the folder.")
            return

        tmp_input = None
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1])
            tfile.write(uploaded_file.read())
            tfile.flush()
            tmp_input = tfile.name
        else:
            tmp_input = selected

        out_video_path = os.path.join(tempfile.gettempdir(), f"annotated_{os.path.basename(tmp_input)}")

        with st.spinner("Processing video — this can take a while depending on length and model size..."):
            video_path, csv_path = process_video(tmp_input, out_video_path, yolo, traffic_model, conf_threshold=conf_threshold, interval_seconds=interval_seconds)

        st.success("Processing finished")
        st.video(video_path)

        with open(video_path, 'rb') as f:
            video_bytes = f.read()
            st.download_button("Download annotated video", data=video_bytes, file_name=os.path.basename(video_path), mime='video/mp4')

        with open(csv_path, 'rb') as f:
            csv_bytes = f.read()
            st.download_button("Download CSV", data=csv_bytes, file_name=os.path.basename(csv_path), mime='text/csv')


if __name__ == '__main__':
    main()

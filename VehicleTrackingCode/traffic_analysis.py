import numpy as np
import datetime
import cv2
import pandas as pd
import calendar
import json
from ultralytics import YOLO
from azure.eventhub import EventHubProducerClient, EventData
from tensorflow.keras.models import load_model
import pickle
# -------------------- Helper function --------------------
def create_video_writer(output_path, frame_width, frame_height, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# -------------------- Configuration --------------------
conf_threshold = 0.5
meters_per_pixel = 0.05
fps_smooth = 30
interval_seconds = 3  # record every 3 seconds

# -------------------- Load YOLO model --------------------
model = YOLO("yolov8n.pt")

# -------------------- Load traffic congestion prediction model (h5 model) --------------------
traffic_model = load_model("traffic_nn_model.h5")

# -------------------- Azure Event Hub Setup --------------------
CONNECTION_STR = "Endpoint=sb://traffic-stream-namespace.servicebus.windows.net/;SharedAccessKeyName=SendPolicy;SharedAccessKey=k7BdRAHHyJhd+plaj4dN+GPsuP5QuYpP7+AEhHs6s5I=;EntityPath=traffic-stream"
EVENT_HUB_NAME = "traffic-stream"
producer = EventHubProducerClient.from_connection_string(conn_str=CONNECTION_STR, eventhub_name=EVENT_HUB_NAME)

# -------------------- Initialize video --------------------
video_cap = cv2.VideoCapture("15 minutes of heavy traffic noise in India _ 14-08-2022(720P_HD).mp4")
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = video_cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = fps_smooth

frame_interval = int(fps * interval_seconds)
writer = create_video_writer("output_speed.mp4", video_width, video_height, fps)

# -------------------- Vehicle tracking storage --------------------
prev_centers = {}
vehicle_speeds = {}

# -------------------- Data storage --------------------
frame_data_list = []
frame_count = 0

# YOLO vehicle classes
vehicle_classes = ['car', 'motorbike', 'bus', 'truck']
# loop encoding files
with open('day_label_encoder.pkl', 'rb') as f:
    label_encoder_days = pickle.load(f)
with open('label_encoder_labels.pkl', 'rb') as f:
    label_encoder_labels = pickle.load(f)
print("🚦 Starting real-time traffic analytics...")

while True:
    start = datetime.datetime.now()
    ret, frame = video_cap.read()
    if not ret:
        print("End of the video file.")
        break

    frame_count += 1
    current_frame_data = []

    results = model.track(frame, persist=True, verbose=False)

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

                # Count types of vehicles
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

                # Speed calculation
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
                cv2.putText(frame, f"ID {obj_id} {cls_name}: {speed_text}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Every interval (e.g., 3 sec)
    if frame_count % frame_interval == 0:
        now = datetime.datetime.now()
        total = car_count + bike_count + bus_count + truck_count

        # Prepare data for ML model
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

        # Predict traffic congestion
        prediction = traffic_model.predict(X[["CarCount", "BikeCount", "BusCount", "TruckCount", "Total"]])[0]
        prediction = label_encoder_labels.inverse_transform([np.argmax(prediction)])[0]
        # Record data
        record = {
            "Time": now.strftime("%H:%M:%S"),
            "Date": now.strftime("%Y-%m-%d"),
            "DayOfWeek": calendar.day_name[now.weekday()],
            "CarCount": car_count,
            "BikeCount": bike_count,
            "BusCount": bus_count,
            "TruckCount": truck_count,
            "Total": total,
            "TrafficSituation": str(prediction)
        }
        frame_data_list.append(record)

        # Send to Azure Event Hub
        try:
            event_data = EventData(json.dumps(record))
            producer.send_batch([event_data])
            print(f"📤 Sent to Event Hub: {record}")
            print("✅ Data sent to Event Hub:", event_data)
        except Exception as e:
            print(f"⚠️ Error sending to Event Hub: {e}")

    # Display
    writer.write(frame)
    frame_display = cv2.resize(frame, (800, 600))
    cv2.imshow("Traffic Monitoring", frame_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()

# -------------------- Save data locally as backup --------------------
df = pd.DataFrame(frame_data_list)
df.to_csv("traffic_data_with_prediction.csv", index=False)
print("✅ Data saved locally as 'traffic_data_with_prediction.csv'")
print(df.head())

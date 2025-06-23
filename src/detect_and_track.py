import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

def detect_and_track(video_path, model_path, output_path):
    model = YOLO(model_path)
    tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.4)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    track_id_to_label = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            cls = int(cls)

            if cls == 0:
                label = "ball"
            elif cls == 1:
                label = "goalkeeper"
            elif cls == 2:
                label = "player"
            elif cls == 3:
                label = "referee"
            else:
                continue

            bbox = [x1, y1, x2 - x1, y2 - y1]  # x, y, w, h
            detections.append((bbox, conf, label))

        # Update tracker and receive matching info
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w, h = track.to_tlwh()
            r, b = int(l + w), int(t + h)

            # Use detection info if matched
            if track.det_class:
                track_id_to_label[track_id] = track.det_class

            label = track_id_to_label.get(track_id, "unknown")

            # Assign colors based on true label
            if label == "player":
                color = (255, 0, 0)
            elif label == "goalkeeper":
                color = (0, 0, 255)
            elif label == "ball":
                color = (0, 255, 0)
            elif label == "referee":
                color = (0, 255, 255)
            else:
                color = (255, 255, 255)

            cv2.rectangle(frame, (int(l), int(t)), (r, b), color, 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"📸 Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"✅ Output saved to: {output_path}")

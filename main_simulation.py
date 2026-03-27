import cv2
import numpy as np
from lane_detection import detect_lanes, region_of_interest
from ultralytics import YOLO

# -------- LOAD MODELS --------
general_model = YOLO("yolov8n.pt")      # cars, people, etc.
pothole_model = YOLO("best.pt")      # YOUR trained model

cap = cv2.VideoCapture("testv1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))
    height, width, _ = frame.shape

    # -------- LANE PROCESSING --------
    edges, roi_edges = detect_lanes(frame)
    roi_mask = region_of_interest(edges)

    # -------- YOLOv8 GENERAL DETECTION --------
    results_general = general_model(frame)[0]

    for box in results_general.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = general_model.names[class_id]

        # Center point
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # ROI FILTER
        if roi_mask[cy, cx] == 255:

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # -------- POTHOLE DETECTION --------
    results_pothole = pothole_model(frame)[0]

    for box in results_pothole.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # ROI FILTER (VERY IMPORTANT)
        if roi_mask[cy, cx] == 255:

            # 🔥 RED box for potholes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, "POTHOLE", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # -------- THERMAL VIEW --------
    thermal = cv2.applyColorMap(edges, cv2.COLORMAP_JET)

    # -------- DISPLAY --------
    cv2.imshow("Final Output (ROI Filtered)", frame)
    cv2.imshow("Edges", edges)
    cv2.imshow("ROI", roi_mask)
    cv2.imshow("Thermal", thermal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
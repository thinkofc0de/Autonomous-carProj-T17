import cv2
import numpy as np
from lane_detection import detect_lanes, region_of_interest
from ultralytics import YOLO

# -------- LOAD MODELS --------
general_model = YOLO("yolov8n.pt")
pothole_model = YOLO("best.pt")

cap = cv2.VideoCapture("testv2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))
    height, width, _ = frame.shape

    # -------- DEFAULT DECISION --------
    decision = "GO"
    delay = 1

    # -------- LANE + ROI --------
    edges, roi_edges = detect_lanes(frame)
    roi_mask = region_of_interest(edges)

    # -------- EDGE DETECTION BANDS --------
    INNER_EDGE_Y = int(height * 0.40)
    OUTER_EDGE_Y = int(height * 0.35)
    EDGE_THICKNESS = 10

    # -------- YOLO GENERAL DETECTION --------
    results_general = general_model(frame)[0]

    allowed_classes = [
        "person",
        "car", "motorbike", "bus", "truck", "bicycle",
        "dog", "cat", "cow", "horse"
    ]

    for box in results_general.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        label = general_model.names[class_id]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # ✅ STRICT ROI FILTER (RESTORED)
        if roi_mask[cy, cx] != 255:
            continue

        if label in allowed_classes:

            # 🔴 INNER EDGE → STOP
            if INNER_EDGE_Y - EDGE_THICKNESS <= cy <= INNER_EDGE_Y + EDGE_THICKNESS:
                decision = "STOP"
                delay = 0

            # 🟡 OUTER EDGE → SLOW
            elif OUTER_EDGE_Y - EDGE_THICKNESS <= cy <= OUTER_EDGE_Y + EDGE_THICKNESS:
                if decision != "STOP":
                    decision = "SLOW"
                    delay = 100

            # Draw detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # -------- POTHOLE DETECTION --------
    results_pothole = pothole_model(frame)[0]

    for box in results_pothole.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if roi_mask[cy, cx] == 255:
            decision = "STOP"
            delay = 0

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, "POTHOLE", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # -------- DRAW OUTER TRAPEZIUM --------
    big_polygon = np.array([[
        (0, int(height * 0.65)),
        (width, int(height * 0.65)),
        (int(width * 0.60), int(height * 0.35)),
        (int(width * 0.40), int(height * 0.35))
    ]], np.int32)

    cv2.polylines(frame, big_polygon, isClosed=True, color=(0, 0, 255), thickness=2)

    # -------- KEEP DEBUG LINES (OPTIONAL) --------
    cv2.line(frame, (0, INNER_EDGE_Y), (width, INNER_EDGE_Y), (0,255,0), 2)
    cv2.line(frame, (0, OUTER_EDGE_Y), (width, OUTER_EDGE_Y), (0,0,255), 2)

    # -------- THERMAL VIEW --------
    thermal = cv2.applyColorMap(edges, cv2.COLORMAP_JET)

    # -------- DECISION DISPLAY --------
    color = (0,255,0)
    if decision == "SLOW":
        color = (0,255,255)
    elif decision == "STOP":
        color = (0,0,255)

    cv2.putText(frame, f"Decision: {decision}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # ✅ RESTORED ALL WINDOWS
    cv2.imshow("Final Output", frame)
    cv2.imshow("Edges", edges)
    cv2.imshow("ROI", roi_mask)
    cv2.imshow("Thermal", thermal)

    # -------- SPEED CONTROL --------
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape

    # 🔥 Expand ROI slightly upward (fix edge detection issue)
    top_y = int(height * 0.35)   # was 0.4 → now 0.35

    polygon = np.array([[
        (0, int(height * 0.6)),
        (width, int(height * 0.6)),
        (int(width * 0.55), top_y),
        (int(width * 0.45), top_y)
    ]], np.int32)

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)

    return mask


def detect_lanes(frame):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Blur (reduce noise)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny edges
    edges = cv2.Canny(blur, 50, 150)

    # 4. ROI mask (focus only road)
    roi = region_of_interest(edges)

    # -------- DRAW ROI BOUNDARY --------
    height, width, _ = frame.shape

    roi_points = np.array([[
        (0, int(height * 0.6)),
        (width, int(height * 0.6)),
        (int(width * 0.55), int(height * 0.4)),
        (int(width * 0.45), int(height * 0.4))
    ]], np.int32)

    cv2.polylines(frame, roi_points, isClosed=True, color=(255, 0, 0), thickness=2)

    return edges, roi
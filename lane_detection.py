import cv2
import numpy as np

def region_of_interest(img):
    height, width = img.shape

    # 🔥 Move ROI upward (ignore bonnet)
    polygon = np.array([[
        (0, int(height * 0.6)),        # left bottom moved UP
        (width, int(height * 0.6)),    # right bottom moved UP
        (int(width * 0.55), int(height * 0.4)),  # top right
        (int(width * 0.45), int(height * 0.4))   # top left
    ]], np.int32)

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)

    masked_image = cv2.bitwise_and(img, mask)
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
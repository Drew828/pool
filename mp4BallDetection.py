import cv2
import numpy as np
import sqlite3

# -------------------------------
# 1. Connect to SQLite database
# -------------------------------
conn = sqlite3.connect('pool_data.db')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS ball_detections (
        frame_id INTEGER,
        circle_id INTEGER,
        x REAL,
        y REAL,
        radius REAL,
        PRIMARY KEY (frame_id, circle_id)
    )
''')
conn.commit()

# ------------------------------------------------
# 2. Detect balls (circles) in the frame using HoughCircles
# ------------------------------------------------
import cv2
import numpy as np

def detect_balls_in_frame(image_bgr):
    """
    Uses both HoughCircles and contour detection to find pool balls,
    excluding detections within 5% of the periphery of the frame.
    Returns a list of detected (x, y, r) circles.
    """
    height, width = image_bgr.shape[:2]
    margin_x = int(width * 0.07) #0.05 pretty good but not perfect,
    margin_y = int(height * 0.07)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Adaptive Thresholding for better ball detection
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Canny Edge Detection
    edges = cv2.Canny(thresh, 50, 150)

    # 1. Hough Circle Detection
    hough_circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                     param1=50, param2=30, minRadius=10, maxRadius=20)
    
    hough_detected = []
    if hough_circles is not None:
        for circle in np.uint16(np.around(hough_circles[0, :])):
            x, y, r = circle
            if margin_x < x < width - margin_x and margin_y < y < height - margin_y:
                hough_detected.append((x, y, r))

    # 2. Contour Detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_detected = []

    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        x, y, radius = int(x), int(y), int(radius)

        if 10 < radius < 20 and margin_x < x < width - margin_x and margin_y < y < height - margin_y:
            contour_detected.append((x, y, radius))

    # Merge results from both methods
    final_detections = hough_detected + contour_detected

    return final_detections




# ------------------------------------------------
# 3. Load MP4 and process each frame
# ------------------------------------------------
video_path = 'PoolVid.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video was opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_idx = 0
while True:
    ret, frame_bgr = cap.read()  # Read one frame
    if not ret:
        break  # Exit when no frames are left

    # ------------------------------------------------
    # 4. Detect balls (circles) in the frame
    # ------------------------------------------------
    circles_in_frame = detect_balls_in_frame(frame_bgr)

    # For demonstration: store each circle in the DB
    for circle_id, (x, y, r) in enumerate(circles_in_frame):
        cursor.execute(''' 
            INSERT OR REPLACE INTO ball_detections (frame_id, circle_id, x, y, radius) 
            VALUES (?, ?, ?, ?, ?) 
        ''', (frame_idx, circle_id, x, y, r))

    # (Optional) visualize or debug
    # We can draw the circles on the frame and show them in a pop-up (disabled by default)
    for (x, y, r) in circles_in_frame:
        cv2.circle(frame_bgr, (x, y), r, (0, 255, 0), 2)

    cv2.imshow("Detected Circles", frame_bgr)
    cv2.waitKey(30)  # Adjust wait time as needed (e.g., 30ms)

    frame_idx += 1

# Clean up
cap.release()
conn.commit()
conn.close()

# If you used cv2.imshow() above, uncomment the following when done:
cv2.destroyAllWindows()

print("Processing complete. Ball detections saved to pool_data.db.")

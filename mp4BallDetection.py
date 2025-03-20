import cv2
import numpy as np
import sqlite3
import os

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

# Create directory for frames
output_dir = "frames_output"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------
# 2. Detect balls (circles) in the frame using HoughCircles
# ------------------------------------------------
def detect_balls_in_frame(image_bgr):
    height, width = image_bgr.shape[:2]
    margin_x = int(width * 0.07)
    margin_y = int(height * 0.07)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    edges = cv2.Canny(thresh, 50, 150)

    hough_circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                     param1=50, param2=30, minRadius=10, maxRadius=20)
    
    hough_detected = []
    if hough_circles is not None:
        for circle in np.uint16(np.around(hough_circles[0, :])):
            x, y, r = circle
            if margin_x < x < width - margin_x and margin_y < y < height - margin_y:
                hough_detected.append((x, y, r))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_detected = []

    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        x, y, radius = int(x), int(y), int(radius)

        if 10 < radius < 20 and margin_x < x < width - margin_x and margin_y < y < height - margin_y:
            contour_detected.append((x, y, radius))

    final_detections = hough_detected + contour_detected

    return final_detections

# ------------------------------------------------
# 3. Load MP4 and process each frame
# ------------------------------------------------
video_path = 'PoolVid.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_idx = 0
for i in range(50):
    ret, frame_bgr = cap.read()
    if not ret:
        break

    # ------------------------------------------------
    # 4. Detect balls (circles) in the frame
    # ------------------------------------------------
    circles_in_frame = detect_balls_in_frame(frame_bgr)
    print('entered step 4')

    for circle_id, (x, y, r) in enumerate(circles_in_frame):
        cursor.execute(''' 
            INSERT OR REPLACE INTO ball_detections (frame_id, circle_id, x, y, radius) 
            VALUES (?, ?, ?, ?, ?) 
        ''', (frame_idx, circle_id, x, y, r))

    # (Optional) visualize or debug
    for circle_id, (x, y, r) in enumerate(circles_in_frame):
        cv2.circle(frame_bgr, (x, y), r, (0, 255, 0), 2)
        cv2.putText(frame_bgr, str(circle_id), (x - 10, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save frame as image
   # frame_filename = os.path.join(output_dir, f"frame_{frame_idx}.png")
   # cv2.imwrite(frame_filename, frame_bgr)
    
    cv2.imshow("Detected Circles", frame_bgr)
    cv2.waitKey(30)

    frame_idx += 1

cap.release()
conn.commit()
conn.close()

cv2.destroyAllWindows()

print("Processing complete. Ball detections saved to pool_data.db. Frames saved in frames_output directory.")

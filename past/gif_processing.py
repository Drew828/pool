import cv2
import numpy as np
import imageio
from PIL import Image
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
# 2. Load GIF and convert to a list of frames
# ------------------------------------------------
gif_path = 'pool_table.gif'
gif = imageio.mimread(gif_path)  # Reads all frames of the GIF into a list

# Convert each frame (a numpy array) to a format OpenCV can handle consistently
frames = []
for frame in gif:
    # frame is in (height, width, channels) format
    # Convert to BGR color space if needed for OpenCV
    # imageio typically returns RGBA or RGB arrays
    if frame.shape[2] == 4:
        # If RGBA, convert to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else:
        # If RGB, convert to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
   
    frames.append(frame_bgr)

# ------------------------------------------------
# 3. Detect circles (naive approach with HoughCircles)
# ------------------------------------------------
def detect_balls_in_frame(image_bgr):
    """
    Returns a list of (x, y, r) circle parameters detected by HoughCircles.
    Adjust params as needed for your scenario.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough Circle detection (tune params carefully)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=12.5, #50 weak, 1 too noisy, 12.5 and 25 were decent
        param2=25,   # 15 detected all circuls but too much noise 60 stopped all detection, 30 allowed for some
        minRadius=5,
        maxRadius=30  # adjust based on scale of your GIF
    )

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        return circles.tolist()  # list of [x, y, r]
    else:
        return []

# -------------------------------
# 4. (Optional) Simple Tracker
# -------------------------------
# For now, we’ll just detect circles in each frame and store them.
# A real tracker would match circle IDs across frames (e.g., Hungarian algorithm, Kalman Filter, etc.).
# We'll assign circle IDs = index of circle in the detection list for that frame.

for frame_idx, frame_bgr in enumerate(frames):
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
    cv2.waitKey(200)  # Wait 200ms just to see the result step by step

conn.commit()
conn.close()

# If you used cv2.imshow() above, uncomment the following when done:
cv2.destroyAllWindows()

print("Processing complete. Ball detections saved to pool_data.db.")
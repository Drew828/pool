import cv2
import numpy as np
import sqlite3

def get_dominant_color(image, x, y, radius):
    """
    Extracts the dominant color of a detected ball using HSV color space.
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 255, -1)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    
    hist = cv2.calcHist([masked_hsv], [0], mask, [180], [0, 180])
    dominant_hue = np.argmax(hist)
    
    # Check if the ball is white (low saturation and high value in HSV)
    mask_saturation = cv2.calcHist([masked_hsv], [1], mask, [256], [0, 256])
    mask_value = cv2.calcHist([masked_hsv], [2], mask, [256], [0, 256])
    avg_saturation = np.mean(mask_saturation)
    avg_value = np.mean(mask_value)
    
    if avg_saturation < 30 and avg_value > 200:
        return 'white'
    
    return dominant_hue

def classify_ball_color(hue):
    """
    Matches the dominant hue to the corresponding billiard ball color.
    """
    if hue == 'white':
        return 'white'
    elif 20 < hue < 35:
        return 'yellow'
    elif 90 < hue < 120:
        return 'blue'
    elif 0 < hue < 10 or 170 < hue < 180:
        return 'red'
    elif 50 < hue < 80:
        return 'green'
    elif 130 < hue < 160:
        return 'purple'
    elif 10 < hue < 20:
        return 'orange'
    elif 140 < hue < 150:
        return 'mauve'
    elif 0 < hue < 180:  # Broad range for black
        return 'black'
    return None

def is_striped(image, x, y, radius):
    """
    Uses Canny edge detection to determine if a ball is striped or solid.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    mask = np.zeros_like(edges)
    cv2.circle(mask, (x, y), radius, 255, -1)
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    edge_density = np.sum(masked_edges) / (np.pi * radius ** 2)
    return edge_density > 0.2  # Threshold for striped detection

def process_ball_detections():
    """
    Reads ball detections, classifies them, and stores results in the database.
    """
    conn = sqlite3.connect('pool_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS ball_identity (
        frame_id INTEGER,
        color TEXT,
        striped TEXT,
        x REAL,
        y REAL,
        PRIMARY KEY (frame_id, color, striped)
    )''')
    conn.commit()
    
    cursor.execute("SELECT frame_id, x, y, radius FROM ball_detections")
    detections = cursor.fetchall()
    
    cap = cv2.VideoCapture('PoolVid.mp4')
    frame_cache = {}
    
    for frame_id, x, y, radius in detections:
        if frame_id not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_cache[frame_id] = frame
        
        frame = frame_cache[frame_id]
        hue = get_dominant_color(frame, x, y, radius)
        color = classify_ball_color(hue)
        if not color:
            continue
        
        if color == 'white':
            striped = 'no'
        else:
            striped = 'yes' if is_striped(frame, x, y, radius) else 'no'
        
        cursor.execute('''INSERT OR REPLACE INTO ball_identity (frame_id, color, striped, x, y)
                          VALUES (?, ?, ?, ?, ?)''', (frame_id, color, striped, x, y))
    
    conn.commit()
    conn.close()
    cap.release()
    print("Ball identity data saved successfully.")

if __name__ == "__main__":
    process_ball_detections()

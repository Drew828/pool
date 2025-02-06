import cv2
import numpy as np

# Load the image
image = cv2.imread('pool_table.jpg')

# Step 1: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Gaussian blur to reduce noise and improve circle detection
gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)

# Step 3: Detect circles using Hough Circle Transform
circles = cv2.HoughCircles(
    gray_blurred, 
    cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=50
)

# Check if circles were detected
if circles is not None:
    # Convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # Loop over the circles and draw them on the image
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)  # Draw the circle in green
        cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # Mark the center
    
    # Show the output image with the detected circles
    cv2.imshow("Detected Circles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No circles were detected.")

# Optional: Save the output image
cv2.imwrite("pool_table_detected.jpg", image)

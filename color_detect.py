import cv2
import numpy as np
from sklearn.cluster import KMeans

# Define the bounding box (Area of Interest)
x1, y1, x2, y2 = 100, 100, 400, 400

def identify_color(rgb):
    """Identify the color name from RGB values."""
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    color_ranges = {
        "red": [(0, 50, 50), (10, 255, 255)],
        "yellow": [(20, 50, 50), (30, 255, 255)],
        "green": [(35, 50, 50), (85, 255, 255)],
        "blue": [(90, 50, 50), (130, 255, 255)],
        "brown": [(10, 20, 70), (20, 100, 200)],
    }
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        if cv2.inRange(np.uint8([[hsv]]), lower, upper):
            return color_name
    return "unknown"

def get_dominant_color(image, k=3):
    """Find the dominant color in an image using K-Means clustering."""
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = labels[np.argmax(counts)]
    dominant_color = kmeans.cluster_centers_[dominant_cluster]
    return [int(c) for c in dominant_color]

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crop the frame to the bounding box (ROI)
    roi = frame[y1:y2, x1:x2]
    roi = cv2.convertScaleAbs(roi, alpha=1.2, beta=10)  # Improve contrast

    # Find the dominant color in the ROI
    dominant_color = get_dominant_color(roi)
    color_label = identify_color(dominant_color)

    # Print the dominant color
    print(f"Dominant Color: {color_label} (RGB: {dominant_color})")

    # Display the ROI with the detected color label
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Color: {color_label}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the webcam feed with the bounding box
    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
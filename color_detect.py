import cv2
import numpy as np
from sklearn.cluster import KMeans

# Define the bounding box (Area of Interest)
# x1, y1 is the location to start drawing the ROI box from upper left corner
# x2, y2 is the size of the ROI box to draw from the start point.
x1, y1, x2, y2 = 100, 100, 200, 200

def identify_color(rgb):
    """
    identify color by name based on rgb value
    :param rgb:
    :return:
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_BGR2HSV)[0][0]
    # hsv above will be a numpy.ndarray

    # Define HSV ranges for colors - Hue Saturation Values
    # https://irlxd.com/rgb-bgr-hsv-for-newbies-who-cant-get-opencv-colors-working
    # https://www.lifewire.com/what-is-hsv-in-design-1078068
    # Color ranges in (Upper), and (Lower) HSV Values
    color_ranges = {
        "red": [(0, 50, 50), (10, 255, 255)],
        "orange": [(10, 50, 50), (25, 255, 255)],
        "yellow": [(25, 50, 50), (35, 255, 255)],
        "green": [(35, 50, 50), (85, 255, 255)],
        "blue": [(90, 50, 50), (130, 255, 255)],
        "brown": [(10, 20, 70), (20, 100, 200)],
    }

    # Check which range the HSV value falls into
    for color_name, (lower, upper) in color_ranges.items():
        # iterate over each color in color_ranges and see where the HSV falls.
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        if cv2.inRange(np.uint8([[hsv]]), lower, upper):
            return color_name

    return "unknown"  # If no match found

def get_dominant_color(image, k=5):
    """
    Find the dominant color in an image using K-Means clustering.
    https://www.geeksforgeeks.org/k-means-clustering-introduction/
    """
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = labels[np.argmax(counts)]
    dominant_color = kmeans.cluster_centers_[dominant_cluster]
    return [int(c) for c in dominant_color]

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # open webcam

    ret, frame = cap.read()
    # read in a frame

    if not ret:
        break

    # Crop the frame to the bounding box (ROI)
    roi = frame[y1:y2, x1:x2]
    roi = cv2.convertScaleAbs(roi, alpha=1.2, beta=30)  # Normalize brightness

    # Find the dominant color in the ROI
    dominant_color = get_dominant_color(roi)

    # identify the color from the dominant color above
    color_label = identify_color(dominant_color)

    # Debugging: Print RGB and HSV values
    print(f"Dominant RGB: {dominant_color}, Color: {color_label}")

    # Draw box with the ROI and the detected color label
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Color: {color_label}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the webcam feed with the bounding box
    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # if q pressed, close the window and exit.
        break

cap.release()
cv2.destroyAllWindows()
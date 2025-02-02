import cv2
import numpy as np
import os

# Function to sketch object regions
def sketch_object(frame):
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Invert edges to make it look like a sketch
    inverted_edges = cv2.bitwise_not(edges)

    return inverted_edges

# Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load an image for object detection
image_file = input("Enter image url: ")
if not os.path.exists(image_file):
    print(f"Error: File '{image_file}' not found.")
    exit()

frame = cv2.imread(image_file)
height, width, channels = frame.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []

# Extract information from YOLO output
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
    dimensions = f"Height: {h} px, Width: {w} px"

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, dimensions, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    object_region = frame[y:y + h, x:x + w]
    if object_region.size > 0:
        sketch = sketch_object(object_region)
        frame[y:y + h, x:x + w] = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

output_folder = "outputImage"
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "output_image.jpg")
cv2.imwrite(output_file, frame)
print(f"Result saved as '{output_file}'.")

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

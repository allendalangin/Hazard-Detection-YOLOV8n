from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Path to your best.pt model file
model_path = 'runs/detect/yolov8n_TUNED_model_final/weights/best.pt'
# Path to the image you want to test
image_path = './Sign4.jpeg'
# --- END OF CONFIGURATION ---

# Load your trained model
model = YOLO(model_path)

# Run inference on the image
results = model(image_path)

# The 'results' object contains the detections. We can plot them.
# The plot() method returns a BGR numpy array of the image with detections.
annotated_image_bgr = results[0].plot()

# Convert the image from BGR (OpenCV's default) to RGB for displaying with Matplotlib
annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

# Display the image with the detections
plt.figure(figsize=(10, 10))
plt.imshow(annotated_image_rgb)
plt.title('YOLOv8 Detections')
plt.axis('off') # Hide axes
plt.show()
from ultralytics import YOLO

# Load the base model
model = YOLO('yolov8n.pt')

# Start the tuning process
# This will take a long time to run!
model.tune(data='./cleaned_hazard_dataset/data.yaml', epochs=30, iterations=100)
# Import libraries at the top
from ultralytics import YOLO
import torch

# Define your main function that contains the training logic
def main():
    # GPU Check
    if torch.cuda.is_available():
        print("✅ Success! CUDA is available. Training will run on the GPU.")
        device = "cuda:0"
    else:
        print("❌ Warning: CUDA not available. Training will run on the CPU.")
        device = "cpu"

    model = YOLO('yolov8n.pt')

    # Train the Model
    results = model.train(
       data='./cleaned_hazard_dataset/data.yaml',
       epochs=100,
       imgsz=640,
       batch=8,      # Kept this at 4 for your 4GB GPU
       name='yolov8n_hazard_model_final',
       device=0,
       workers=4     # Recommended to keep at 0 for best Windows compatibility
    )
    
    print("\n🎉 Training complete!")
    print("Your trained model and results are saved in the 'runs/detect/' directory.")


# --- Main execution block ---
# This is the crucial part. The code inside this block will only run
# when you execute "python training.py".
if __name__ == '__main__':
    main()
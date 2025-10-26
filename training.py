from ultralytics import YOLO
import torch

def main():
    # GPU Check
    if torch.cuda.is_available():
        print("✅ Success! CUDA is available. Training will run on the GPU.")
    else:
        print("❌ Warning: CUDA not available. Training will run on the CPU.")

    model = YOLO('yolov8n.pt')

    # --- Train the Model with OPTIMIZED Hyperparameters ---
    # These are the values you got from the tuning process.
    results = model.train(
       data='./cleaned_hazard_dataset/data.yaml',
       epochs=100,
       imgsz=640,
       batch=8,
       workers=4,
       device=0,
       name='yolov8n_TUNED_model_final',  # A new name for the new run

       # --- HYPERPARAMETERS FROM YOUR TUNING RESULTS ---
       lr0=0.01,
       lrf=0.01,
       momentum=0.937,
       weight_decay=0.0005,
       warmup_epochs=3.0,
       warmup_momentum=0.8,
       box=7.5,
       cls=0.5,
       dfl=1.5,
       hsv_h=0.015,
       hsv_s=0.7,
       hsv_v=0.4,
       degrees=0.0,
       translate=0.1,
       scale=0.5,
       shear=0.0,
       perspective=0.0,
       flipud=0.0,
       fliplr=0.5,
       mosaic=1.0,
       mixup=0.0,
       copy_paste=0.0,
       close_mosaic=10
    )
    
    print("\n🎉 Tuned training complete!")
    print("Your new, optimized model is saved in a new 'runs/detect/' directory.")


if __name__ == '__main__':
    main()
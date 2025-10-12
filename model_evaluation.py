from ultralytics import YOLO

def main():
    # Load your best trained model from the training run
    model = YOLO('runs/detect/yolov8n_hazard_model_final/weights/best.pt')

    # Run evaluation on the 'test' split of your data
    # Ensure your data.yaml file has a 'test:' path pointing to your test images
    metrics = model.val(split='test')

    # --- Print the key metrics ---
    print("-" * 30)
    print("Final Model Performance Metrics:")

    # mAP scores (primary accuracy indicators)
    print(f"  - mAP50-95 (Box): {metrics.box.map:.4f}")   # More strict
    print(f"  - mAP50 (Box):    {metrics.box.map50:.4f}") # Standard metric, as seen in your screenshot

    # Precision and Recall
    print(f"  - Precision (Box):  {metrics.box.mp:.4f}")  # 'P' in your screenshot
    print(f"  - Recall (Box):     {metrics.box.mr:.4f}")   # 'R' in your screenshot
    print("-" * 30)

if __name__ == '__main__':
    main()
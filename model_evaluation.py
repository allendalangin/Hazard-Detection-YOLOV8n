import numpy as np
from ultralytics import YOLO

def calculate_metrics():
    """
    Loads a trained YOLOv8 model, evaluates it on the test set,
    and calculates precision, recall, and overall accuracy.
    """
    # --- CONFIGURATION ---
    # Path to your best trained model file
    model_path = 'runs/detect/yolov8n_TUNED_model_final/weights/best.pt'
    # --- END OF CONFIGURATION ---

    print("🚀 Loading model and running evaluation on the test set...")

    # Load your trained model
    model = YOLO(model_path)

    # Evaluate the model on the 'test' split of your data
    # This generates all the metrics we need
    metrics = model.val(split='test')

    # --- Standard Metrics (Precision and Recall) ---
    # These are directly available from the metrics object
    precision = metrics.box.mp  # Mean Precision
    recall = metrics.box.mr     # Mean Recall

    print("\n" + "="*40)
    print("Standard Performance Metrics:")
    print(f"  - Precision: {precision:.4f} ({precision:.2%})")
    print("    (When the model detects a hazard, it's correct this often)")
    print(f"  - Recall:    {recall:.4f} ({recall:.2%})")
    print("    (Of all actual hazards, the model successfully finds this many)")
    print("="*40)


    # --- Overall Accuracy from Confusion Matrix ---
    # Get the confusion matrix data
    confusion_matrix = metrics.confusion_matrix.matrix

    # Calculate Overall Accuracy = (Sum of Correct Predictions) / (Total Predictions)
    # The diagonal of the matrix represents correct predictions (True Positives)
    true_positives = np.trace(confusion_matrix)
    total_instances = np.sum(confusion_matrix)

    if total_instances > 0:
        overall_accuracy = true_positives / total_instances
        print("\nOverall Accuracy (from Confusion Matrix):")
        print(f"  - Accuracy: {overall_accuracy:.4f} ({overall_accuracy:.2%})")
        print("    (Calculated as: Sum of diagonal / Sum of all cells)")
        print("="*40)
    else:
        print("Could not calculate overall accuracy. Ensure the confusion matrix was generated.")


# This block ensures the code only runs when the script is executed directly
if __name__ == '__main__':
    calculate_metrics()
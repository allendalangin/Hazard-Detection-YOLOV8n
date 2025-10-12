import pandas as pd
import numpy as np
from ultralytics import YOLO

def evaluate_model():
    """
    Loads the trained model and CSV results to calculate and print
    final performance metrics for the thesis.
    """
    # --- CONFIGURATION ---
    # Path to the results file from your training run
    results_path = 'runs/detect/yolov8n_hazard_model_final/results.csv'
    model_path = 'runs/detect/yolov8n_hazard_model_final/weights/best.pt'
    # --- END OF CONFIGURATION ---


    # --- Part 1: Get Final Loss and Performance Metrics from CSV ---
    try:
        # Load the results and clean up column names
        results = pd.read_csv(results_path)
        results.columns = results.columns.str.strip()

        # Get the data from the very last epoch (iloc[-1])
        last_epoch_data = results.iloc[-1]

        print("-" * 30)
        print("Metrics from Final Epoch (100/100):")
        
        # Validation Losses
        print(f"  - Validation Box Loss:  {last_epoch_data['val/box_loss']:.4f}")
        print(f"  - Validation Cls Loss:  {last_epoch_data['val/cls_loss']:.4f}")
        print(f"  - Validation DFL Loss:  {last_epoch_data['val/dfl_loss']:.4f}")
        
        # Performance Metrics
        print(f"  - Precision (P):        {last_epoch_data['metrics/precision(B)']:.4f}")
        print(f"  - Recall (R):           {last_epoch_data['metrics/recall(B)']:.4f}")
        print(f"  - mAP50 (mAP@.5):       {last_epoch_data['metrics/mAP50(B)']:.4f}")
        print(f"  - mAP50-95 (mAP@.5:.95):{last_epoch_data['metrics/mAP50-95(B)']:.4f}")
        print("-" * 30)

    except FileNotFoundError:
        print(f"Error: Could not find the results file at {results_path}")
    except KeyError as e:
        print(f"Error: A required column is missing from the CSV file: {e}")


    # --- Part 2: Calculate Overall Accuracy from the Confusion Matrix ---
    print("Running evaluation on the test set to calculate overall accuracy...")
    
    # Load your best trained model
    model = YOLO(model_path)

    # Run validation on the test set to get the metrics object
    metrics = model.val(split='test') 
    cm = metrics.confusion_matrix.matrix

    # Overall Accuracy = (Sum of True Positives) / (Total Number of Instances)
    true_positives = np.trace(cm)
    total_instances = np.sum(cm)

    if total_instances > 0:
        overall_accuracy = true_positives / total_instances
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy:.2%})")
        print("This is calculated as (sum of diagonal) / (sum of all cells) in the confusion matrix.")
        print("-" * 30)
    else:
        print("Could not calculate overall accuracy. Ensure the confusion matrix was generated.")

# This block ensures the code only runs when the script is executed directly
if __name__ == '__main__':
    evaluate_model()
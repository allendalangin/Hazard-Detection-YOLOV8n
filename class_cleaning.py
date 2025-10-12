import os

def process_single_dataset(dataset_path, target_class_id):
    # This function is the same as before, but now handles missing folders
    print(f"🚀 Processing dataset at: {dataset_path}")
    print(f"Assigning all annotations to new Class ID: {target_class_id}")
    
    for split in ['train', 'valid', 'test']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        
        if not os.path.exists(labels_dir):
            print(f"  - No '{split}' labels folder found. Skipping.")
            continue
            
        file_count = 0
        for filename in os.listdir(labels_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(labels_dir, filename)
                updated_lines = []
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts: continue
                        parts[0] = str(target_class_id)
                        updated_lines.append(" ".join(parts) + "\n")
                
                with open(file_path, 'w') as f:
                    f.writelines(updated_lines)
                file_count += 1
                
        print(f"  - Processed {file_count} files in the '{split}' set.")
    print("-" * 30)

# --- HOW TO USE ---
# IMPORTANT: We are now pointing to the BALANCED folders!
# Also, I have unified your folder names with the final class names from your thesis.
final_classes = {
    0: 'pothole',
    1: 'puddle',
    2: 'manhole',
    3: 'crack',
    4: 'post',
    5: 'road_sign' 
    # NOTE: Your thesis mentions 'curb' and 'stair' but your datasets are different.
    # I am using the classes from your datasets. Adjust as needed.
}

process_single_dataset('./balanced_datasets/pothole_balanced/', target_class_id=0)
process_single_dataset('./balanced_datasets/puddle_balanced/', target_class_id=1)
process_single_dataset('./balanced_datasets/manhole_balanced/', target_class_id=2)
process_single_dataset('./balanced_datasets/crack_balanced/', target_class_id=3)
process_single_dataset('./balanced_datasets/post_balanced/', target_class_id=4)
process_single_dataset('./balanced_datasets/road_sign_balanced/', target_class_id=5)

print("✅ All individual datasets have been processed!")
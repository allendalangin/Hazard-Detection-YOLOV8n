import os
import shutil

# --- CONFIGURATION ---
# Path to your clean, resplit dataset folders
source_base_path = './resplit_datasets/'

# List of your dataset folder names
dataset_names = [
    'pothole_balanced',
    'puddle_balanced',
    'manhole_balanced',
    'crack_balanced',
    'post_balanced',
    'road_sign_balanced'
]

# Path to your final, merged dataset directory
destination_folder = './cleaned_hazard_dataset/'

# --- SCRIPT ---
print("🚀 Starting the final merge process...")

def merge_files(source_path, dest_path):
    """Copies all files from a source directory to a destination directory."""
    for filename in os.listdir(source_path):
        shutil.copy(os.path.join(source_path, filename), dest_path)

# Iterate through each source dataset and copy its contents
for dataset_name in dataset_names:
    source_dataset_path = os.path.join(source_base_path, dataset_name)
    print(f"Merging files from: {source_dataset_path}")
    
    for split in ['train', 'valid', 'test']:
        # Copy images
        img_src = os.path.join(source_dataset_path, split, 'images')
        if os.path.exists(img_src):
            merge_files(img_src, os.path.join(destination_folder, split, 'images'))
        
        # Copy labels
        lbl_src = os.path.join(source_dataset_path, split, 'labels')
        if os.path.exists(lbl_src):
            merge_files(lbl_src, os.path.join(destination_folder, split, 'labels'))

print("\n✅ Merge complete! Your final dataset is ready in 'cleaned_hazard_dataset/train/imag'.")
import os
import shutil
import random

# --- CONFIGURATION ---
# Your original, unedited dataset folders
source_folders = {
    'crack': './crack/',
    'post': './lamp-post/',
    'manhole': './manhole/',
    'pothole': './pothole/',
    'puddle': './puddle/',
    'road_sign': './road-sign/'
}

# Where to save the new, balanced datasets
output_base_path = './balanced_datasets/'
# The maximum number of images to keep for any class
IMAGE_CAP = 1000

# --- SCRIPT ---
print("🚀 Starting dataset balancing process...")
os.makedirs(output_base_path, exist_ok=True)

for class_name, source_path in source_folders.items():
    print(f"\nProcessing '{class_name}'...")
    
    # Create a new destination folder for the balanced dataset
    dest_path = os.path.join(output_base_path, f"{class_name}_balanced")
    
    total_images_in_class = []
    # First, collect all image paths from train, valid, and test splits
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(source_path, split, 'images')
        if os.path.exists(img_dir):
            for img_name in os.listdir(img_dir):
                # Store the split and the image name
                total_images_in_class.append((split, img_name))

    # Randomly sample up to the IMAGE_CAP
    num_to_sample = min(len(total_images_in_class), IMAGE_CAP)
    sampled_images = random.sample(total_images_in_class, num_to_sample)
    print(f"  - Found {len(total_images_in_class)} images. Capping to {len(sampled_images)}.")

    # Copy the sampled image and label pairs to the new directory
    for split, img_name in sampled_images:
        # Create destination directories
        dest_img_dir = os.path.join(dest_path, split, 'images')
        dest_lbl_dir = os.path.join(dest_path, split, 'labels')
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lbl_dir, exist_ok=True)

        # Source paths
        src_img_path = os.path.join(source_path, split, 'images', img_name)
        
        # Labels have the same name but with a .txt extension
        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_lbl_path = os.path.join(source_path, split, 'labels', label_name)
        
        # Copy files
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dest_img_dir)
        if os.path.exists(src_lbl_path):
            shutil.copy(src_lbl_path, dest_lbl_dir)

print("\n✅ Balancing complete! New datasets are in the 'balanced_datasets' folder.")
import os
import shutil
import random

# --- CONFIGURATION ---
# Path to the folders created by the balancing script
input_base_path = './balanced_datasets/'

# Where to save the newly split datasets
output_base_path = './resplit_datasets/'

# The desired split ratios
split_ratios = {'train': 0.6, 'valid': 0.2, 'test': 0.2}

# List of your balanced folder names
dataset_names = [
    'pothole_balanced',
    'puddle_balanced',
    'manhole_balanced',
    'crack_balanced',
    'post_balanced',
    'road_sign_balanced'
]

# --- SCRIPT ---
print("🚀 Starting dataset resplitting process...")
if os.path.exists(output_base_path):
    shutil.rmtree(output_base_path) # Start with a clean directory
os.makedirs(output_base_path, exist_ok=True)

for dataset_name in dataset_names:
    source_path = os.path.join(input_base_path, dataset_name)
    if not os.path.exists(source_path):
        print(f"Warning: Source path not found, skipping: {source_path}")
        continue

    print(f"\nProcessing '{dataset_name}'...")

    # 1. Collect all image and label file pairs from all existing splits
    all_files = []
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(source_path, split, 'images')
        lbl_dir = os.path.join(source_path, split, 'labels')
        if os.path.exists(img_dir):
            for img_name in os.listdir(img_dir):
                label_name = os.path.splitext(img_name)[0] + '.txt'
                src_img_path = os.path.join(img_dir, img_name)
                src_lbl_path = os.path.join(lbl_dir, label_name)
                if os.path.exists(src_lbl_path):
                    all_files.append((src_img_path, src_lbl_path))

    # 2. Shuffle the collected files randomly
    random.shuffle(all_files)
    total_files = len(all_files)
    print(f"  - Found and shuffled {total_files} image/label pairs.")

    # 3. Calculate split points
    train_end = int(total_files * split_ratios['train'])
    valid_end = train_end + int(total_files * split_ratios['valid'])

    # 4. Create new destination directories
    for split in split_ratios.keys():
        os.makedirs(os.path.join(output_base_path, dataset_name, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_base_path, dataset_name, split, 'labels'), exist_ok=True)

    # 5. Distribute files into the new directories
    splits_data = {
        'train': all_files[:train_end],
        'valid': all_files[train_end:valid_end],
        'test': all_files[valid_end:]
    }

    for split_name, files in splits_data.items():
        dest_dir = os.path.join(output_base_path, dataset_name, split_name)
        for img_path, lbl_path in files:
            shutil.copy(img_path, os.path.join(dest_dir, 'images'))
            shutil.copy(lbl_path, os.path.join(dest_dir, 'labels'))
        print(f"  - Copied {len(files)} files to '{split_name}'.")

print("\n✅ Resplitting complete! New datasets are in the 'resplit_datasets' folder.")
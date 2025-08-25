import os
import shutil
import random

# Source and target root directories
SRC_DIR = r'E:\MRI_LOWMEM\Training'
DST_DIR = r'E:\MRI_LOWMEM\tiny_mri'
IMAGES_PER_CLASS = 500

# Create target root directory if it doesn't exist
os.makedirs(DST_DIR, exist_ok=True)

# For each class/subfolder in source
for class_name in os.listdir(SRC_DIR):
    src_class_dir = os.path.join(SRC_DIR, class_name)
    dst_class_dir = os.path.join(DST_DIR, class_name)

    # Only proceed if it is a directory
    if not os.path.isdir(src_class_dir):
        continue

    # Get all files (images) in the subfolder
    images = [f for f in os.listdir(src_class_dir) if os.path.isfile(os.path.join(src_class_dir, f))]
    random.shuffle(images)
    selected = images[:IMAGES_PER_CLASS]

    # Create destination subfolder
    os.makedirs(dst_class_dir, exist_ok=True)

    # Copy selected files
    for fname in selected:
        src_path = os.path.join(src_class_dir, fname)
        dst_path = os.path.join(dst_class_dir, fname)
        shutil.copy2(src_path, dst_path)

    print(f"Copied {len(selected)} images to {dst_class_dir}")

print("Done! All selected images have been copied.")

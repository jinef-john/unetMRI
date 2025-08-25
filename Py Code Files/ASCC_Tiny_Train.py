import os
import shutil
import random

SRC_DIR = r'E:\ASCC_LOWMEM\Printing'
DST_DIR = r'E:\ASCC_LOWMEM\tiny_train'
IMAGES_PER_CLASS = 500

os.makedirs(DST_DIR, exist_ok=True)

for class_name in os.listdir(SRC_DIR):
    src_class_dir = os.path.join(SRC_DIR, class_name)
    dst_class_dir = os.path.join(DST_DIR, class_name)

    # Only process if it's a directory (subfolder)
    if not os.path.isdir(src_class_dir):
        continue

    images = [f for f in os.listdir(src_class_dir) if os.path.isfile(os.path.join(src_class_dir, f))]
    random.shuffle(images)
    selected = images[:IMAGES_PER_CLASS]

    os.makedirs(dst_class_dir, exist_ok=True)

    for fname in selected:
        src_path = os.path.join(src_class_dir, fname)
        dst_path = os.path.join(dst_class_dir, fname)
        shutil.copy2(src_path, dst_path)

    print(f"Copied {len(selected)} images to {dst_class_dir}")

print("Done! All selected images have been copied.")

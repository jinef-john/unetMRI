import os
import shutil
import random

# Paths
DATASET_DIR = r"E:\ASCC_LOWMEM\Printing"
VAL_OUT_DIR = r"E:\ASCC_LOWMEM\Printing_val"
TRAIN_OUT_DIR = DATASET_DIR  # train stays in original location

# Split ratio
VAL_RATIO = 0.25  # 25% validation

# Classes
CLASSES = [
    "Extrude", "Idle",
    "Inner_wall_Anomalous", "Inner_wall_Normal",
    "Outer_wall_Anomalous", "Outer_wall_Normal",
    "Print"
]

# Ensure val subfolders exist
for cls in CLASSES:
    val_dir = os.path.join(VAL_OUT_DIR, cls)
    os.makedirs(val_dir, exist_ok=True)

# For each class, split files and move val images
for cls in CLASSES:
    class_dir = os.path.join(DATASET_DIR, cls)
    val_dir = os.path.join(VAL_OUT_DIR, cls)
    files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    random.shuffle(files)
    n_val = int(len(files) * VAL_RATIO)
    val_files = files[:n_val]

    print(f"{cls}: {len(files)} images, {n_val} for validation")

    # Move val files to validation folder and delete from train
    for fname in val_files:
        src = os.path.join(class_dir, fname)
        dst = os.path.join(val_dir, fname)
        shutil.move(src, dst)

print("\nDataset split completed!")

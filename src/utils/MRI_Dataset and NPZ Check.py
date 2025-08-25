import os
from PIL import Image
import numpy as np

IMG_ROOT = r'E:\MRI_LOWMEM\tiny_mri'
NPZ_ROOT = r'E:\MRI_LOWMEM\MRI-NPZ_latent_skip64_skip128'
EXTS = ('.png', '.jpg', '.jpeg')

# Собираем все изображения по классам
summary = []
errors = []

for cls in os.listdir(IMG_ROOT):
    cls_dir = os.path.join(IMG_ROOT, cls)
    if not os.path.isdir(cls_dir):
        continue
    for fname in os.listdir(cls_dir):
        if not fname.lower().endswith(EXTS):
            continue
        img_path = os.path.join(cls_dir, fname)
        try:
            with Image.open(img_path) as img:
                mode = img.mode
                size = img.size
                arr = np.array(img)
                if arr.ndim == 2:  # grayscale
                    channels = 1
                elif arr.ndim == 3:
                    channels = arr.shape[2]
                else:
                    channels = '?'
        except Exception as e:
            errors.append(f"[IMG READ ERROR] {img_path}: {e}")
            continue

        # Проверяем NPZ
        npz_name = fname.replace('.png', '.npz').replace('.jpg', '.npz').replace('.jpeg', '.npz')
        npz_path = os.path.join(NPZ_ROOT, cls, npz_name)
        npz_info = {}
        if os.path.exists(npz_path):
            try:
                with np.load(npz_path) as data:
                    for k in data.files:
                        npz_info[k] = data[k].shape
            except Exception as e:
                errors.append(f"[NPZ READ ERROR] {npz_path}: {e}")
                npz_info = "NPZ ERROR"
        else:
            npz_info = "NOT FOUND"

        summary.append({
            'class': cls,
            'image': fname,
            'mode': mode,
            'shape': arr.shape,
            'channels': channels,
            'size': size,
            'npz_info': npz_info,
            'npz_path': npz_path,
        })

# Аналитика по каналам и размерам
from collections import Counter
img_shapes = Counter(tuple(x['shape']) for x in summary)
img_channels = Counter(x['channels'] for x in summary)
npz_latent_shapes = Counter(tuple(x['npz_info']['latent']) for x in summary if isinstance(x['npz_info'], dict) and 'latent' in x['npz_info'])
npz_skip64_shapes = Counter(tuple(x['npz_info']['skip64']) for x in summary if isinstance(x['npz_info'], dict) and 'skip64' in x['npz_info'])

print("=== IMAGE SHAPE STATS ===")
for shape, count in img_shapes.items():
    print(f"Image shape {shape}: {count}")

print("\n=== CHANNEL STATS ===")
for ch, count in img_channels.items():
    print(f"Channels {ch}: {count}")

print("\n=== NPZ latent SHAPE STATS ===")
for shape, count in npz_latent_shapes.items():
    print(f"latent shape {shape}: {count}")

print("\n=== NPZ skip64 SHAPE STATS ===")
for shape, count in npz_skip64_shapes.items():
    print(f"skip64 shape {shape}: {count}")

print("\n=== ERRORS ===")
for e in errors:
    print(e)

# Если хочешь — можешь выгрузить всё в .csv для Excel анализа
import csv
csv_path = r'E:\MRI_LOWMEM\shape_stats_report.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['class','image','mode','shape','channels','size','npz_info','npz_path'])
    writer.writeheader()
    for row in summary:
        writer.writerow(row)

print(f"\nFull summary written to: {csv_path}")

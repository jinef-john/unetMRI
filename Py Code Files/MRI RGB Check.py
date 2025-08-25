import os
from PIL import Image

DATA_ROOT = r"E:\MRI_LOWMEM\tiny_mri_512"
TARGET_SHAPE = (512, 512)

def is_grayscale(img):
    if img.mode == 'L':
        return True
    if img.mode == 'RGB':
        arr = img.convert('RGB')
        for pixel in arr.getdata():
            r, g, b = pixel
            if r != g or r != b:
                return False
        return True
    return False

grayscale_count = 0
rgb_count = 0
other_count = 0
files_checked = 0
bad_files = []
bad_shape_files = []

for root, dirs, files in os.walk(DATA_ROOT):
    for fname in files:
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            path = os.path.join(root, fname)
            try:
                img = Image.open(path)
                files_checked += 1
                # Проверка размера
                if img.size != TARGET_SHAPE:
                    bad_shape_files.append((path, img.size))
                # Проверка каналов
                if img.mode == 'L':
                    grayscale_count += 1
                elif img.mode == 'RGB':
                    if is_grayscale(img):
                        grayscale_count += 1
                    else:
                        rgb_count += 1
                else:
                    other_count += 1
            except Exception as e:
                bad_files.append((path, str(e)))
                continue

print(f"Всего файлов: {files_checked}")
print(f"Grayscale (1-канальные): {grayscale_count}")
print(f"RGB (реальные, разноканальные): {rgb_count}")
print(f"Other modes: {other_count}")

if bad_shape_files:
    print(f"\n[Файлы НЕ 512x512]: {len(bad_shape_files)}")
    for path, shape in bad_shape_files:
        print(f"{path}: {shape}")

if bad_files:
    print("\n[Ошибка чтения файлов]:")
    for path, err in bad_files:
        print(f"{path}: {err}")

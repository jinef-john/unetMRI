import os
from PIL import Image, ImageOps

DATA_ROOT = r"E:\MRI_LOWMEM\Training_512"
OUT_ROOT = DATA_ROOT + "_512"
TARGET_SIZE = 512

total_files = 0
modified_files = 0
bad_files = []

for root, dirs, files in os.walk(DATA_ROOT):
    for fname in files:
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            src_path = os.path.join(root, fname)
            # Построим выходной путь (та же структура, но OUT_ROOT)
            rel_dir = os.path.relpath(root, DATA_ROOT)
            out_dir = os.path.join(OUT_ROOT, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname)
            try:
                img = Image.open(src_path)
                orig_mode = img.mode
                orig_size = img.size
                changed = False

                # Конвертация в grayscale, если надо
                if img.mode != 'L':
                    img = img.convert('L')
                    changed = True

                # Central crop если больше 512x512
                if img.size[0] > TARGET_SIZE or img.size[1] > TARGET_SIZE:
                    left = (img.size[0] - TARGET_SIZE) // 2
                    top = (img.size[1] - TARGET_SIZE) // 2
                    right = left + TARGET_SIZE
                    bottom = top + TARGET_SIZE
                    img = img.crop((left, top, right, bottom))
                    changed = True

                # Padding если меньше 512x512
                if img.size[0] < TARGET_SIZE or img.size[1] < TARGET_SIZE:
                    delta_w = TARGET_SIZE - img.size[0]
                    delta_h = TARGET_SIZE - img.size[1]
                    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
                    img = ImageOps.expand(img, padding, fill=0)
                    changed = True

                img.save(out_path)
                if changed:
                    print(f"Сохранено (исправлено): {out_path} | mode: {orig_mode} → L, size: {orig_size} → {img.size}")
                total_files += 1
                modified_files += 1
            except Exception as e:
                bad_files.append((src_path, str(e)))

print(f"\nВсего обработано: {total_files}")
print(f"Сохранено файлов в {OUT_ROOT}: {modified_files}")

if bad_files:
    print("\n[Ошибки при обработке файлов]:")
    for path, err in bad_files:
        print(f"{path}: {err}")

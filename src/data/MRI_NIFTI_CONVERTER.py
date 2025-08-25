import os
import numpy as np
from PIL import Image
import nibabel as nib
from tqdm import tqdm

# Пути — меняй под себя
DATA_ROOT = r"E:\MRI_LOWMEM\Training"            # Исходные PNG/JPG
NIFTI_ROOT = r"E:\MRI_LOWMEM\train_nifti"     # Куда сохранять NIfTI

# Создать директорию, если нет
os.makedirs(NIFTI_ROOT, exist_ok=True)

def convert_img_to_nifti(img_path, save_path):
    # Загрузка изображения
    img = Image.open(img_path).convert('L')   # grayscale
    img_np = np.array(img, dtype=np.float32)

    # Добавляем ось глубины, так как NIfTI — 3D формат
    img_np = img_np[np.newaxis, :, :]   # shape (1, H, W)

    # Создаем NIfTI объект (без аффинной матрицы — Identity)
    nifti_img = nib.Nifti1Image(img_np, affine=np.eye(4))

    # Сохраняем
    nib.save(nifti_img, save_path)

for root, dirs, files in os.walk(DATA_ROOT):
    # Относительный путь от DATA_ROOT
    rel_dir = os.path.relpath(root, DATA_ROOT)
    # Создаем такой же подкаталог в NIFTI_ROOT
    target_dir = os.path.join(NIFTI_ROOT, rel_dir)
    os.makedirs(target_dir, exist_ok=True)

    for f in tqdm(files, desc=f"Processing {rel_dir}"):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        src_path = os.path.join(root, f)
        # Новое имя с расширением .nii.gz
        base_name = os.path.splitext(f)[0]
        save_path = os.path.join(target_dir, base_name + '.nii.gz')

        # Пропускаем, если файл уже существует
        if os.path.exists(save_path):
            continue

        convert_img_to_nifti(src_path, save_path)
        print(f"Converted {src_path} → {save_path}")

print("All images converted to NIfTI.")

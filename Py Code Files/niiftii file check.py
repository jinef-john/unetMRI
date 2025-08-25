import os
import nibabel as nib
import numpy as np


def check_nii_file(path):
    try:
        nii = nib.load(path)
        data = nii.get_fdata()
        print(f"\nФайл: {path}")
        print(f"Форма данных: {data.shape}")
        print(f"Мин/Макс значение: {np.min(data)}, {np.max(data)}")
        print(f"Среднее значение: {np.mean(data)}")
        print(f"Количество ненулевых элементов: {np.count_nonzero(data)}")
        print(f"Аффинная матрица:\n{nii.affine}")

        if np.count_nonzero(data) == 0:
            print("Внимание: Файл содержит только нули!")
    except Exception as e:
        print(f"Ошибка при загрузке {path}: {e}")


def batch_check_nii_files(root_dir):
    nii_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, f))
    print(f"Найдено {len(nii_files)} NIfTI файлов для проверки.")

    for fpath in nii_files:
        check_nii_file(fpath)


if __name__ == "__main__":
    DATA_DIR = r"E:\MRI_LOWMEM\train_nifti"  # путь к твоей папке с NIfTI
    batch_check_nii_files(DATA_DIR)

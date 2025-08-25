import os
import subprocess
from tqdm import tqdm
import nibabel as nib
import numpy as np
from PIL import Image

# Пути — подставь свои
# Get project root directory (three levels up from current file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "nifti_files")  # Update path as needed
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "hd_bet_masks")
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "output", "hd_bet_masks_jpeg")  # Папка для jpeg масок
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

HD_BET_CMD = "hd-bet"
DEVICE = "cuda"

def nii_to_jpeg(nii_path, jpeg_dir):
    try:
        nii_img = nib.load(nii_path)
        mask_3d = nii_img.get_fdata()
        print(f"[nii_to_jpeg] Загружена маска {nii_path} с формой {mask_3d.shape} и мин/макс: {mask_3d.min()}/{mask_3d.max()}")
        mask_3d = (mask_3d > 0).astype(np.uint8)

        base_name = os.path.splitext(os.path.basename(nii_path))[0]
        if nii_path.endswith('.nii.gz'):
            base_name = base_name[:-4]

        os.makedirs(jpeg_dir, exist_ok=True)

        shape = mask_3d.shape
        axes = [0, 1, 2]
        spatial_axes = [ax for ax in axes if shape[ax] == 512]

        if len(spatial_axes) != 2:
            print(f"[nii_to_jpeg] Warning: не удалось найти 2 spatial axes размером 512, shape={shape}")
            slice_axis = 2
        else:
            slice_axis = list(set(axes) - set(spatial_axes))[0]

        num_slices = shape[slice_axis]

        for i in range(num_slices):
            if slice_axis == 0:
                slice_2d = mask_3d[i, :, :]
            elif slice_axis == 1:
                slice_2d = mask_3d[:, i, :]
            else:
                slice_2d = mask_3d[:, :, i]

            print(f"[nii_to_jpeg] Срез {i} имеет форму {slice_2d.shape}")

            slice_img = Image.fromarray(slice_2d * 255)
            slice_img.save(os.path.join(jpeg_dir, f"{base_name}_slice{i:03d}.jpeg"))
    except Exception as e:
        print(f"[nii_to_jpeg] Ошибка при конвертации {nii_path} в JPEG: {e}")

def run_hd_bet_on_file(input_path, output_dir, device="cuda"):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if input_path.endswith('.nii.gz'):
        base_name = base_name[:-4]

    output_file = os.path.join(output_dir, base_name + '_bet_bet.nii.gz')
    output_file_simple = os.path.join(output_dir, base_name + '_bet_bet.nii.gz')

    cmd = [
        HD_BET_CMD,
        '-i', input_path,
        '-o', output_file_simple,
        '-device', device,
        '--verbose',
        '--disable_tta'
    ]

    print(f"[run_hd_bet_on_file] Запускаю команду: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[run_hd_bet_on_file] Ошибка при обработке {input_path}:\n{result.stderr}")
    else:
        print(f"[run_hd_bet_on_file] Обработан {input_path}, маска должна сохраниться в {output_file}")
        if os.path.exists(output_file):
            jpeg_dir = os.path.join(VISUALIZATION_DIR, os.path.relpath(output_dir, OUTPUT_DIR))
            nii_to_jpeg(output_file, jpeg_dir)
        else:
            print(f"[run_hd_bet_on_file] Файл {output_file} не найден для конвертации")

def batch_inference(input_dir, output_dir, device="cuda"):
    for root, _, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        out_dir = os.path.join(output_dir, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        nii_files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]
        for f in tqdm(nii_files, desc=f"Обработка {rel_path}"):
            input_path = os.path.join(root, f)
            run_hd_bet_on_file(input_path, out_dir, device)

if __name__ == "__main__":
    batch_inference(INPUT_DIR, OUTPUT_DIR, DEVICE)
    print("Обработка HD-BET завершена.")

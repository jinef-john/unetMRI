import os
import numpy as np
import nibabel as nib
from PIL import Image
import cv2
from tqdm import tqdm
from nnunet.inference.predict import predict

# Пути к данным
NIFTI_DIR = r"E:\MRI_LOWMEM\train_nifti"
ORIG_IMG_ROOT = r"E:\MRI_LOWMEM\Training"
MASK_OUTPUT_DIR = r"E:\MRI_LOWMEM\resultant_Masks"
OVERLAY_OUTPUT_DIR = os.path.join(MASK_OUTPUT_DIR, "overlays")

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']


def run_nnunet_inference():
    print("Запускаем nnU-Net inference...")
    predict(
        input_folder=r"E:/MRI_LOWMEM/train_nifti",
        output_folder=r"E:/MRI_LOWMEM/resultant_Masks",
        task='001',
        model='3d_fullres',
        folds=[0],
        save_npz=False,
        num_threads_preprocessing=4,
        num_threads_nifti_save=4,
        overwrite_existing=False,
        mode='normal'
    )
    print("Inference завершён.")

def nii_to_png_slices(nii_path, png_dir):
    nii_img = nib.load(nii_path)
    mask_3d = nii_img.get_fdata()
    mask_3d = (mask_3d > 0).astype(np.uint8)

    os.makedirs(png_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(nii_path))[0]

    for i in range(mask_3d.shape[2]):
        slice_2d = mask_3d[:, :, i] * 255
        Image.fromarray(slice_2d).save(os.path.join(png_dir, f"{base_name}_slice{i:03d}.png"))

def create_overlay(img_path, mask_path, save_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        print(f"Ошибка загрузки {img_path} или {mask_path}")
        return
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img_rgb, 0.7, mask_color, 0.3, 0)
    cv2.imwrite(save_path, overlay)

def generate_overlays():
    for root, _, files in os.walk(MASK_OUTPUT_DIR):
        rel_path = os.path.relpath(root, MASK_OUTPUT_DIR)
        if rel_path == "overlays":  # избегаем рекурсии
            continue
        overlay_dir = os.path.join(OVERLAY_OUTPUT_DIR, rel_path)
        os.makedirs(overlay_dir, exist_ok=True)
        for f in tqdm(files, desc=f"Создаём overlay в {rel_path}"):
            if not f.endswith('.png'):
                continue
            mask_path = os.path.join(root, f)
            parts = rel_path.split(os.sep)
            cls = parts[0] if parts[0] in CLASSES else ''
            base_name = '_'.join(f.split('_')[:-1]) + '.png'
            orig_img_path = os.path.join(ORIG_IMG_ROOT, cls, base_name)
            overlay_path = os.path.join(overlay_dir, f.replace('.png', '_overlay.png'))
            if os.path.exists(orig_img_path):
                create_overlay(orig_img_path, mask_path, overlay_path)

if __name__ == "__main__":
    run_nnunet_inference()
    for root, _, files in os.walk(MASK_OUTPUT_DIR):
        for f in files:
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                nii_to_png_slices(os.path.join(root, f), os.path.join(MASK_OUTPUT_DIR, os.path.relpath(root, MASK_OUTPUT_DIR)))
    generate_overlays()
    print("Всё готово.")

import os
import numpy as np
import nibabel as nib
from PIL import Image
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm


# Пути — подставь свои
MODEL_PATH = r"E:\MRI_LOWMEM\U2Net\u2net.pth"
NIFTI_DIR = r"E:\MRI_LOWMEM\train_nifti"
ORIG_IMG_ROOT = r"E:\MRI_LOWMEM\Training"
MASK_PNG_ROOT = r"E:\MRI_LOWMEM\u2net_masks_png"
OVERLAY_ROOT = r"E:\MRI_LOWMEM\u2net_masks_overlay"

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

from u2net import U2NET  # Импортируй из твоего u2net.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели
model = U2NET(3, 1)  # in_ch=3, out_ch=1, т.к. дублируем grayscale в RGB
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])

def gray_to_rgb_tensor(tensor_gray):
    # tensor_gray shape: [1, 1, H, W]
    return tensor_gray.repeat(1, 3, 1, 1)  # повторяем канал три раза


normalize = transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])

def predict_mask_on_slice(img_pil):
    img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0)  # [1,1,H,W]
    img_rgb = gray_to_rgb_tensor(img_tensor)
    img_rgb = normalize(img_rgb[0]).unsqueeze(0).to(device)  # нормализация + перемещение на device
    with torch.no_grad():
        d1, *_ = model(img_rgb)
        pred = d1[:, 0, :, :]
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        mask_np = (pred.cpu().numpy()[0] > 0.33).astype(np.uint8) * 255  # новый порог 0.33
    return mask_np


def nii_to_slices_and_predict(nii_path, mask_out_dir, overlay_out_dir, processed_counter):
    nii_img = nib.load(nii_path)
    data_3d = nii_img.get_fdata()
    data_3d = np.nan_to_num(data_3d)

    shape = data_3d.shape
    # Находим два максимальных измерения и их индексы
    sorted_dims = sorted([(dim, idx) for idx, dim in enumerate(shape)], reverse=True)
    spatial_dims = [sorted_dims[0][1], sorted_dims[1][1]]
    slice_axis = 3 - sum(spatial_dims)

    num_slices = shape[slice_axis]

    """if processed_counter % 10 == 0:
        print(f"[{processed_counter}] Processing: {nii_path}")
        print(f"Shape: {shape}, slice_axis: {slice_axis}, spatial_dims: {spatial_dims}")
        print(f"Number of slices: {num_slices}")"""

    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(overlay_out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(nii_path))[0]

    for i in range(num_slices):
        if slice_axis == 0:
            slice_2d = data_3d[i, :, :]
        elif slice_axis == 1:
            slice_2d = data_3d[:, i, :]
        else:  # slice_axis == 2
            slice_2d = data_3d[:, :, i]

        #if processed_counter % 10 == 0 and i == 0:
        #    print(f"Slice 0 shape: {slice_2d.shape}")

        norm_slice = (slice_2d - np.min(slice_2d)) / (np.ptp(slice_2d) + 1e-8)
        img_pil = Image.fromarray((norm_slice * 255).astype(np.uint8))

        mask_np = predict_mask_on_slice(img_pil)

        mask_path = os.path.join(mask_out_dir, f"{base_name}_slice{i:03d}.png")
        Image.fromarray(mask_np).save(mask_path)

        class_name = None
        for cls in CLASSES:
            if cls in nii_path:
                class_name = cls
                break

        orig_img_path = None
        if class_name:
            orig_img_path = os.path.join(ORIG_IMG_ROOT, class_name, f"{base_name}.png")

        if orig_img_path and os.path.exists(orig_img_path):
            img_orig = cv2.imread(orig_img_path, cv2.IMREAD_GRAYSCALE)
            if img_orig is not None:
                mask_color = cv2.applyColorMap(mask_np, cv2.COLORMAP_JET)
                img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(img_rgb, 0.7, mask_color, 0.3, 0)
                overlay_path = os.path.join(overlay_out_dir, f"{base_name}_slice{i:03d}_overlay.png")
                cv2.imwrite(overlay_path, overlay)

def main():
    processed_counter = 0
    for root, _, files in os.walk(NIFTI_DIR):
        rel_path = os.path.relpath(root, NIFTI_DIR)
        mask_dir = os.path.join(MASK_PNG_ROOT, rel_path)
        overlay_dir = os.path.join(OVERLAY_ROOT, rel_path)

        for f in tqdm(files, desc=f"Processing {rel_path}"):
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                nii_path = os.path.join(root, f)
                nii_to_slices_and_predict(nii_path, mask_dir, overlay_dir, processed_counter)
                processed_counter += 1

if __name__ == "__main__":
    main()

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from tqdm import tqdm
from multiprocessing import Pool, set_start_method

DATA_ROOT = r'E:\MRI_LOWMEM\Training'
OUT_NPY = r'E:\MRI_LOWMEM\C1_Saliency\NPY'
OUT_PNG = r'E:\MRI_LOWMEM\C1_Saliency\PNG'
C1_PTH = r'E:\MRI_LOWMEM\C1-B3-CBAM\MRI-C1EfficientNet_B3_CBAM.pth'
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

os.makedirs(OUT_NPY, exist_ok=True)
os.makedirs(OUT_PNG, exist_ok=True)
for cls in CLASSES:
    os.makedirs(os.path.join(OUT_NPY, cls), exist_ok=True)
    os.makedirs(os.path.join(OUT_PNG, cls), exist_ok=True)

def process_single(args):
    cls, fname = args
    src_dir = os.path.join(DATA_ROOT, cls)
    img_path = os.path.join(src_dir, fname)
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from MRI_C1_B3_CBAM import EfficientNetB3_CBAM_Bottleneck
    # === Создание модели с правильным input ===
    c1 = EfficientNetB3_CBAM_Bottleneck(num_classes=4).to(device)
    # Обязательная подмена первого conv на grayscale
    c1.base.features[0][0] = nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False)
    state_dict = torch.load(C1_PTH, map_location=device)
    # Убираем module. если есть
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    c1.load_state_dict(state_dict)
    c1 = c1.to(device)
    c1.eval()

    img_pil = Image.open(img_path).convert('L').resize((512, 512), Image.BICUBIC)
    img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        saliency = c1.get_cbam_attention(img_tensor).squeeze().cpu().numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    base = os.path.splitext(fname)[0]
    npy_out = os.path.join(OUT_NPY, cls, f"{base}_sal.npy")
    png_out = os.path.join(OUT_PNG, cls, f"{base}_sal.png")
    np.save(npy_out, saliency)
    sal_map_png = (saliency * 255).clip(0,255).astype(np.uint8)
    Image.fromarray(sal_map_png).save(png_out)
    return fname

if __name__ == "__main__":
    set_start_method('spawn', force=True)
    all_jobs = []
    for cls in CLASSES:
        src_dir = os.path.join(DATA_ROOT, cls)
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        all_jobs += [(cls, fname) for fname in files]

    with Pool(processes=6) as pool:
        for _ in tqdm(pool.imap_unordered(process_single, all_jobs), total=len(all_jobs)):
            pass

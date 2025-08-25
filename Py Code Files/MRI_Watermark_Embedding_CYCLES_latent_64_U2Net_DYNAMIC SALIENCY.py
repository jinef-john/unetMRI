import os, csv, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3
from MRI_C1_B3_CBAM import CBAM, EfficientNetB3_CBAM_Bottleneck
from MRI_Encoder_Latent_64_Train import Encoder, Decoder
from torchvision import transforms
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torchvision.transforms.functional as TF
import io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from datetime import datetime
import datetime
# ============ CONFIG ============
DATA_ROOT = r"E:\MRI_LOWMEM\tiny_mri"
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
saliency_root = r'E:\MRI_LOWMEM\C1_Saliency\NPY'
WM_ROOT  = r"E:\MRI_LOWMEM\watermarked_cycles_latent_skip64_U2NET"
CSV_ROOT = r"E:\MRI_LOWMEM\metrics_logs_latent_skip64"
C1_PATH = r"E:\MRI_LOWMEM\C1-B3-CBAM\MRI-C1EfficientNet_B3_CBAM.pth"
NPZ_ROOT = r"E:\MRI_LOWMEM\MRI-NPZ_latent_skip64"
ENCODER_PTH = r"E:\MRI_LOWMEM\Encoder_latent_64\autoencoder_epoch7.pth"
u2net_mask_root = r"E:\MRI_LOWMEM\u2net_masks_png"

CYCLES = 5
EPOCHS_PER_CYCLE = 5
BATCH_SIZE = 2
epoch_metrics = []
EARLY_STOP_C2_CLEAN = 0.20
EARLY_STOP_C2_WM    = 0.90

PAYLOAD_DIM, SEED_DIM = 256, 32
LATENT_SHAPE, SKIP64_SHAPE, SKIP128_SHAPE = (1024, 32, 32), (512, 64, 64), (256, 128, 128)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

os.makedirs(saliency_root, exist_ok=True)
os.makedirs(WM_ROOT, exist_ok=True)
os.makedirs(CSV_ROOT, exist_ok=True)

encoder = Encoder().to(device)
decoder = Decoder().to(device)

IMG_SIZE = 512


mri_augment = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),                         # 1. Всегда grayscale
    transforms.Resize((IMG_SIZE, IMG_SIZE)),                             # 2. Resize (без масштабирования!)
    transforms.RandomHorizontalFlip(p=0.5),                              # 3. Горизонтальный флип (безопасно)
    transforms.RandomRotation(degrees=3),                                # 4. Мягкое вращение [-3°, +3°]
    transforms.RandomAffine(degrees=0, translate=(0.01, 0.01)),          # 5. Малый сдвиг (1%)
    transforms.ColorJitter(brightness=0.07, contrast=0.07, saturation=0, hue=0), # 6. Мягкий jitter (не более 0.07, saturation=0)
    transforms.ToTensor(),                                               # 7. Перевод в Tensor
    transforms.Normalize([0.5], [0.5]),                                  # 8. В [-1, 1]
    transforms.RandomErasing(p=0.03, scale=(0.01, 0.03), ratio=(0.3, 2.5), value='random'), # 9. Очень редкое и маленькое стирание
])


class MRIImageDataset(Dataset):
    def __init__(self, root, classes, augment=None):
        self.samples = []
        for cls in classes:
            folder = os.path.join(root, cls)
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.png', '.jpg')):
                    self.samples.append((os.path.join(folder, fname), cls))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls = self.samples[idx]
        img_pil = Image.open(img_path)
        img_512 = ensure_grayscale_512(img_pil)      # <-- гарантирует 512x512 grayscale

        # Аугментация применяем только к уже обработанному img_512!
        img_aug = self.augment(img_512) if self.augment else transforms.ToTensor()(img_512)
        img_orig = np.array(img_512, dtype=np.uint8)  # [H, W], uint8 — raw original для коллажа
        label = self.class_to_idx[cls]
        fname = os.path.basename(img_path)
        return img_aug, label, fname, img_orig

def check_nan_inf(tensor, name):
    n_nan = torch.isnan(tensor).sum().item()
    n_inf = torch.isinf(tensor).sum().item()
    if n_nan or n_inf:
        print(f"!!! [ALERT] {name}: nan={n_nan}, inf={n_inf}, min={tensor.min().item()}, max={tensor.max().item()}")
    else:
        print(f"[OK] {name}: min={tensor.min().item():.5f}, max={tensor.max().item():.5f}, mean={tensor.mean().item():.5f}")



def log_batch_to_csv(row, csv_path):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, row.keys())
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        writer.writerow(row)

def ensure_grayscale_512(img):
    # 1. Grayscale
    if img.mode != 'L':
        img = img.convert('L')
    w, h = img.size

    # 2. Если размер уже 512x512 — ничего не делаем
    if w == 512 and h == 512:
        return img

    # 3. Если меньше — pad до центра
    if w < 512 or h < 512:
        pad_w = max(0, (512 - w) // 2)
        pad_h = max(0, (512 - h) // 2)
        pad = (pad_w, pad_h, 512 - w - pad_w, 512 - h - pad_h)
        img = ImageOps.expand(img, pad, fill=0)
        w, h = img.size

    # 4. Если больше — crop по центру
    if w > 512 or h > 512:
        left = (w - 512) // 2
        top = (h - 512) // 2
        img = img.crop((left, top, left + 512, top + 512))
    return img

def load_saliency_mask_batch(fnames, batch_class_names, saliency_root, target_shape, device):
    saliency_maps = []
    for fname, cls in zip(fnames, batch_class_names):
        fbase = os.path.splitext(fname)[0]
        sal_path = os.path.join(saliency_root, cls, fbase + '_sal.npy')
        if not os.path.exists(sal_path):
            print(f"[WARN] Missing saliency: {sal_path} — using zeros")
            sal_map = np.zeros(target_shape, dtype=np.float32)
        else:
            sal_map = np.load(sal_path)
            if sal_map.shape != target_shape:
                import torch.nn.functional as F
                sal_t = torch.from_numpy(sal_map).unsqueeze(0).unsqueeze(0).float()
                sal_t = F.interpolate(sal_t, size=target_shape, mode='bilinear', align_corners=False)
                sal_map = sal_t.squeeze().numpy()
        saliency_maps.append(sal_map)
    saliency_maps = np.stack(saliency_maps)
    saliency_maps = torch.from_numpy(saliency_maps).unsqueeze(1).float().to(device)
    return saliency_maps

# ============ WATERMARK GEN + MASK ============

class WatermarkGen(nn.Module):
    def __init__(self, payload_dim, seed_dim, latent_shape):
        super().__init__()
        self.payload_proj = nn.Linear(payload_dim+seed_dim, np.prod(latent_shape))
        self.fusion = nn.Sequential(
            nn.Conv2d(latent_shape[0]*2, latent_shape[0], 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(latent_shape[0], latent_shape[0], 3, 1, 1),
            nn.Tanh()
        )
        self.latent_shape = latent_shape
    def forward(self, payload, seed, x, _=None):
        B = x.shape[0]
        full_payload = torch.cat([payload, seed], 1)
        payload_feat = self.payload_proj(full_payload).view(B, *self.latent_shape)
        fused = torch.cat([x, payload_feat], 1)
        return self.fusion(fused)

class MaskHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        nn.init.constant_(self.conv.bias, 1.0)  # <-- bias в +1 (можно от 0.5 до 2.0 тестировать)
    def forward(self, x):
        return torch.tanh(self.conv(x))

# ============ EFFICIENTNET + CBAM (GRAYSCALE PATCH) ============
class EfficientNetB3_CBAM_Bottleneck(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.base = efficientnet_b3(weights=None, num_classes=num_classes)
        old_conv = self.base.features[0][0]
        new_conv = nn.Conv2d(1, 40, 3, 2, 1, bias=False)
        if hasattr(old_conv, "weight") and old_conv.weight.shape[1] == 3:
            with torch.no_grad():
                gray_w = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(gray_w)
        self.base.features[0][0] = new_conv
        self.features1 = nn.Sequential(*list(self.base.features.children())[:6])
        self.cbam = CBAM(136)
        self.features2 = nn.Sequential(*list(self.base.features.children())[6:])
        self.avgpool = self.base.avgpool
        self.classifier = self.base.classifier
    def forward(self, x):
        x = self.features1(x)
        x = self.cbam(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    def get_cbam_attention(self, x):
        """
        Вернуть spatial attention карту CBAM после features1 (до classifier).
        x: [B, 1, 512, 512]
        return: [B, 1, H, W] — обычно [B, 1, 32, 32]
        """
        x_feat = self.features1(x)  # [B, C, H, W]
        ch_att = self.cbam.ca(x_feat)  # [B, C, 1, 1]
        x_ca = x_feat * ch_att  # [B, C, H, W]
        spat_att = self.cbam.sa(x_ca)  # [B, 1, H, W]
        return spat_att


def calc_ssim(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    if img1.max() > 1.5: img1 /= 255.0
    if img2.max() > 1.5: img2 /= 255.0
    return ssim(img1, img2, data_range=1.0)

def get_percentile_mask(sal_map: np.ndarray, mask_fraction: float = 0.15) -> np.ndarray:
    flat = sal_map.ravel()
    quantile = mask_fraction * 100
    th = np.percentile(flat, quantile)
    mask = (sal_map < th).astype(np.float32)
    if mask.sum() == 0:
        min_idx = np.argmin(flat)
        mask_flat = np.zeros_like(flat)
        mask_flat[min_idx] = 1
        mask = mask_flat.reshape(sal_map.shape)
    return mask



def replace_ext_with_png(filename):
    base = os.path.splitext(filename)[0]
    if base.endswith('.nii'):
        base = os.path.splitext(base)[0]
    return base + '.png'

def load_u2net_mask_batch(fnames, mask_root, target_shape, device, class_names):
    masks = []
    for fname in fnames:
        if '/' in fname or '\\' in fname:
            class_folder, file_name = os.path.split(fname)
        else:
            class_folder = None
            file_name = fname

        mask_file = replace_ext_with_png(file_name)
        mask_path = None
        if class_folder is not None:
            mask_path = os.path.join(mask_root, class_folder, mask_file)
        else:
            for cls in class_names:
                candidate = os.path.join(mask_root, cls, mask_file)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

        if mask_path and os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert('L')
            mask_tensor = TF.to_tensor(mask_img)
            mask_resized = TF.resize(mask_tensor, target_shape, interpolation=Image.NEAREST)
            masks.append(mask_resized)
            print(f"[U2Net Mask] FOUND: {mask_path} для {fname}")
        else:
            masks.append(torch.zeros((1, *target_shape)))
            print(f"[U2Net Mask] NOT FOUND: {mask_path} для {fname}")

    batch_mask = torch.stack(masks).to(device)
    return batch_mask


def save_watermark_collage(
    raw_orig_np,          # np.array или torch.Tensor (1,H,W) — оригинал
    recon_orig_img,       # torch.Tensor [1,1,H,W]
    wm_img,               # torch.Tensor [1,1,H,W]
    go_mask_latent,       # torch.Tensor [1,1,h,w]
    go_mask_skip64,       # torch.Tensor [1,1,h,w]
    u2net_mask_latent,    # torch.Tensor [1,1,h,w]
    total_no_go,          # torch.Tensor [1,1,h,w]
    saliency_overlay,     # np.array [512,512,3]
    embedding_mask_overlay, # np.array [512,512,3]
    out_path: str
):
    # полный путь «…_collage.png»
    # Вспомогательные функции
    def to_np(img):
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy()
        if img.ndim == 4:
            img = img[0]
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        return img

    def normalize_01(x):
        x = x.astype(np.float32)
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def mask2heat(mask, title, H=512, W=512):
        mask = to_np(mask)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_color = cv2.applyColorMap((normalize_01(mask) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.putText(mask_color, title, (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return mask_color

    def tensor2np(img):
        arr = img.detach().cpu().numpy()
        arr = np.squeeze(arr)                   # [H,W]
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return arr

    def get_seismic_diff_panel_np(
            orig_img, wm_img, size=(512, 512), label="Signed Difference (seismic)", vmax=0.05
    ):
        # --- Convert tensors to numpy ---
        orig = orig_img.squeeze().detach().cpu().numpy()
        wm = wm_img.squeeze().detach().cpu().numpy()
        diff_signed = wm - orig

        # --- Проверка на NaN/Inf и удаление ---
        diff_signed = np.where(np.isfinite(diff_signed), diff_signed, 0.0)

        # Логируем значения только после получения diff_signed!
        print(f"Seismic diff mean={diff_signed.mean():.3f}, min={diff_signed.min():.3f}, max={diff_signed.max():.3f}")

        # --- Строго фиксированный диапазон ---
        # Белый — только при diff=0, синий/красный — даже при малых отличиях!
        # vmax задаёт reference (обычно 0.01)
        vmin = -vmax
        vmax = +vmax

        # --- Создание карты ---
        buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=(2.3, 2.1), dpi=160)
        im = ax.imshow(diff_signed, cmap='seismic', vmin=vmin, vmax=vmax)
        ax.axis('off')
        plt.tight_layout(pad=0)

        cax = inset_axes(ax, width="24%", height="2.5%", loc='lower right', borderpad=0.2)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label('Depth', fontsize=7)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=False)
        plt.close()
        buf.seek(0)
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.resize(img, size)

        # --- Подпись ---
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 50)
        except:
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([0, 0, img_pil.width, text_height + 6], fill=(255, 255, 255, 200))
        draw.text((6, 3), label, fill=(0, 0, 0), font=font)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img

    def put_label(img, text, font_scale=0.6):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (255,255,255), -1)
        cv2.putText(img, text, (5,22), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1, cv2.LINE_AA)
        return img

    # 1. Оригинал с диска (grayscale)
    orig_rgb = cv2.cvtColor(raw_orig_np, cv2.COLOR_GRAY2RGB)
    orig_labeled = put_label(orig_rgb, "Original (raw)")

    # 2. Original-Reconstructed (decoder)
    recon_arr = tensor2np(recon_orig_img)
    recon_rgb = cv2.cvtColor(recon_arr, cv2.COLOR_GRAY2RGB)
    recon_labeled = put_label(recon_rgb, "Original-Reconstructed")

    # 3. Watermarked
    wm_arr = tensor2np(wm_img)
    wm_rgb = cv2.cvtColor(wm_arr, cv2.COLOR_GRAY2RGB)
    wm_labeled = put_label(wm_rgb, "Watermarked")

    # 4. Seismic Map (former diff on white)
    seismic_diff_panel = get_seismic_diff_panel_np(
        recon_orig_img, wm_img, size=(512, 512),
        label="Signed Difference (seismic)", vmax=0.05)

    # Размер для всех масок
    H, W = recon_arr.shape

    # 5-7. Mask heatmaps
    from skimage.transform import resize


    def img2graypanel(img, title, H=512, W=512):
        img = to_np(img)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        img = normalize_01(img)
        img_gray = (img * 255).astype(np.uint8)
        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cv2.putText(img_color, title, (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return img_color

    def get_seismic_diff_panel_np(
            orig_img, wm_img, size=(512, 512), label="Signed Difference (seismic)", vmax=0.05
    ):
        # --- Convert tensors to numpy ---
        orig = orig_img.squeeze().detach().cpu().numpy()
        wm = wm_img.squeeze().detach().cpu().numpy()
        diff_signed = wm - orig

        # --- Проверка на NaN/Inf и удаление ---
        diff_signed = np.where(np.isfinite(diff_signed), diff_signed, 0.0)

        # Логируем значения только после получения diff_signed!
        print(f"Seismic diff mean={diff_signed.mean():.3f}, min={diff_signed.min():.3f}, max={diff_signed.max():.3f}")

        # --- Строго фиксированный диапазон ---
        # Белый — только при diff=0, синий/красный — даже при малых отличиях!
        # vmax задаёт reference (обычно 0.01)
        vmin = -vmax
        vmax = +vmax

        # --- Создание карты ---

        buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=(2.3, 2.1), dpi=160)
        im = ax.imshow(diff_signed, cmap='seismic', vmin=vmin, vmax=vmax)
        ax.axis('off')
        plt.tight_layout(pad=0)

        cax = inset_axes(ax, width="24%", height="2.5%", loc='lower right', borderpad=0.2)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label('Depth', fontsize=7)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=False)
        plt.close()
        buf.seek(0)
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = cv2.resize(img, size)

        # --- Подпись ---
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 50)
        except:
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([0, 0, img_pil.width, text_height + 6], fill=(255, 255, 255, 200))
        draw.text((6, 3), label, fill=(0, 0, 0), font=font)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img

    H, W = 512, 512
    # Финальный коллаж
    orig_panel = img2graypanel(raw_orig_np, "Original", H, W)
    recon_panel = img2graypanel(recon_orig_img, "Reconstructed", H, W)
    wm_panel = img2graypanel(wm_img, "Watermarked", H, W)

    # Здесь seismic panel — из твоей get_seismic_diff_panel_np
    seismic_diff_panel = get_seismic_diff_panel_np(
        recon_orig_img, wm_img, size=(512, 512), label="Signed Difference (seismic)", vmax=0.05)

    u2net_panel = mask2heat(u2net_mask_latent, "U2Net Mask", H, W)
    latent_panel = mask2heat(go_mask_latent, "Latent Mask", H, W)
    skip64_panel = mask2heat(go_mask_skip64, "Skip64 Mask", H, W)
    total_panel = mask2heat(total_no_go, "Total No-Go", H, W)

    # Новые панели: saliency_overlay и embedding_mask_overlay уже подписаны
    saliency_overlay_panel = saliency_overlay.copy()
    cv2.putText(saliency_overlay_panel, "C1 Saliency Map", (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    embedding_mask_overlay_panel = embedding_mask_overlay.copy()
    cv2.putText(embedding_mask_overlay_panel, "Embedding Mask Zone", (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Первый ряд (5 панелей)
    first_row = np.concatenate([
        orig_panel, recon_panel, wm_panel, seismic_diff_panel, saliency_overlay_panel
    ], axis=1)
    # Второй ряд (5 панелей)
    second_row = np.concatenate([
        u2net_panel, latent_panel, skip64_panel, total_panel, embedding_mask_overlay_panel
    ], axis=1)
    collage = np.concatenate([first_row, second_row], axis=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, collage)
    print(f"[OK] Saved collage: {out_path}")

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

#lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)  # Можно net='alex', но vgg обычно лучше

def to_3c(x):
    # x: [B, 1, H, W] -> [B, 3, H, W]
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    elif x.shape[1] == 3:
        return x
    else:
        raise ValueError(f"Tensor has {x.shape[1]} channels (expected 1 or 3). Shape: {x.shape}")

import torch.nn.functional as F

def norm_tensor(x):
    # x: [B, 1, H, W]
    return (x - x.amin(dim=(2,3), keepdim=True)) / (x.amax(dim=(2,3), keepdim=True) - x.amin(dim=(2,3), keepdim=True) + 1e-8)

# ============ TRAINING LOOP ============
def main():

    intensity = 0.25  # стартовая интенсивность watermark
    INTENSITY_STEP = 0.25
    INTENSITY_MAX = 1.25
    INTENSITY_MIN = 0.05
    MAX_C1_GAP = 0.05

    mask_fraction = 0.15  # стартовый payload (можешь начать с 0.10-0.15)
    mask_fraction_min = 0.05  # минимальный допустимый
    mask_fraction_max = 0.33  # максимальный допустимый
    payload_step = 0.03

    dataset = MRIImageDataset(DATA_ROOT, CLASSES, augment=mri_augment)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=False)

    # MODELS (FP16)
    wm_latent = WatermarkGen(PAYLOAD_DIM, SEED_DIM, LATENT_SHAPE).to(device)
    wm_skip64 = WatermarkGen(PAYLOAD_DIM, SEED_DIM, SKIP64_SHAPE).to(device)
    #wm_skip128 = WatermarkGen(PAYLOAD_DIM, SEED_DIM, SKIP128_SHAPE).to(device)

    state_dict = torch.load(ENCODER_PTH, map_location=device)
    encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    decoder_state = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}
    decoder.load_state_dict(decoder_state)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    c2 = EfficientNetB3_CBAM_Bottleneck(num_classes=4).to(device)
    mask_latent = MaskHead(1024).to(device)
    mask_skip64 = MaskHead(512).to(device)
    #mask_skip128 = MaskHead(256).to(device)
    # C1: strict eval, frozen
    c1 = EfficientNetB3_CBAM_Bottleneck(num_classes=4).to(device)
    state_dict = torch.load(C1_PATH, map_location='cpu')
    def clean_state_dict(state_dict):
        if any(k.startswith('module.') for k in state_dict):
            from collections import OrderedDict
            return OrderedDict((k.replace('module.', '', 1), v) for k, v in state_dict.items())
        return state_dict
    c1.load_state_dict(clean_state_dict(state_dict))
    for p in c1.parameters(): p.requires_grad = False
    c1.eval()

    optimizer_c2 = torch.optim.AdamW(c2.parameters(), lr=1e-5)
    optimizer_gen = torch.optim.AdamW(
        list(wm_latent.parameters()) +
        list(wm_skip64.parameters()) +
        #list(wm_skip128.parameters()) +  # <--- добавить!
        list(mask_latent.parameters()) +
        list(mask_skip64.parameters()),
        #list(mask_skip128.parameters()),  # <--- добавить!
        lr=1e-5
    )
    scaler = GradScaler()

    def entropy_loss(logits):
        prob = torch.softmax(logits, 1)
        log_prob = torch.log_softmax(logits, 1)
        entropy = -(prob * log_prob).sum(1)
        return entropy.mean()

    def norm(x):
        return (x - 0.5) / 0.5

    def save_metrics_csv(metrics, filename):
        if not metrics: return
        keys = sorted(metrics[0].keys())
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(metrics)

    def draw_label(img, text, font_size=32):
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_height = bbox[3] - bbox[1]
        draw.rectangle([0, 0, img_pil.width, text_height + 4], fill=(255, 255, 255, 200))
        draw.text((5, 2), text, fill=(0, 0, 0), font=font)
        return np.array(img_pil)

    for cycle in range(CYCLES):
        print(f"\n[WATERMARK] Cycle {cycle+1}, Intensity = {intensity:.3f}")
        early_stop = False

        cycle_metrics = []
        for epoch in range(EPOCHS_PER_CYCLE):
            print(f"  Epoch {epoch+1}/{EPOCHS_PER_CYCLE}")
            epoch_metrics = []
            cycle_root = os.path.join(CSV_ROOT, f"cycle{cycle + 1}_epoch{epoch + 1}_live.csv")
            for batch in loader:
                img_aug, ys, fnames, img_orig = batch
                img_aug = img_aug.to(device, non_blocking=True)
                check_nan_inf(img_aug, "img_aug")

                img_orig = img_orig.to(device, non_blocking=True)
                if img_orig.ndim == 3:
                    img_orig = img_orig.unsqueeze(1)  # [B, 1, H, W]
                img_orig_torch = img_orig.to(device, dtype=torch.float32) / 255.0

                ys = ys.to(device, non_blocking=True)

                B = img_aug.shape[0]
                with torch.no_grad():
                    latents, skip64s = encoder(img_aug)

                check_nan_inf(latents, "latents")
                check_nan_inf(skip64s, "skip64s")
                #check_nan_inf(skip128s, "skip128s")

                payload = torch.randint(0, 2, (B, PAYLOAD_DIM), dtype=torch.float32, device=device) * 2 - 1
                # Теперь payload ∈ {-1, +1}

                random_seed = torch.randn(B, SEED_DIM, device=device).half()
                with autocast(device_type="cuda"):
                    THRESH = 0.1

                    watermark_latent = wm_latent(payload, random_seed, latents, None)
                    watermark_skip64 = wm_skip64(payload, random_seed, skip64s, None)

                    # Загрузка масок U2Net
                    u2net_mask_latent = load_u2net_mask_batch(
                        fnames, u2net_mask_root, latents.shape[2:], device, class_names)
                    u2net_mask_skip64 = load_u2net_mask_batch(
                        fnames, u2net_mask_root, skip64s.shape[2:], device, class_names)

                    # Загрузка saliency mask (CBAM/GradCAM)
                    batch_class_names = [class_names[y.item()] for y in ys]  # ys — индексы классов

                    saliency_latent = load_saliency_mask_batch(
                        fnames, batch_class_names, saliency_root, latents.shape[2:], device)
                    saliency_skip64 = load_saliency_mask_batch(
                        fnames, batch_class_names, saliency_root, skip64s.shape[2:], device)

                    # Threshold (можешь выставить threshold, например, 0.3)

                    B = saliency_latent.size(0)
                    saliency_mask_lat = torch.zeros_like(saliency_latent)
                    saliency_mask_skip64 = torch.zeros_like(saliency_skip64)
                    for j in range(B):
                        sal_map = saliency_latent[j, 0].detach().cpu().numpy()
                        mask = get_percentile_mask(sal_map, mask_fraction=mask_fraction)
                        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
                        saliency_mask_lat[j] = mask_tensor
                        # Аналогично для skip64:
                        sal_map_skip = saliency_skip64[j, 0].detach().cpu().numpy()
                        mask_skip = get_percentile_mask(sal_map_skip, mask_fraction=mask_fraction)
                        mask_tensor_skip = torch.from_numpy(mask_skip).unsqueeze(0).to(device)
                        saliency_mask_skip64[j] = mask_tensor_skip

                    # Обучаемые маски C2
                    go_mask_latent = mask_latent(latents).clamp(0, 1)
                    go_mask_skip64 = mask_skip64(skip64s).clamp(0, 1)

                    # Итоговые no-go маски — пересечение с U2Net и saliency
                    combined_mask_lat = go_mask_latent * saliency_mask_lat * (1 - u2net_mask_latent)
                    combined_mask_skip64 = go_mask_skip64 * saliency_mask_skip64 * (1 - u2net_mask_skip64)

                    # Внедрение watermark с учетом комбинированных масок
                    watermarked_latent = latents + intensity * watermark_latent * combined_mask_lat
                    watermarked_skip64 = skip64s + intensity * watermark_skip64 * combined_mask_skip64

                    # Далее проверки и penalty как раньше
                    check_nan_inf(watermark_latent, "watermark_latent")
                    check_nan_inf(watermark_skip64, "watermark_skip64")
                    check_nan_inf(go_mask_latent, "go_mask_latent")
                    check_nan_inf(go_mask_skip64, "go_mask_skip64")
                    wm_mean_penalty_latent = torch.abs(watermark_latent.mean())
                    wm_mean_penalty_skip64 = torch.abs(watermark_skip64.mean())
                    wm_mean_penalty = wm_mean_penalty_latent + wm_mean_penalty_skip64

                    check_nan_inf(watermarked_latent, "watermarked_latent")
                    check_nan_inf(watermarked_skip64, "watermarked_skip64")

                    wm_img = decoder(watermarked_latent, watermarked_skip64)
                    clean_img = decoder(latents, skip64s)
                    wm_img_norm = normalize(wm_img).float()
                    clean_img_norm = normalize(clean_img).float()

                    check_nan_inf(wm_img, "wm_img")
                    check_nan_inf(clean_img, "clean_img")
                    diff = wm_img - clean_img
                    check_nan_inf(diff, "wm_img - clean_img")

                    for i in range(img_aug.size(0)):

                        raw_img_np = img_orig[i].cpu().numpy()
                        if raw_img_np.ndim == 3:
                            raw_img_np = raw_img_np.squeeze()
                        raw_img_np = np.clip(raw_img_np, 0, 255).astype(np.uint8)

                        target_shape = (512, 512)
                        u2net = F.interpolate(u2net_mask_latent[i:i + 1], target_shape, mode='nearest')
                        latent = F.interpolate(go_mask_latent[i:i + 1], target_shape, mode='nearest')
                        skip64 = F.interpolate(go_mask_skip64[i:i + 1], target_shape, mode='nearest')

                        total_no_go = torch.clamp(
                            u2net
                            + (1 - latent)
                            + (1 - skip64),
                            0, 1
                        )

                        # --- Payload/SSIM/L1 monitoring ---
                        embedding_mask = F.interpolate(saliency_mask_lat[i:i + 1], (512, 512), mode='nearest')[0, 0].detach().cpu().numpy()
                        payload_pixels = embedding_mask.sum()
                        payload_percent = 100 * payload_pixels / embedding_mask.size

                        wm_np = wm_img[i].detach().cpu().squeeze().numpy()
                        clean_np = clean_img[i].detach().cpu().squeeze().numpy()
                        l1_val = np.abs(wm_np - clean_np).mean()
                        ssim_val = calc_ssim(wm_np, clean_np)

                        print(f"[{fnames[i]}] Payload: {payload_pixels:.0f} px ({payload_percent:.2f}%), L1: {l1_val:.5f}, SSIM: {ssim_val:.4f}")

                        # --- Saliency overlay ---
                        saliency_lat_upsampled = F.interpolate(saliency_latent[i:i + 1], (512, 512), mode='bilinear', align_corners=False)
                        sal_map_np = saliency_lat_upsampled[0, 0].detach().cpu().numpy()

                        #print(f"[{fnames[i]}] Saliency: min={sal_map_np.min():.4f}, max={sal_map_np.max():.4f}, mean={sal_map_np.mean():.4f}")

                        #plt.figure(figsize=(6, 3))
                        #plt.hist(sal_map_np.ravel(), bins=50, color='red', alpha=0.75)
                        #plt.title(f"Saliency histogram for {fnames[i]}")
                        #plt.xlabel("Saliency value")
                        #plt.ylabel("Count")
                        #plt.grid(True)
                        #plt.tight_layout()
                        #plt.show()

                        sal_map_np_norm = (sal_map_np - sal_map_np.min()) / (sal_map_np.max() - sal_map_np.min() + 1e-8)
                        clean_np = clean_img[i].detach().cpu().squeeze().numpy()
                        clean_np = (clean_np - clean_np.min()) / (clean_np.max() - clean_np.min() + 1e-8)
                        clean_uint8 = (clean_np * 255).astype(np.uint8)
                        clean_rgb = cv2.cvtColor(clean_uint8, cv2.COLOR_GRAY2BGR)
                        sal_heat = cv2.applyColorMap((sal_map_np_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        sal_overlay = cv2.addWeighted(clean_rgb, 0.7, sal_heat, 0.3, 0)

                        # --- Embedding mask overlay ---
                        final_mask_upsampled = F.interpolate(combined_mask_lat[i:i + 1], (512, 512), mode='nearest')
                        mask_np = final_mask_upsampled[0, 0].detach().cpu().numpy()
                        mask_norm = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)
                        mask_rgb = np.zeros_like(clean_rgb)
                        mask_rgb[:, :, 1] = (mask_norm * 180).astype(np.uint8)
                        mask_overlay = cv2.addWeighted(clean_rgb, 0.7, mask_rgb, 0.3, 0)


                        save_path = os.path.join(
                            WM_ROOT, f"cycle{cycle + 1}", f"epoch{epoch + 1}",
                            CLASSES[ys[i].item()], f"{fnames[i]}_collage.png"
                        )

                        save_watermark_collage(
                            raw_img_np,
                            clean_img[i:i + 1].cpu(),
                            wm_img[i:i + 1].cpu(),
                            latent.cpu(),  # (1,1,512,512)
                            skip64.cpu(),  # (1,1,512,512)
                            u2net.cpu(),  # (1,1,512,512)
                            total_no_go.cpu(),
                            sal_overlay,  # <-- overlay saliency
                            mask_overlay,  # <-- overlay embedding mask
                            save_path
                        )

                print("latent      min/max/mean:", latents.min().item(), latents.max().item(), latents.mean().item())
                print("skip64      min/max/mean:", skip64s.min().item(), skip64s.max().item(), skip64s.mean().item())
                #print("skip128     min/max/mean:", skip128s.min().item(), skip128s.max().item(), skip128s.mean().item())
                print("wm_latent   min/max/mean:", watermark_latent.min().item(), watermark_latent.max().item(),
                      watermark_latent.mean().item())
                print("wm_skip64   min/max/mean:", watermark_skip64.min().item(), watermark_skip64.max().item(),
                      watermark_skip64.mean().item())
                #print("wm_skip128  min/max/mean:", watermark_skip128.min().item(), watermark_skip128.max().item(),
                      #watermark_skip128.mean().item())
                print("go_mask_latent   min/max/mean:", go_mask_latent.min().item(), go_mask_latent.max().item(),
                      go_mask_latent.mean().item())
                print("go_mask_skip64   min/max/mean:", go_mask_skip64.min().item(), go_mask_skip64.max().item(),
                      go_mask_skip64.mean().item())
                #print("go_mask_skip128  min/max/mean:", go_mask_skip128.min().item(), go_mask_skip128.max().item(),
                      #go_mask_skip128.mean().item())

                print(sum(p.numel() for p in wm_latent.parameters()))
                print(sum(p.numel() for p in wm_skip64.parameters()))
                #print(sum(p.numel() for p in wm_skip128.parameters()))
                print(sum(p.numel() for p in mask_latent.parameters()))
                print(sum(p.numel() for p in mask_skip64.parameters()))
                #print(sum(p.numel() for p in mask_skip128.parameters()))
                with torch.no_grad():
                    recon_img = decoder(latents, skip64s)
                    print(
                        "recon_img min/max/mean:",
                        recon_img.min().item(),
                        recon_img.max().item(),
                        recon_img.mean().item()
                    )

                with torch.no_grad():
                    acc_c1_clean = (c1(clean_img_norm).argmax(1) == ys).float().mean().item()
                    acc_c1_wm = (c1(wm_img_norm).argmax(1) == ys).float().mean().item()

                    c2_clean_pred = c2(clean_img_norm).argmax(1)
                    c2_wm_pred = c2(wm_img_norm).argmax(1)
                    acc_c2_clean = (c2_clean_pred == ys).float().mean().item()
                    acc_c2_wm = (c2_wm_pred == ys).float().mean().item()

                    # ----------- Auto-tune mask_fraction под C2 на clean ------------

                    if acc_c2_clean > 0.2:
                        mask_fraction = min(mask_fraction + payload_step, mask_fraction_max)
                        status = f"↑ INCREASE mask_fraction → {mask_fraction:.3f} (C2 Clean Acc {acc_c2_clean:.2f})"
                    elif acc_c2_clean < 0.07:
                        mask_fraction = max(mask_fraction - payload_step, mask_fraction_min)
                        status = f"↓ DECREASE mask_fraction → {mask_fraction:.3f} (C2 Clean Acc {acc_c2_clean:.2f})"
                    else:
                        status = f"mask_fraction stays at {mask_fraction:.3f} (C2 Clean Acc {acc_c2_clean:.2f})"
                    print(f"[Auto-tune Payload] {status}")

                # Новый adversarial C2-loss с alpha
                alpha = 0.5  # регулируй по задаче (0.3 - 1.0)
                logits_clean_c2 = c2(clean_img_norm)
                prob_clean = logits_clean_c2.softmax(1)
                entropy_clean = - (prob_clean * prob_clean.log()).sum(1).mean()
                loss_clean_c2 = -entropy_clean  # поощрять fail на clean

                logits_wm_c2 = c2(wm_img_norm)
                loss_wm_c2 = F.cross_entropy(logits_wm_c2, ys)  # поощрять угадывание на wm

                adv_c2_loss = alpha * loss_clean_c2 + loss_wm_c2

                l1_img = (wm_img - clean_img).abs().mean()

                energy_loss_latent = (watermark_latent.abs() * go_mask_latent).mean()
                energy_loss_skip64 = (watermark_skip64.abs() * go_mask_skip64).mean()
                #energy_loss_skip128 = (watermark_skip128.abs() * go_mask_skip128).mean()

                mask_sparsity_latent = go_mask_latent.mean()
                mask_sparsity_skip64 = go_mask_skip64.mean()
                #mask_sparsity_skip128 = go_mask_skip128.mean()

                mask_l2_latent = (go_mask_latent ** 2).mean()
                mask_l2_skip64 = (go_mask_skip64 ** 2).mean()
                #mask_l2_skip128 = (go_mask_skip128 ** 2).mean()

                # После go_mask_latent = ..., go_mask_skip64 = ..., go_mask_skip128 = ...
                mask_activation_latent = go_mask_latent.mean()
                mask_activation_skip64 = go_mask_skip64.mean()

                l1_recon = (clean_img - img_orig_torch).abs().mean()

                entropy_latent = -(go_mask_latent * torch.log(go_mask_latent + 1e-8) +
                                   (1 - go_mask_latent) * torch.log(1 - go_mask_latent + 1e-8)).mean()
                entropy_skip64 = -(go_mask_skip64 * torch.log(go_mask_skip64 + 1e-8) +
                                   (1 - go_mask_skip64) * torch.log(1 - go_mask_skip64 + 1e-8)).mean()

                penalty_c1 = torch.tensor(0.0, device=device)
                if acc_c1_wm < 0.97:
                    penalty_c1 = torch.tensor(abs(acc_c1_clean - acc_c1_wm) * 40.0, device=device)
                l1_recon = (clean_img - img_orig_torch).abs().mean()

                print("clean_img min/max/mean:",
                      clean_img.min().item(), clean_img.max().item(), clean_img.mean().item())
                print("img_orig_torch min/max/mean:",
                      img_orig_torch.min().item(), img_orig_torch.max().item(), img_orig_torch.mean().item())

                total_loss = (
                        adv_c2_loss
                        + 0.15 * l1_recon
                        + 0.15 * l1_img
                        # + 0.03 * lpips_wm_clean  # чуть мягче по imperceptibility (если нужно)
                        + 0.01 * energy_loss_latent
                        + 0.010 * energy_loss_skip64
                        # + 0.000 * energy_loss_skip128
                        + 0.05 * mask_activation_latent
                        + 0.050 * mask_activation_skip64
                        # + 0.000 * mask_activation_skip128
                        # + 0.00 * mask_sparsity_skip128
                        + 0.15 * mask_sparsity_skip64
                        + 0.15 * mask_sparsity_latent
                    # + 0.15 * (penalty_latent + penalty_skip64)
                )

                scaler.scale(total_loss).backward()
                scaler.step(optimizer_c2)
                scaler.step(optimizer_gen)
                scaler.update()
                optimizer_c2.zero_grad(set_to_none=True)
                optimizer_gen.zero_grad(set_to_none=True)



                # В начало main() ДО цикла:


                # В твоём батч-цикле, сразу после backward/update:
                if (len(epoch_metrics) % 1 == 0):
                    # C1 losses
                    with torch.no_grad():
                        logits_clean_c1 = c1(clean_img_norm)
                        logits_wm_c1 = c1(wm_img_norm)
                        acc_clean_c1 = (logits_clean_c1.argmax(1) == ys).float().mean().item()
                        acc_wm_c1 = (logits_wm_c1.argmax(1) == ys).float().mean().item()
                        loss_c1_clean = F.cross_entropy(logits_clean_c1, ys).item()
                        loss_c1_wm = F.cross_entropy(logits_wm_c1, ys).item()

                        logits_c2_clean = c2(clean_img_norm)
                        logits_c2_wm = c2(wm_img_norm)
                        loss_c2_clean = F.cross_entropy(logits_c2_clean, ys).item()
                        loss_c2_wm = F.cross_entropy(logits_c2_wm, ys).item()
                        acc_c2_clean = (logits_c2_clean.argmax(1) == ys).float().mean().item()
                        acc_c2_wm = (logits_c2_wm.argmax(1) == ys).float().mean().item()

                    metrics_row = {
                        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "cycle": cycle + 1,
                        "epoch": epoch + 1,
                        "batch": len(epoch_metrics),
                        "loss_c1_clean": loss_c1_clean,
                        "loss_c1_wm": loss_c1_wm,
                        "acc_c1_clean": acc_clean_c1,
                        "acc_c1_wm": acc_wm_c1,
                        "loss_c2_clean": loss_c2_clean,
                        "loss_c2_wm": loss_c2_wm,
                        "acc_c2_clean": acc_c2_clean,
                        "acc_c2_wm": acc_c2_wm,
                        "l1_img": l1_img.item(),
                        "l1_recon": l1_recon.item(),
                        "intensity": intensity,
                        'mask_fraction': mask_fraction,
                        'payload': payload_pixels,
                        'L1_img': l1_val,
                        'SSIM': ssim_val,
                        'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    log_batch_to_csv(metrics_row, cycle_root)

                    c1_clean_list = [m['acc_c1_clean'] for m in epoch_metrics]
                    c1_wm_list = [m['acc_c1_wm'] for m in epoch_metrics]

                    if len(c1_clean_list) == 0 or len(c1_wm_list) == 0:
                        mean_c1_clean = float('nan')
                        mean_c1_wm = float('nan')
                        print("Warning: Empty accuracy lists for C1, skipping intensity update.")
                    else:
                        mean_c1_clean = np.mean(c1_clean_list)
                        mean_c1_wm = np.mean(c1_wm_list)
                    gap = abs(mean_c1_clean - mean_c1_wm)
                    print(f"[C1‑gate] Mean C1 Clean Acc: {mean_c1_clean:.3f}, WM Acc: {mean_c1_wm:.3f}, Gap: {gap:.3f}")

                    if gap > MAX_C1_GAP:
                        print(f"[C1‑gate] GAP>{MAX_C1_GAP:.2f}. Watermark intensity will NOT increase (remains {intensity:.3f})!")
                        # intensity = max(intensity * 0.8, INTENSITY_MIN)  # если хочешь снижать
                    else:
                        intensity = min(intensity + INTENSITY_STEP, INTENSITY_MAX)
                        print(f"[C1‑gate] Watermark intensity increased → {intensity:.3f}")

                    # Печать на экран
                    print(
                        f"[BATCH {len(epoch_metrics)}] "
                        f"C1 Clean Loss: {loss_c1_clean:.4f} | C1 WM Loss: {loss_c1_wm:.4f} | "
                        f"C1 Clean Acc: {acc_c1_clean:.3f} | C1 WM Acc: {acc_c1_wm:.3f} || "
                        f"C2 Clean Loss: {loss_c2_clean:.4f} | C2 WM Loss: {loss_c2_wm:.4f} | "
                        f"C2 Clean Acc: {acc_c2_clean:.3f} | C2 WM Acc: {acc_c2_wm:.3f} | "
                        f"L1_img: {l1_img.item():.5f} | L1_recon: {l1_recon.item():.5f} | "
                        f"Time: {metrics_row['datetime']}"
                    )

                torch.cuda.empty_cache()

                # --- Logging batch metrics ---
                epoch_metrics.append({
                    'cycle': cycle+1, 'epoch': epoch+1,
                    'acc_c2_clean': acc_c2_clean, 'acc_c2_wm': acc_c2_wm,
                    'energy_loss_latent': energy_loss_latent.item(),
                    'energy_loss_skip64': energy_loss_skip64.item(),
                    'intensity': intensity,
                    'acc_c1_clean': acc_c1_clean, 'acc_c1_wm': acc_c1_wm,
                    'c1_drop': acc_c1_clean - acc_c1_wm,
                    "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "batch": len(epoch_metrics),
                    "loss_c1_clean": loss_c1_clean,
                    "loss_c1_wm": loss_c1_wm,
                    "loss_c2_clean": loss_c2_clean,
                    "loss_c2_wm": loss_c2_wm,
                    "l1_img": l1_img.item(),
                    "l1_recon": l1_recon.item(),
                    'mask_fraction': mask_fraction,
                    'payload': payload_pixels,
                    'L1_img': l1_val,
                    'SSIM': ssim_val,
                    'time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            # --- Save per-epoch CSV ---
            csv_file = os.path.join(CSV_ROOT, f"metrics_cycle{cycle+1}_epoch{epoch+1}.csv")
            save_metrics_csv(epoch_metrics, csv_file)
            # --- Early stopping ---
            if (acc_c2_clean < EARLY_STOP_C2_CLEAN) and (acc_c2_wm > EARLY_STOP_C2_WM):
                print(f"Early stopping in cycle {cycle+1} at epoch {epoch+1}")
                early_stop = True
                break
        # --- Save cycle CSV ---
        csv_file = os.path.join(CSV_ROOT, f"metrics_cycle{cycle+1}_full.csv")
        save_metrics_csv(cycle_metrics, csv_file)
        if early_stop: continue

    print("Training complete. All results saved.")

if __name__ == "__main__":
    import multiprocessing; multiprocessing.set_start_method('spawn', force=True)
    main()

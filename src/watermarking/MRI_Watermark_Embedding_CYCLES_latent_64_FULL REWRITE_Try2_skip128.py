import os, csv, torch, random, numpy as np
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
#import lpips
import cv2
import io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cv2

# ============ CONFIG ============
DATA_ROOT = r"E:\MRI_LOWMEM\tiny_mri"
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
SAL_ROOT = r"E:\MRI_LOWMEM\saliency_cycles_latent_skip64"
WM_ROOT  = r"E:\MRI_LOWMEM\watermarked_cycles_latent_skip64"
CSV_ROOT = r"E:\MRI_LOWMEM\metrics_logs_latent_skip64"
C1_PATH = r"E:\MRI_LOWMEM\C1-B3-CBAM\MRI-C1EfficientNet_B3_CBAM.pth"
NPZ_ROOT = r"E:\MRI_LOWMEM\MRI-NPZ_latent_skip64"
ENCODER_PTH = r"E:\MRI_LOWMEM\Encoder_latent_64\autoencoder_epoch7.pth"


CYCLES = 5
EPOCHS_PER_CYCLE = 5
BATCH_SIZE = 2
INTENSITIES = [0.25, 0.50, 0.75, 1.0, 1.25]

EARLY_STOP_C2_CLEAN = 0.20
EARLY_STOP_C2_WM    = 0.90

PAYLOAD_DIM, SEED_DIM = 256, 32
LATENT_SHAPE, SKIP64_SHAPE, SKIP128_SHAPE = (1024, 32, 32), (512, 64, 64), (256, 128, 128)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(SAL_ROOT, exist_ok=True)
os.makedirs(WM_ROOT, exist_ok=True)
os.makedirs(CSV_ROOT, exist_ok=True)

encoder = Encoder().to(device)
decoder = Decoder().to(device)


'''
# ============ DATASET ============
class MRIWatermarkDataset(Dataset):
    def __init__(self, root, npz_root, classes):
        self.samples = []
        for cls in classes:
            folder = os.path.join(root, cls)
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.png','.jpg')):
                    npz_path = os.path.join(npz_root, cls, os.path.splitext(fname)[0]+'.npz')
                    if os.path.exists(npz_path):
                        self.samples.append((os.path.join(folder, fname), npz_path, cls))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, npz_path, cls = self.samples[idx]
        img = Image.open(img_path).convert('L').resize((512,512), Image.BICUBIC)
        img = torch.from_numpy(np.array(img, dtype=np.uint8)[None,:,:]).to(torch.float32) / 255.0
        with np.load(npz_path, mmap_mode='r') as d:
            latent = torch.from_numpy(d['latent']).to(torch.float32)
            skip64 = torch.from_numpy(d['skip64']).to(torch.float32)
        label = CLASSES.index(cls)
        return img, latent, skip64, label, os.path.basename(img_path)
'''


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
"""
class WatermarkGen(nn.Module):
    def __init__(self, payload_dim, seed_dim, latent_shape):
        super().__init__()
        self.latent_shape = latent_shape  # остальные параметры не нужны для теста

    def forward(self, payload, seed, x, _=None):
        # Вариант 1: всё 0.5 (одинаковый watermark)
        return torch.ones_like(x) * -0.5
        # Вариант 2: всё -0.5 (чтобы увидеть отрицательный сдвиг)
        # return torch.ones_like(x) * -0.5
        # Вариант 3: случайный шум
        # return torch.randn_like(x)
"""


class MaskHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        nn.init.constant_(self.conv.bias, 1.0)  # <-- bias в +1 (можно от 0.5 до 2.0 тестировать)
    def forward(self, x):
        return torch.tanh(self.conv(x))


'''
# ============ DECODER ============
class Decoder(nn.Module):
    def __init__(self, bottleneck_channels=1024):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(bottleneck_channels, 512, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(512+512, 256, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv5 = nn.Conv2d(64, 1, 3, 1, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, latent, skip64):
        x = self.relu(self.deconv1(latent))
        x = torch.cat([x, skip64], 1)
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.relu(self.deconv4(x))
        x = self.tanh(self.deconv5(x))
        return (x + 1) / 2
'''
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


def save_watermark_collage(
    raw_orig_np,                # numpy [H,W] uint8, raw оригинал (с диска)
    recon_orig_img,             # torch.Tensor [1,1,H,W] или [1,H,W], [0,1]
    wm_img,                     # torch.Tensor [1,1,H,W] или [1,H,W], [0,1]
    go_mask_latent, go_mask_skip64, #go_mask_skip128,
    fname, out_dir
):
    # Вспомогательные функции
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
        recon_orig_img, wm_img, size=(512, 512), label="Signed Difference (seismic)", vmax=0.05
    )

    # Размер для всех масок
    H, W = recon_arr.shape

    # 5-7. Mask heatmaps
    from skimage.transform import resize

    def amp2heat(wm_img, clean_img, label, H, W):
        wm = wm_img.squeeze().detach().cpu().numpy()
        cl = clean_img.squeeze().detach().cpu().numpy()
        amp = np.abs(wm - cl)
        amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)  # <--- добавь эту строку!
        if amp.shape != (H, W):
            try:
                amp_resized = resize(amp, (H, W), order=1, mode='constant', anti_aliasing=True, preserve_range=True)
            except Exception as e:
                print(f"[ERROR] skimage.resize failed for amp shape {amp.shape} label {label}: {e}")
                amp_resized = np.zeros((H, W), dtype=np.float32)
        else:
            amp_resized = amp
        amp_resized = np.nan_to_num(amp_resized, nan=0.0, posinf=0.0, neginf=0.0)  # <--- ещё раз для надёжности
        vmax = np.percentile(amp_resized, 99.9)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        amp_norm = np.clip(amp_resized / (vmax + 1e-8), 0, 1)
        amp_norm = np.nan_to_num(amp_norm, nan=0.0, posinf=0.0, neginf=0.0)  # <--- и на финальный массив!
        print(
            f"[{label}] Real WM amplitude mean={amp_norm.mean():.3f}, min={amp_norm.min():.3f}, max={amp_norm.max():.3f}, vmax={vmax:.4f}")
        heat = cv2.applyColorMap((amp_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return put_label(heat, label)

    def mask2heat(mask, label, H, W):
        mask = mask.detach().cpu().numpy()
        mask = np.squeeze(mask)
        if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0 or np.ptp(mask) == 0 or not np.isfinite(
                mask).all():
            print(f"[WARNING] mask2heat: invalid or empty mask for {label}, shape {mask.shape}. Filling zeros.")
            mask = np.zeros((H, W), dtype=np.float32)
        if mask.shape != (H, W):
            try:
                mask_resized = resize(mask, (H, W), order=1, mode='constant', anti_aliasing=True, preserve_range=True)
            except Exception as e:
                print(f"[ERROR] skimage.resize failed for mask shape {mask.shape} label {label}: {e}")
                mask_resized = np.zeros((H, W), dtype=np.float32)
        else:
            mask_resized = mask
        mask_norm = np.clip(mask_resized, 0, 1)
        print(f"[{label}] Mask mean={mask_norm.mean():.3f}, min={mask_norm.min():.3f}, max={mask_norm.max():.3f}")
        heat = cv2.applyColorMap((mask_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return put_label(heat, label)

    latent_labeled = mask2heat(go_mask_latent, "Latent Mask Heatmap", H, W)
    skip64_labeled = mask2heat(go_mask_skip64, "Skip64 Mask Heatmap", H, W)
    #skip128_labeled = mask2heat(go_mask_skip128, "Skip128 Mask Heatmap", H, W)

    real_wm_amp_heat = amp2heat(wm_img, recon_orig_img, "Real WM Amplitude", H, W)

    # Финальный коллаж
    collage = np.concatenate([
        orig_labeled,
        recon_labeled,
        wm_labeled,
        seismic_diff_panel,
        latent_labeled,
        skip64_labeled,
        # skip128_labeled,      # <-- УДАЛИТЬ
        real_wm_amp_heat
    ], axis=1)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{fname}_collage.png")
    cv2.imwrite(out_path, collage)
    print(f"[OK] Saved collage: {out_path}")



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
        intensity = INTENSITIES[cycle]
        early_stop = False
        print(f"\n=== Cycle {cycle+1}/{CYCLES} | Intensity: {intensity} ===")
        cycle_metrics = []
        for epoch in range(EPOCHS_PER_CYCLE):
            print(f"  Epoch {epoch+1}/{EPOCHS_PER_CYCLE}")
            epoch_metrics = []
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

                with torch.no_grad():
                    # img_aug: [B, 1, 512, 512]
                    if hasattr(c1, 'module'):
                        att_map = c1.module.get_cbam_attention(img_aug)
                    else:
                        att_map = c1.get_cbam_attention(img_aug)

                att_map_latent = F.interpolate(att_map, size=latents.shape[2:], mode='bilinear', align_corners=False)
                att_map_skip64 = F.interpolate(att_map, size=skip64s.shape[2:], mode='bilinear', align_corners=False)

                norm_att_latent = norm_tensor(att_map_latent)
                norm_att_skip64 = norm_tensor(att_map_skip64)

                THRESH = 0.90
                tumor_mask_latent = (norm_att_latent > THRESH).float()
                tumor_mask_skip64 = (norm_att_skip64 > THRESH).float()



                random_seed = torch.randn(B, SEED_DIM, device=device).half()
                with autocast(device_type="cuda"):
                    watermark_latent = wm_latent(payload, random_seed, latents, None)
                    watermark_skip64 = wm_skip64(payload, random_seed, skip64s, None)
                    #watermark_skip128 = wm_skip128(payload, random_seed, skip128s, None)
                    go_mask_latent = mask_latent(latents).clamp(0, 1)
                    go_mask_skip64 = mask_skip64(skip64s).clamp(0, 1)

                    # Применить запрет на tumor:
                    go_mask_latent = go_mask_latent * (1 - tumor_mask_latent)
                    go_mask_skip64 = go_mask_skip64 * (1 - tumor_mask_skip64)

                    penalty_latent = (go_mask_latent * tumor_mask_latent).mean()
                    penalty_skip64 = (go_mask_skip64 * tumor_mask_skip64).mean()
                    #total_loss = total_loss + 0.05 * (penalty_latent + penalty_skip64)

                    # Внедрение watermark:
                    watermarked_latent = (latents + intensity * watermark_latent * go_mask_latent)
                    watermarked_skip64 = (skip64s + intensity * watermark_skip64 * go_mask_skip64)

                    check_nan_inf(watermark_latent, "watermark_latent")
                    check_nan_inf(watermark_skip64, "watermark_skip64")
                    #check_nan_inf(watermark_skip128, "watermark_skip128")

                    check_nan_inf(go_mask_latent, "go_mask_latent")
                    check_nan_inf(go_mask_skip64, "go_mask_skip64")
                    #check_nan_inf(go_mask_skip128, "go_mask_skip128")

                    # <---- ВСТАВЬ СЮДА penalty за среднее значение watermark:
                    wm_mean_penalty_latent = torch.abs(watermark_latent.mean())
                    wm_mean_penalty_skip64 = torch.abs(watermark_skip64.mean())
                    #wm_mean_penalty_skip128 = torch.abs(watermark_skip128.mean())
                    wm_mean_penalty = wm_mean_penalty_latent + wm_mean_penalty_skip64  #+wm_mean_penalty_skip128
                    # --------------------------------------


                    check_nan_inf(watermarked_latent, "watermarked_latent")
                    check_nan_inf(watermarked_skip64, "watermarked_skip64")
                    #check_nan_inf(watermarked_skip128, "watermarked_skip128")


                    wm_img = decoder(watermarked_latent, watermarked_skip64)
                    clean_img = decoder(latents, skip64s)
                    wm_img_norm = norm(wm_img).float()
                    clean_img_norm = norm(clean_img).float()

                    check_nan_inf(wm_img, "wm_img")
                    check_nan_inf(clean_img, "clean_img")
                    diff = wm_img - clean_img
                    check_nan_inf(diff, "wm_img - clean_img")

                for i in range(B):
                    raw_img_np = img_orig[i].cpu().numpy()
                    if raw_img_np.ndim == 3:
                        raw_img_np = raw_img_np.squeeze()
                    raw_img_np = np.clip(raw_img_np, 0, 255).astype(np.uint8)

                    class_name = CLASSES[ys[i].item()]
                    out_dir = os.path.join(
                        WM_ROOT,
                        f"cycle{cycle + 1}",
                        f"epoch{epoch + 1}",
                        class_name
                    )
                    save_watermark_collage(
                        raw_img_np,
                        clean_img[i:i + 1].cpu(),
                        wm_img[i:i + 1].cpu(),
                        go_mask_latent[i:i + 1].cpu(),
                        go_mask_skip64[i:i + 1].cpu(),
                        fnames[i],
                        out_dir
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
                logits = c2(wm_img_norm)
                loss_wm = F.cross_entropy(logits, ys)
                l1_img = (wm_img - clean_img).abs().mean()

                loss_clean = -entropy_loss(c2(clean_img_norm))

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
                #mask_activation_skip128 = go_mask_skip128.mean()


                # [0,1] -> [-1,1] для LPIPS, batch first, channel first (B,1,H,W)
                # [0,1] -> [-1,1] для LPIPS
                #wm_img_lpips = to_3c((wm_img.clamp(0, 1) * 2 - 1).float())
                #clean_img_lpips = to_3c((clean_img.clamp(0, 1) * 2 - 1).float())
                #img_orig_lpips = to_3c((img_orig_torch.clamp(0, 1) * 2 - 1).float())

                #print("LPIPS shapes:", wm_img_lpips.shape, clean_img_lpips.shape, img_orig_lpips.shape)

                #lpips_wm_clean = lpips_loss_fn(wm_img_lpips, clean_img_lpips).mean()
                #lpips_clean_orig = lpips_loss_fn(clean_img_lpips, img_orig_lpips).mean()

                l1_recon = (clean_img - img_orig_torch).abs().mean()

                entropy_latent = -(go_mask_latent * torch.log(go_mask_latent + 1e-8) +
                                   (1 - go_mask_latent) * torch.log(1 - go_mask_latent + 1e-8)).mean()
                entropy_skip64 = -(go_mask_skip64 * torch.log(go_mask_skip64 + 1e-8) +
                                   (1 - go_mask_skip64) * torch.log(1 - go_mask_skip64 + 1e-8)).mean()
                #entropy_skip128 = -(go_mask_skip128 * torch.log(go_mask_skip128 + 1e-8) +
                 #                   (1 - go_mask_skip128) * torch.log(1 - go_mask_skip128 + 1e-8)).mean()

                penalty_c1 = torch.tensor(0.0, device=device)
                if acc_c1_wm < 0.97:
                    penalty_c1 = torch.tensor(abs(acc_c1_clean - acc_c1_wm) * 40.0, device=device)
                l1_recon = (clean_img - img_orig_torch).abs().mean()

                print("clean_img min/max/mean:",
                      clean_img.min().item(), clean_img.max().item(), clean_img.mean().item())
                print("img_orig_torch min/max/mean:",
                      img_orig_torch.min().item(), img_orig_torch.max().item(), img_orig_torch.mean().item())



                # total_loss = (
                #         loss_wm + loss_clean
                #         + 0.04 * l1_recon
                #         + 0.05 * l1_img
                #         + 0.0 * lpips_wm_clean
                #         + 0.0 * lpips_clean_orig
                #         + 0.005 * energy_loss_latent + 0.005 * energy_loss_skip64 + 0.005 * energy_loss_skip128
                #         + 0.0 * mask_sparsity_latent
                #         + 0.0 * mask_sparsity_skip64
                #         + 0.0 * mask_sparsity_skip128
                #         + 0.0 * entropy_latent
                #         + 0.0 * entropy_skip64
                #         + 0.0 * entropy_skip128
                #         + 0.0 * wm_mean_penalty_latent
                #         + 0.0 * wm_mean_penalty_skip64
                #         + 0.0 * wm_mean_penalty_skip128
                #         + 0.01 * mask_activation_latent
                #         + 0.01 * mask_activation_skip64
                #         + 0.01 * mask_activation_skip128
                # )

                total_loss = (
                        loss_wm + loss_clean
                        + 0.15 * l1_recon
                        + 0.15 * l1_img
                        #+ 0.03 * lpips_wm_clean  # чуть мягче по imperceptibility (если нужно)
                        + 0.01 * energy_loss_latent
                        + 0.010 * energy_loss_skip64
                       # + 0.000 * energy_loss_skip128
                        + 0.05 * mask_activation_latent  # мягко (latent пусть живёт)
                        + 0.050 * mask_activation_skip64  # почти убираем стимул
                       # + 0.000 * mask_activation_skip128  # максимально охлаждаем стимул
                       # + 0.00 * mask_sparsity_skip128  # в 2 раза больше sparsity penalty
                        + 0.15 * mask_sparsity_skip64  # ещё сильнее sparsity penalty
                        + 0.15 * mask_sparsity_latent  # можно чуть усилить sparsity на latent
                        + 0.15 * (penalty_latent + penalty_skip64)
                )


                scaler.scale(total_loss).backward()
                scaler.step(optimizer_c2)
                scaler.step(optimizer_gen)
                scaler.update()
                optimizer_c2.zero_grad(set_to_none=True)
                optimizer_gen.zero_grad(set_to_none=True)

                import datetime

                # В начало main() ДО цикла:
                log_batch_metrics = []

                # В твоём батч-цикле, сразу после backward/update:
                if (len(epoch_metrics) % 1 == 0):
                    # C1 losses
                    with torch.no_grad():
                        logits_c1_clean = c1(clean_img_norm)
                        logits_c1_wm = c1(wm_img_norm)
                        loss_c1_clean = F.cross_entropy(logits_c1_clean, ys).item()
                        loss_c1_wm = F.cross_entropy(logits_c1_wm, ys).item()

                        logits_c2_clean = c2(clean_img_norm)
                        logits_c2_wm = c2(wm_img_norm)
                        loss_c2_clean = F.cross_entropy(logits_c2_clean, ys).item()
                        loss_c2_wm = F.cross_entropy(logits_c2_wm, ys).item()

                    metrics_row = {
                        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "cycle": cycle + 1,
                        "epoch": epoch + 1,
                        "batch": len(epoch_metrics),
                        "loss_c1_clean": loss_c1_clean,
                        "loss_c1_wm": loss_c1_wm,
                        "acc_c1_clean": acc_c1_clean,
                        "acc_c1_wm": acc_c1_wm,
                        "loss_c2_clean": loss_c2_clean,
                        "loss_c2_wm": loss_c2_wm,
                        "acc_c2_clean": acc_c2_clean,
                        "acc_c2_wm": acc_c2_wm,
                        "l1_img": l1_img.item(),
                        "l1_recon": l1_recon.item(),
                        "intensity": intensity,
                        # Можно добавить любые доп.метрики тут!
                    }
                    log_batch_metrics.append(metrics_row)

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

                    # Сохраняем файл в корневой директории (каждые 100 батчей или в конце эпохи)
                    if (len(epoch_metrics) % 100 == 0 or (len(epoch_metrics) + 1 == len(loader))):
                        cycle_root = os.path.join(CSV_ROOT,
                                                  f"cycle{cycle + 1}_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                        import csv
                        with open(cycle_root, 'w', newline='') as f:
                            writer = csv.DictWriter(f, log_batch_metrics[0].keys())
                            writer.writeheader()
                            writer.writerows(log_batch_metrics)

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

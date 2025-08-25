# -*- coding: utf-8 -*-
"""
Safe-watermark training pipeline (MRI 1ch, 512×512):
- U2Net exclusion (hard no-go)
- Auto-quantile "maskability" map
- Auto-tune mask_fraction & intensity from C2 signals
- Frozen C1 & Frozen Autoencoder
- C2: entropy on clean + CE on WM
- CSV logs & C2 checkpoints
"""

import os, sys, csv, glob, time, math, random, pathlib
from datetime import datetime
import numpy as np
import cv2
import csv
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from collections import deque
from skimage.metrics import structural_similarity as ssim
from torch.fft import fft2, ifft2

# ====== YOUR LOCAL MODULES (must exist) =====
from MRI_C1_B3_CBAM import EfficientNetB3_CBAM_Bottleneck
from MRI_Encoder_Latent_64_Train import Encoder, Decoder
# ===========================================

# ---------------- Paths & Const ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device= "cuda" if torch.cuda.is_available() else "cpu"

# Get project root directory (two levels up from current file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_ROOT   = os.path.join(PROJECT_ROOT, "dataset", "brain-tumor-mri-dataset", "Training")
U2NET_ROOT   = os.path.join(PROJECT_ROOT, "output", "u2net_masks_png")
WM_ROOT      = os.path.join(PROJECT_ROOT, "output", "watermarked_cycles_latent_skip64_U2NET_AUTOTUNE")
CSV_ROOT     = os.path.join(PROJECT_ROOT, "output", "csv_logs")
C1_PATH      = os.path.join(PROJECT_ROOT, "pt models", "MRI-C1EfficientNet_B3_CBAM.pth")
AE_PATH      = os.path.join(PROJECT_ROOT, "pt models", "autoencoder_epoch7.pth")
C2_SAVE_DIR  = os.path.join(PROJECT_ROOT, "output", "C2-B3")

os.makedirs(WM_ROOT, exist_ok=True)
os.makedirs(CSV_ROOT, exist_ok=True)
os.makedirs(C2_SAVE_DIR, exist_ok=True)

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS2IDX = {c:i for i,c in enumerate(CLASSES)}

# ---------------- Training hyperparams ----------------
BATCH_SIZE          = 4
EPOCHS              = 10
NUM_WORKERS         = 6
AMP                 = True

# C2 schedule
C2_STAGE_A_EPOCHS   = 2     # только голова
LR_HEAD             = 1e-4  # голова
LR_BACKBONE         = 1e-4  # поздний unfreeze

# Adversarial loss weight
ALPHA_CLEAN_ENTROPY = 1.25

# Payload / intensity auto-tune
MASK_FRAC_INIT      = 0.40
MASK_FRAC_MIN       = 0.10   #
MASK_FRAC_MAX       = 0.80
MASK_FRAC_STEP      = 0.1

INTENSITY_INIT      = 0.60
INTENSITY_MIN       = 0.10   #
INTENSITY_MAX       = 3.00
INTENSITY_STEP      = 0.10

# --- Параметры сглаживания и авто-тюна payload ---
SMOOTH_N     = 8      # окно скользящего среднего по батчам
CLEAN_LOW    = 0.15   # если smoothed C2 clean acc ниже — уменьшаем payload
CLEAN_HIGH   = 0.45 # если выше — увеличиваем payload (пусть путается)
WM_MIN_OK    = 0.60   # если wm-acc ниже — слегка подкинем payload вверх

MF_STEP_UP   = 0.01   # шаг увеличения mask_fraction
MF_STEP_DOWN = 0.02   # шаг уменьшения mask_fraction
MF_MIN       = 0.05   # нижняя граница payload (по площади)
MF_MAX       = 0.30   # верхняя граница payload
mask_fraction = 0.15  # стартовое значение
TARGET_PAYLOAD_PCT  = (10.0, 18.0)   # ориентир в процентах площади на 512×512
C1_GAP_MAX          = 0.30          # не портить C1

# --- Normalization blocks ---
C1_MEAN = torch.tensor([0.5], device=DEVICE).view(1, 1, 1, 1)
C1_STD  = torch.tensor([0.5], device=DEVICE).view(1, 1, 1, 1)

def norm_c1(x01):        # x01: (B,1,512,512) в [0,1]
    return (x01 - C1_MEAN) / C1_STD

def denorm_c1(xn):       # обратно в [0,1]
     x = xn * C1_STD + C1_MEAN
     return x.clamp(0, 1)

#def to_3c(x):  # (B,1,H,W) -> (B,3,H,W) для grayscale
#    return x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)

#def norm_c1(x01):        # ожидает [0,1]
#    x3 = to_3c(x01)
#    return (x3 - C1_MEAN) / C1_STD

#def denorm_c1(xn):       # обратно в [0,1]
#    x3 = xn * C1_STD + C1_MEAN
#    return x3.clamp(0, 1)

# Seeds
def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(1337)

def _is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def jet_map(x):
    x = np.asarray(x)
    # squeeze лишние размерности
    if x.ndim > 2:
        x = x.squeeze()
    # убедись, что размер (512,512)
    if x.shape != (512, 512):
        x = cv2.resize(x, (512, 512), interpolation=cv2.INTER_LINEAR)
    # нормализуй в [0,1] если надо
    if x.max() > 1.0 or x.min() < 0.0:
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x8 = np.clip((x * 255).astype(np.uint8), 0, 255)
    return cv2.applyColorMap(x8, cv2.COLORMAP_JET)


def log_row_to_csv(row: Dict, csv_path: str):
    """
    Append a single row (dict) into CSV at csv_path.
    If file does not exist, writes header first.
    """
    CSV_FIELDS = [
        "time", "cycle", "epoch", "batch",
        "acc_c1_clean", "acc_c1_wm", "acc_c2_clean", "acc_c2_wm",
        "loss_c2_clean", "loss_c2_wm", "adv_c2_loss",
        "l1_img", "ssim", "payload_pixels", "payload_pct",
        "mask_fraction", "intensity",
        "frac_list", "payload_px_list", "payload_pct_list"
    ]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

class WatermarkGeneratorMiniUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels_latent=1024, out_channels_skip=512):
        super().__init__()
        self.enc1 = nn.Conv2d(in_channels, 32, 3, 1, 1)
        self.enc2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.enc3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.enc4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.lat_bridge = nn.Conv2d(128, 32, 1)
        self.skip_bridge = nn.Conv2d(128, 64, 1)         # <--- Добавь это!
        self.lat_head = nn.Conv2d(32, out_channels_latent, 1)
        self.skip_head = nn.Conv2d(64, out_channels_skip, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, allow_mask_lat, allow_mask_skip, intensity=1.0):
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        e4 = self.relu(self.enc4(e3))
        u3 = self.relu(self.up3(e4)) + e3
        u2 = self.relu(self.up2(u3)) + e2
        u1 = self.relu(self.up1(u2)) + e1

        # Для latent
        z_lat = F.interpolate(u3, size=(32,32), mode='bilinear')
        z_lat = self.relu(self.lat_bridge(z_lat))
        wm_lat = self.lat_head(z_lat)

        # Для skip
        z_skip = F.interpolate(u3, size=(64,64), mode='bilinear')
        z_skip = self.relu(self.skip_bridge(z_skip))      # <--- Вот эта строка!
        wm_skip = self.skip_head(z_skip)

        wm_lat = intensity * wm_lat * allow_mask_lat
        wm_skip = intensity * wm_skip * allow_mask_skip
        return wm_lat, wm_skip





def save_epoch_summary(rows: List[Dict], out_path: str, extra_meta: Dict = None):
    """
    Compute mean for all numeric keys across rows and write a 1-row CSV.
    Keeps non-numeric meta as is (if provided via extra_meta).
    """
    if len(rows) == 0:
        # still create an empty CSV with meta if possible
        with open(out_path, 'w', newline='') as f:
            if extra_meta:
                w = csv.DictWriter(f, fieldnames=list(extra_meta.keys()))
                w.writeheader()
                w.writerow(extra_meta)
            else:
                f.write("")  # empty file
        return

    # aggregate numeric means
    agg = {}
    keys = set().union(*[r.keys() for r in rows])
    for k in keys:
        vals = [r[k] for r in rows if (k in r)]
        num_vals = [float(v) for v in vals if _is_number(v)]
        if len(num_vals) > 0:
            agg[k] = float(np.mean(num_vals))

    # merge with meta (cycle/epoch etc.)
    if extra_meta:
        for mk, mv in extra_meta.items():
            agg[mk] = mv

    # write single-row CSV
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(agg.keys()))
        w.writeheader()
        w.writerow(agg)


# ---------------- Dataset ----------------
class MRIDataset(Dataset):
    def __init__(self, root, classes):
        self.root = root
        self.classes = classes
        self.items = []
        for c in classes:
            for fp in glob.glob(os.path.join(root, c, "*.jpg")):
                self.items.append((fp, c))
        self.items.sort()
        self.tf = transforms.Compose([
            transforms.ToTensor(),  # [0..1], C×H×W, C=1 if grayscale read later
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, cls = self.items[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 1ch
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        img = (img.astype(np.float32) / 255.0)
        img_t = torch.from_numpy(img)[None, ...]  # 1×H×W
        y = CLASS2IDX[cls]
        fname = os.path.basename(path)
        return img_t, y, fname, cls

# ---------------- Utils ----------------

def make_heat(x):  # x in [0,1] float np
    x8 = np.clip((x*255).astype(np.uint8), 0, 255)
    return cv2.applyColorMap(x8, cv2.COLORMAP_JET)

def make_collage_save(save_path, raw_gray_512, clean_512, wm_512,
                      u2_512, allow_bin_512, maskability_512, diff_512,
                      spectrum_latent_512=None, spectrum_skip_512=None):
    """
    raw_gray_512, clean_512, wm_512: np float [0..1], H×W
    u2_512, allow_bin_512, maskability_512: np float [0..1], H×W
    diff_512: np float (может быть отриц), H×W -> визуализируем симметрично
    spectrum_latent_512, spectrum_skip_512: np float [0..1], H×W
    """
    H, W = 512, 512

    def jet_map(x):
        x = np.asarray(x)
        x = np.squeeze(x)
        # Always ensure 2D (512,512)
        if x.shape != (512, 512):
            x = cv2.resize(x, (512, 512), interpolation=cv2.INTER_LINEAR)
        # Normalize if needed
        if x.max() > 1.0 or x.min() < 0.0:
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x8 = np.clip((x * 255).astype(np.uint8), 0, 255)
        return cv2.applyColorMap(x8, cv2.COLORMAP_JET)

    spectrum_lat_img = jet_map(spectrum_latent_512) if spectrum_latent_512 is not None else np.zeros((512, 512, 3), np.uint8)
    spectrum_skip_img = jet_map(spectrum_skip_512) if spectrum_skip_512 is not None else np.zeros((512, 512, 3), np.uint8)

    def gray3(x):
        g = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    # Собираем изображения для первого ряда
    images = [
        gray3(raw_gray_512),              # 0. Original
        gray3(clean_512),                 # 1. Clean Recon
        gray3(wm_512),                    # 2. Watermarked Recon
        jet_map((diff_512 - diff_512.min()) / (diff_512.max() - diff_512.min() + 1e-8)), # 3. Diff Heatmap
        cv2.addWeighted(gray3(clean_512), 0.7, (np.stack([u2_512*255]*3, axis=-1)).astype(np.uint8), 0.3, 0), # 4. U2Net Overlay
        spectrum_lat_img,                 # 5. Latent Spectrum
        spectrum_skip_img                 # 6. Skip64 Spectrum
    ]
    row1_labels = ['Original', 'Clean', 'Watermarked', 'Diff', 'U2Net Overlay', 'Latent Spectrum', 'Skip64 Spectrum']

    # Добавляем подписи к каждому изображению
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)
    thickness = 2
    for i, (img, label) in enumerate(zip(images, row1_labels)):
        cv2.putText(img, label, (10, 36), font, font_scale, font_color, thickness, cv2.LINE_AA)
        images[i] = img

    row1 = np.hstack(images)
    grid = row1

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, grid)


# ---------------- U2Net mask loader ----------------
def load_u2net_mask_batch(fnames, classes, size_512=(512,512)):
    """
    Returns float tensor B×1×H×W in [0,1].
    If file missing, returns zeros (i.e., nothing excluded).
    """
    masks = []
    for f, cls in zip(fnames, classes):
        p = os.path.join(U2NET_ROOT, cls, os.path.splitext(f)[0]+".png")
        if os.path.exists(p):
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, size_512, interpolation=cv2.INTER_NEAREST)
            m = (m.astype(np.float32)/255.0)
        else:
            m = np.zeros(size_512, np.float32)
        masks.append(m[None, ...])  # 1×H×W
    m = np.stack(masks, 0)  # B×1×H×W
    return torch.from_numpy(m)

# ---------------- Maskability map ----------------
def sobel_energy_map(img_bchw):
    """
    img_bchw: torch (B,1,512,512) in [0,1]
    returns: torch (B,1,512,512) in [0,1]
    """
    B, C, H, W = img_bchw.shape
    assert C==1 and H==512 and W==512
    # Sobel kernels
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=img_bchw.device).view(1,1,3,3)/4.0
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=img_bchw.device).view(1,1,3,3)/4.0
    gx = F.conv2d(img_bchw, kx, padding=1)
    gy = F.conv2d(img_bchw, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy)  # [0..~]
    # normalize per-sample
    mag = mag - mag.amin(dim=(2,3), keepdim=True)
    mag = mag / (mag.amax(dim=(2,3), keepdim=True) + 1e-8)
    return mag.clamp(0,1)

def quantile_binary(maskability, go_mask, target_fraction):
    """
    maskability, go_mask: B×1×H×W [0,1]
    target_fraction in [0.05..0.25], keep exact area inside go area.
    returns: allow_bin (B×1×H×W {0/1}), actual_fraction list
    """
    B, _, H, W = maskability.shape
    allow = []
    fracs = []
    m = (go_mask>0.5).float() * maskability  # only where allowed by U2
    for i in range(B):
        v = m[i,0]
        flat = v.flatten()
        pos = flat[flat>0]
        if pos.numel()==0:
            thr = 1.1
            a = (v>=thr).float()
            frac = 0.0
        else:
            K = int(max(1, round(target_fraction * pos.numel())))
            K = min(K, pos.numel())
            # threshold at (1 - target_fraction) quantile
            thr_val = torch.topk(pos, k=K, largest=True).values.min()
            a = (v>=thr_val).float()
            frac = a.sum().item() / (H*W)
        allow.append(a[None,None,...])
        fracs.append(frac)
    allow = torch.cat(allow, 0)
    return allow, fracs

# ---------------- Watermark generators ----------------
import torch

def make_noise_like(x, seed):
    try:
        g = torch.Generator(device=x.device)
        g.manual_seed(int(seed) & 0x7fffffff)
        return torch.randn_like(x, generator=g)
    except TypeError:
        # Старый torch — fallback на numpy
        shape = x.shape
        device = x.device
        dtype = x.dtype
        np.random.seed(int(seed) & 0x7fffffff)
        noise = torch.from_numpy(np.random.randn(*shape).astype(np.float32)).to(device=device, dtype=dtype)
        return noise


def inject_watermark(latents, skip64s, allow_lat, allow_skip, intensity, seed):
    """
    latents: B×C×32×32
    skip64s: B×C2×64×64
    allow_*: B×1×h×w (broadcast)
    """
    wm_lat_noise  = make_noise_like(latents,  seed*7919 + 13)
    wm_skip_noise = make_noise_like(skip64s, seed*104729 + 29)

    wm_lat = intensity * wm_lat_noise * allow_lat
    wm_skip = intensity * wm_skip_noise * allow_skip

    lat_wm = latents + intensity * wm_lat
    #lat_wm  = latents  + wm_lat
    skip_wm = skip64s  + wm_skip
    return lat_wm, skip_wm, wm_lat, wm_skip

# ---------------- Models: load/freeze ----------------
def load_c1_frozen():
    c1 = EfficientNetB3_CBAM_Bottleneck(num_classes=len(CLASSES))  # только num_classes
    # Подмена первого слоя на 1-канальный conv для grayscale
    c1.base.features[0][0] = nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False)
    sd = torch.load(C1_PATH, map_location='cpu')
    # DataParallel compatibility:
    if 'module.' in next(iter(sd.keys())):
        sd = {k.replace('module.',''): v for k,v in sd.items()}
    c1.load_state_dict(sd, strict=False)
    c1.to(DEVICE).eval()
    for p in c1.parameters():
        p.requires_grad=False
    return c1


def load_autoencoder_frozen():
    enc = Encoder().to(DEVICE)
    dec = Decoder().to(DEVICE)
    sd = torch.load(AE_PATH, map_location='cpu')
   #print(list(sd.keys()))

    enc.load_state_dict({k.replace('encoder.', ''): v for k, v in sd.items() if k.startswith('encoder.')})
    dec.load_state_dict({k.replace('decoder.', ''): v for k, v in sd.items() if k.startswith('decoder.')})

    if isinstance(sd, dict) and ('encoder' in sd or 'decoder' in sd):
        enc.load_state_dict(sd['encoder'], strict=False)
        dec.load_state_dict(sd['decoder'], strict=False)
    else:
        # single file with both? adapt to your checkpoint format if needed
        try:
            enc.load_state_dict(sd, strict=False)
        except:
            pass
        try:
            dec.load_state_dict(sd, strict=False)
        except:
            pass
    enc.eval(); dec.eval()
    for p in enc.parameters(): p.requires_grad=False
    for p in dec.parameters(): p.requires_grad=False
    return enc, dec

def build_c2():
    c2 = timm.create_model('efficientnet_b3', pretrained=True,
                           num_classes=len(CLASSES), in_chans=1)
    c2.to(DEVICE)
    return c2

def set_c2_stage_A(c2: nn.Module):
    # freeze all except classifier head
    for p in c2.parameters():
        p.requires_grad=False
    # timm effnet_b3 head is usually "classifier"
    for n,p in c2.named_parameters():
        if 'classifier' in n:
            p.requires_grad=True

def set_c2_stage_B(c2: nn.Module):
    # unfreeze last 2 blocks + head
    for p in c2.parameters():
        p.requires_grad=False
    for n,p in c2.named_parameters():
        if ('blocks.6' in n) or ('blocks.5' in n) or ('conv_head' in n) or ('bn2' in n) or ('classifier' in n):
            p.requires_grad=True

def run_encoder(encoder: nn.Module, img_1x512):
    # adapt to your encoder forward; must return (latents: B×C×32×32, skip64s: B×C2×64×64)
    latents, skip64s = encoder(img_1x512)  # <--- ensure this matches your implementation
    return latents, skip64s

def run_decoder(decoder: nn.Module, latents, skip64s):
    img = decoder(latents, skip64s)        # <--- ensure this matches your implementation
    return img.clamp(0,1)

# ---------------- Losses ----------------
def entropy_from_logits(logits):
    prob = logits.softmax(1) + 1e-12
    return -(prob * prob.log()).sum(1).mean()

# ---------------- Training ----------------
def main():
    ds = MRIDataset(TRAIN_ROOT, CLASSES)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    c1 = load_c1_frozen()
    encoder, decoder = load_autoencoder_frozen()
    c2 = build_c2()

    # Stage A optimizer (head only)
    set_c2_stage_A(c2)
    c2_optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, c2.parameters()), lr=LR_HEAD, weight_decay=1e-4)

    scaler = torch.amp.GradScaler('cuda', enabled=AMP)

    # dynamic knobs
    mask_fraction = MASK_FRAC_INIT
    intensity     = INTENSITY_INIT

    best_margin = -1e9
    best_ckpt   = None
    cycle_id    = 1   # формально «цикл»=1, можно расширить

    dummy_img = torch.zeros(1, 1, 512, 512).to(DEVICE)
    with torch.no_grad():
        dummy_latents, dummy_skip64s = encoder((dummy_img - 0.5) / 0.5)
    latent_channels = dummy_latents.shape[1]
    skip_channels = dummy_skip64s.shape[1]

    patch_lat = nn.Parameter(torch.zeros_like(dummy_latents))  # shape [1, 1024, 32, 32]
    patch_lat = patch_lat.to(DEVICE)

    watermark_gen_cnn = WatermarkGeneratorMiniUNet(
        in_channels=1,  # если хочешь, можно передать 2 или 3: [image, mask, maskability]
        out_channels_latent=latent_channels,
        out_channels_skip=skip_channels
    ).to(DEVICE)

    class_embed_lat = nn.Embedding(len(CLASSES), latent_channels).to(DEVICE)

    gen_params = list(watermark_gen_cnn.parameters()) + list(class_embed_lat.parameters())
    # если используешь ещё trainable patch_lat, добавь его тоже:
    # gen_params += [patch_lat]

    gen_optim = torch.optim.AdamW(gen_params, lr=1e-3)

    for epoch in range(1, EPOCHS+1):
        # stage switch for C2

        c1_gap_cooldown = 0


        # сглаживание по 5–10 батчам
        SMOOTH_N = 8
        acc_c2_clean_hist = deque(maxlen=SMOOTH_N)
        acc_c2_wm_hist = deque(maxlen=SMOOTH_N)

        # cooldown (в батчах) между изменениями
        COOLDOWN_STEPS = 5
        payload_cooldown = 0
        intensity_cooldown = 0

        # шаги корректировок
        MASK_FRAC_STEP = 0.01
        INTENSITY_STEP_SOFT = max(1e-6, INTENSITY_STEP / 2)  # твоя половина шага

        if epoch == (C2_STAGE_A_EPOCHS + 1):
            set_c2_stage_B(c2)
            c2_optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, c2.parameters()),
                                         lr=LR_BACKBONE, weight_decay=1e-4)

        c2.train()
        epoch_rows = []
        epoch_acc_clean_c1 = []
        epoch_acc_wm_c1    = []
        epoch_acc_clean_c2 = []
        epoch_acc_wm_c2    = []

        # per-epoch CSV file
        epoch_csv = os.path.join(CSV_ROOT, f"cycle{cycle_id}_epoch{epoch}.csv")
        if os.path.exists(epoch_csv):
            os.remove(epoch_csv)

        for it, (img_1, ys, fnames, cls_names) in enumerate(loader):
            img_1 = img_1.to(DEVICE, non_blocking=True).float()  # B×1×512×512 in [0,1]
            if not torch.is_tensor(ys):
                ys = torch.tensor(ys, device=DEVICE, dtype=torch.long)
            else:
                ys = ys.to(DEVICE).long()

            # ---- Encoder → latents/skip, Clean & WM decode ----
            with torch.no_grad():
                img_ae = (img_1 - 0.5) / 0.5
                latents, skip64s = run_encoder(encoder, img_ae)
                clean_img = run_decoder(decoder, latents, skip64s)   # B×1×512×512 [0,1]

            # ---- Build masks (U2 + maskability) on 512×512 ----
            u2 = load_u2net_mask_batch(fnames, cls_names, size_512=(512,512)).to(DEVICE).float()  # 1=brain
            go_from_u2 = (1.0 - (u2>0.5).float())  # 1 where allowed
            brain_mask = (u2 > 0.5).float()  # [B,1,512,512]

            maskability = sobel_energy_map(img_1)   # [0,1]
            allow_raw_512 = (go_from_u2 * maskability).clamp(0,1)

            allow_bin_512, frac_list = quantile_binary(maskability, go_from_u2, target_fraction=mask_fraction)

            allow_bin_512 = F.avg_pool2d(allow_bin_512.float(), kernel_size=9, stride=1, padding=4)
            allow_bin_512 = (allow_bin_512 > 0.05).float()

            # Запрет внедрения watermark внутри мозга:

            allow_bin_512 = allow_bin_512 * (1 - brain_mask)  # только по краю, вне мозга
            #adaptive_mask = (maskability > 0.04).float()  # только где есть “шум”
            #allow_bin_512 = allow_bin_512 * adaptive_mask  # только “шумные” зоны

            img1_min = img_1.amin(dim=(2, 3), keepdim=True)
            img1_max = img_1.amax(dim=(2, 3), keepdim=True)
            img1_for_bg = (img_1 - img1_min) / (img1_max - img1_min + 1e-8)

            background_mask = (img1_for_bg < 0.33).float()
            allow_bin_512 = torch.max(allow_bin_512, background_mask)

            payload_pixels = int(allow_bin_512.sum().item())
            payload_pct = 100.0 * payload_pixels / (allow_bin_512.numel())


            # --- per-image payload ---
            # allow_bin_512: (B,1,512,512) из quantile_binary
            payload_px_list = allow_bin_512.flatten(1).sum(dim=1).tolist()  # список px по каждому элементу батча
            payload_pct_list = [round(100.0 * p / (512 * 512), 4) for p in payload_px_list]

            payload_info = {
                "payload_pixels": int(payload_pixels),
                "payload_pct": round(payload_pct, 4),
                "payload_px_list": "|".join(str(int(p)) for p in payload_px_list),
                "payload_pct_list": "|".join(f"{v:.4f}" for v in payload_pct_list),
            }

            # ---- Downsample masks for latent (32) and skip64 (64) ----
            allow_lat  = F.interpolate(allow_bin_512, size=(32,32), mode='nearest')
            allow_skip = F.interpolate(allow_bin_512, size=(64,64), mode='nearest')

            # --- Внедрение watermark в LATENT (Variant B: spatial-domain) ---
            #watermark_lat, watermark_skip = watermark_gen_cnn(img_1, allow_lat, allow_skip, intensity)
            #watermark_lat_masked = watermark_lat  # без маски для чистого теста
            #lat_wm = latents + intensity * watermark_lat_masked  # <-- spatial внедрение

            watermark_lat, watermark_skip = watermark_gen_cnn(img_1, allow_lat, allow_skip, intensity)

            ys = ys.long()  # на всякий случай
            bias_lat = class_embed_lat(ys).view(ys.size(0), latent_channels, 1, 1).expand(-1, -1, 32, 32)

            beta = 0.6  # можно 0.5–1.0
            wm_lat_total = watermark_lat + beta * bias_lat

            # Variant B — spatial внедрение ТОЛЬКО в latent; skip пока без вклада
            lat_wm = latents + intensity * wm_lat_total
            skip_wm = skip64s  # чтобы не мешать диагностике

           # print("watermark_lat mean:", watermark_lat_masked.mean().item(), "max:", watermark_lat_masked.max().item())
            print("wm_lat_total mean:", wm_lat_total.mean().item(), "max:", wm_lat_total.max().item())

            # Корректная визуализация спектра: считаем FFT от фактических тензоров
            latent_freq = fft2(latents)
            latent_freq_wm = fft2(lat_wm)  # <-- от lat_wm, а не «расчётной» формулы
            latent_spec_diff = torch.abs(latent_freq_wm - latent_freq).mean(dim=1, keepdim=True)
            latent_spec_diff_norm = (latent_spec_diff - latent_spec_diff.min()) / (latent_spec_diff.max() - latent_spec_diff.min() + 1e-8)
            latent_spec_diff_up = F.interpolate(latent_spec_diff_norm.detach(), size=(512, 512), mode='bilinear').squeeze().cpu().numpy()

            # --- Внедрение watermark в SKIP64 (64x64x512, частотная область) ---
            #skip64_freq = fft2(skip64s)  # [B,512,64,64]
            #watermark_skip_masked = watermark_skip * allow_skip  # mask из U2Net (downsampled)
            #skip64_freq_wm = skip64_freq + intensity * watermark_skip_masked
            #skip_wm = ifft2(skip64_freq_wm).real

            #skip_freq = fft2(skip64s)
            #skip_freq_wm = fft2(skip_wm)
            #skip_spec_diff = torch.abs(skip_freq_wm - skip_freq).mean(dim=1, keepdim=True)
            #skip_spec_diff_norm = (skip_spec_diff - skip_spec_diff.min()) / (skip_spec_diff.max() - skip_spec_diff.min() + 1e-8)
            #skip_spec_diff_up = F.interpolate(skip_spec_diff_norm.detach(), size=(512, 512), mode='bilinear').squeeze().cpu().numpy()
            # --- Skip64: выключаем для чистого теста Variant B ---


            # (опционально можно рисовать нулевой спектр для коллажа)
            skip_spec_diff_up = np.zeros((img_1.size(0), 512, 512), dtype=np.float32)

            #wm_img = run_decoder(decoder, lat_wm, skip_wm)
            wm_img = run_decoder(decoder, lat_wm, skip64s)

            # ---- Classifier inputs ----

            # изображения от декодера в [0,1]
            wm_img_01 = wm_img.clamp(0, 1)
            clean_img_01 = clean_img.clamp(0, 1)

            # Нормализация для C1/C2
            wm_img_norm_c1 = norm_c1(wm_img_01)
            clean_img_norm_c1 = norm_c1(clean_img_01)

            # Если C2 на той же нормализации:
            wm_img_norm_c2 = wm_img_norm_c1
            clean_img_norm_c2 = clean_img_norm_c1
            # Если хочешь ImageNet-норм для C2 — раскомментируй ниже и замени 2 строки выше:
            # wm_img_norm_c2    = norm_imnet(wm_img_01)
            # clean_img_norm_c2 = norm_imnet(clean_img_01)

            # ---- C1 metrics (frozen) ----
            with torch.no_grad():
                logits_c1_clean = c1(clean_img_norm_c1)
                logits_c1_wm = c1(wm_img_norm_c1)
                acc_c1_clean = (logits_c1_clean.argmax(1) == ys).float().mean().item()
                acc_c1_wm = (logits_c1_wm.argmax(1) == ys).float().mean().item()
                print(f"[C1] clean={acc_c1_clean:.3f} | wm={acc_c1_wm:.3f}")
                epoch_acc_clean_c1.append(acc_c1_clean)
                epoch_acc_wm_c1.append(acc_c1_wm)

            # ---- C2 adversarial loss ----
            with torch.autocast(device_type='cuda', enabled=AMP):
                logits_clean_c2 = c2(clean_img_norm_c2)
                logits_wm_c2 = c2(wm_img_norm_c2)

                # Clean: maximize entropy → minimize negative entropy
                # Если у тебя есть helper entropy_from_logits, оставь его:
                ent_clean = entropy_from_logits(logits_clean_c2)
                loss_clean_c2 = -ent_clean

                # Получи фичи для contrastive loss:
                feat_clean_c2 = c2.forward_features(clean_img_norm_c2)
                feat_wm_c2 = c2.forward_features(wm_img_norm_c2)

                # Обрежь spatial (усредни по H,W) — получится (B, F)
                c2_feat_clean = feat_clean_c2.mean(dim=(-2, -1))  # [B, feat_dim]
                c2_feat_wm = feat_wm_c2.mean(dim=(-2, -1))

                # WM: standard CE
                loss_wm_c2 = F.cross_entropy(logits_wm_c2, ys, label_smoothing=0.1)

                ALPHA_CLEAN_ENTROPY = 1.5

                WM_CE_WEIGHT = 2.0  # если у тебя было 1.5
                adv_c2_loss = ALPHA_CLEAN_ENTROPY * loss_clean_c2 + WM_CE_WEIGHT * loss_wm_c2

                # а чтобы C2-clean сильнее валился:


                # --- Losses ---
                l1_img = (wm_img - clean_img).abs().mean()
                from pytorch_msssim import ssim as torch_ssim
                ssim_loss = 1 - torch_ssim(wm_img, clean_img, data_range=1.0, size_average=True)

                L1_WEIGHT = 0.00
                SSIM_WEIGHT = 0.00
                CONTRASTIVE_WEIGHT = 1.00
                EDGE_WEIGHT = 0.0

                edge_loss = (torch.abs(wm_img - clean_img) * ((maskability < 0.04).float() * (1 - brain_mask))).mean()
                contrastive_loss = 1 - F.cosine_similarity(c2_feat_wm, c2_feat_clean, dim=1).mean()

                total_loss = adv_c2_loss + L1_WEIGHT * l1_img + SSIM_WEIGHT * ssim_loss + EDGE_WEIGHT * edge_loss + CONTRASTIVE_WEIGHT * contrastive_loss

            gen_optim.zero_grad()
            c2_optim.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.step(gen_optim)
            scaler.step(c2_optim)
            scaler.update()

            with torch.no_grad():
                c2_clean_pred = logits_clean_c2.argmax(1)
                c2_wm_pred = logits_wm_c2.argmax(1)
                acc_c2_clean = (c2_clean_pred == ys).float().mean().item()
                acc_c2_wm = (c2_wm_pred == ys).float().mean().item()

                # --- скользящее среднее по окну батчей ---
                acc_c2_clean_hist.append(float(acc_c2_clean))
                acc_c2_wm_hist.append(float(acc_c2_wm))
                sm_acc_c2_clean = sum(acc_c2_clean_hist) / len(acc_c2_clean_hist)
                sm_acc_c2_wm = sum(acc_c2_wm_hist) / len(acc_c2_wm_hist)

                # — опционально: быстрый принт для отладки
                print(f"[C2-smooth] clean={sm_acc_c2_clean:.3f} (inst {acc_c2_clean:.3f}) | "
                      f"wm={sm_acc_c2_wm:.3f} (inst {acc_c2_wm:.3f})")

                # --- авто-тюн payload по сглаженной clean-accuracy (+ гистерезис по WM) ---
                # если C2 слишком «уверен» на clean -> payload ↑, пусть путается
                if sm_acc_c2_clean > CLEAN_HIGH:
                    old_mf = mask_fraction
                    mask_fraction = min(mask_fraction + MF_STEP_UP, MF_MAX)
                    if mask_fraction != old_mf:
                        print(f"[Auto-tune Payload] ↑ INCREASE mask_fraction → {mask_fraction:.3f} "
                              f"(sm_clean {sm_acc_c2_clean:.2f} > {CLEAN_HIGH:.2f})")

                # если C2 и так «падает» на clean -> payload ↓, чтобы не переусердствовать
                elif sm_acc_c2_clean < CLEAN_LOW:
                    old_mf = mask_fraction
                    mask_fraction = max(mask_fraction - MF_STEP_DOWN, MF_MIN)
                    if mask_fraction != old_mf:
                        print(f"[Auto-tune Payload] ↓ DECREASE mask_fraction → {mask_fraction:.3f} "
                              f"(sm_clean {sm_acc_c2_clean:.2f} < {CLEAN_LOW:.2f})")

                # дополнительный «хвост»: если WM слишком низкий — немного подбросим payload
                elif sm_acc_c2_wm < WM_MIN_OK and sm_acc_c2_clean <= CLEAN_HIGH:
                    old_mf = mask_fraction
                    mask_fraction = min(mask_fraction + MF_STEP_UP, MF_MAX)
                    if mask_fraction != old_mf:
                        print(f"[Auto-tune Payload] ↗ BUMP mask_fraction → {mask_fraction:.3f} "
                              f"(sm_wm {sm_acc_c2_wm:.2f} < {WM_MIN_OK:.2f})")

                # записываем в списки эпохи ПОСЛЕ авто-тюна

                # --- мягкий авто-тюн intensity (под теми же сглаженными метриками) ---
                gap_c1 = abs(acc_c1_clean - acc_c1_wm)

                if gap_c1 > C1_GAP_MAX:
                    intensity = max(INTENSITY_MIN, intensity - INTENSITY_STEP)
                    mask_fraction = max(MASK_FRAC_MIN, mask_fraction - MF_STEP_DOWN)
                    c1_gap_cooldown = 5  # freeze на 5 батчей
                    print(f"[C1 GAP CONTROL] GAP={gap_c1:.3f} > {C1_GAP_MAX}, lowering intensity→{intensity:.2f}, payload→{mask_fraction:.2f}")
                    continue

                if c1_gap_cooldown > 0:
                    c1_gap_cooldown -= 1
                    continue  # НЕ повышаем payload/intensity — batch полностью safe

                # дальше обычный auto-tune по C2...

                # 2. Только если всё ок с C1, делаем обычный auto-tune
                # если на WM плохо (ниже порога), а на clean не слишком высоко → слегка повысим
                if sm_acc_c2_wm < WM_MIN_OK and sm_acc_c2_clean <= CLEAN_HIGH:
                    old_int = intensity
                    intensity = min(INTENSITY_MAX, intensity + INTENSITY_STEP * 0.5)
                    if intensity != old_int:
                        print(f"[Auto-tune Intensity] ↗ increase → {intensity:.3f} (sm_wm {sm_acc_c2_wm:.2f} < {WM_MIN_OK:.2f})")

                # если на clean слишком хорошо → слегка понизим
                if sm_acc_c2_clean > CLEAN_HIGH:
                    old_int = intensity
                    intensity = max(INTENSITY_MIN, intensity - INTENSITY_STEP * 0.5)
                    if intensity != old_int:
                        print(f"[Auto-tune Intensity] ↘ decrease → {intensity:.3f} (sm_clean {sm_acc_c2_clean:.2f} > {CLEAN_HIGH:.2f})")

                epoch_acc_clean_c2.append(acc_c2_clean)
                epoch_acc_wm_c2.append(acc_c2_wm)

            # ---- Metrics: L1, SSIM, diff ----
            l1_img = (wm_img - clean_img).abs().mean().item()
            # SSIM per-sample
            ssim_vals = []
            clean_np = clean_img.detach().cpu().numpy()
            wm_np    = wm_img.detach().cpu().numpy()
            for b in range(clean_np.shape[0]):
                ssim_vals.append(ssim(clean_np[b,0], wm_np[b,0], data_range=1.0))
            ssim_mean = float(np.mean(ssim_vals))

            B = img_1.size(0)

            # ---- Save collage for a couple samples ----
            for b in range(min(2, img_1.size(0))):
                allow_bin_512 = F.avg_pool2d(allow_bin_512.float(), kernel_size=5, stride=1, padding=2)
                allow_bin_512 = (allow_bin_512 > 0.05).float()

                raw_np = img_1[b, 0].detach().cpu().numpy()
                cln_np = clean_img[b, 0].detach().cpu().numpy()
                wm_np1 = wm_img[b, 0].detach().cpu().numpy()
                u2_np = u2[b, 0].detach().cpu().numpy()
                allow_np = allow_bin_512[b, 0].detach().cpu().numpy()
                maskab_np = maskability[b, 0].detach().cpu().numpy()
                diff_np = (wm_np1 - cln_np)

                # Передавай по одному изображению спектра!
                lat_spec = latent_spec_diff_up[b] if latent_spec_diff_up.shape[0] == B else latent_spec_diff_up
                skip_spec = skip_spec_diff_up[b] if skip_spec_diff_up.shape[0] == B else skip_spec_diff_up

                save_path = os.path.join(WM_ROOT, f"cycle{cycle_id}", f"epoch{epoch}",
                                         cls_names[b], f"{fnames[b]}_collage.png")

                make_collage_save(
                    save_path, raw_np, cln_np, wm_np1, u2_np, allow_np, maskab_np, diff_np,
                    spectrum_latent_512=lat_spec,
                    spectrum_skip_512=skip_spec,
                )

            # ---- Log row ----
            row = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "cycle": cycle_id, "epoch": epoch, "batch": it + 1,
                "acc_c1_clean": acc_c1_clean, "acc_c1_wm": acc_c1_wm,
                "acc_c2_clean": acc_c2_clean, "acc_c2_wm": acc_c2_wm,
                "loss_c2_clean": float(loss_clean_c2.item()),
                "loss_c2_wm": float(loss_wm_c2.item()),
                "adv_c2_loss": float(adv_c2_loss.item()),
                "l1_img": l1_img, "ssim": ssim_mean,
                "mask_fraction": mask_fraction, "intensity": intensity,
                "frac_list": [round(f, 5) for f in frac_list],
            }

            row.update(payload_info)

            log_row_to_csv(row, epoch_csv)

        # ---- Epoch summary & C2 checkpoint ----
        mean_c1c = float(np.mean(epoch_acc_clean_c1)) if epoch_acc_clean_c1 else float('nan')
        mean_c1w = float(np.mean(epoch_acc_wm_c1))    if epoch_acc_wm_c1    else float('nan')
        mean_c2c = float(np.mean(epoch_acc_clean_c2)) if epoch_acc_clean_c2 else float('nan')
        mean_c2w = float(np.mean(epoch_acc_wm_c2))    if epoch_acc_wm_c2    else float('nan')
        print(f"[EPOCH {epoch}] C1 clean/wm: {mean_c1c:.3f}/{mean_c1w:.3f} | C2 clean/wm: {mean_c2c:.3f}/{mean_c2w:.3f}")
        print(f"[Auto-tune Payload] sm_clean={sm_acc_c2_clean:.2f}, sm_wm={sm_acc_c2_wm:.2f} → mask_fraction={mask_fraction:.3f}")

        # save epoch summary copy
        # (у нас уже есть построчный csv; ниже дублируем имя с «_full» для удобства)
        #save_epoch_summary(rows=[], out_path=epoch_full, extra_meta={
        #    "cycle": cycle_id,
        #    "epoch": epoch,
        #    "mean_acc_c1_clean": mean_c1c,
        #    "mean_acc_c1_wm": mean_c1w,
        #    "mean_acc_c2_clean": mean_c2c,
        #    "mean_acc_c2_wm": mean_c2w,
        #    "mean_payload_pct": float(np.mean([r["payload_pct"] for r in csv.DictReader(open(epoch_csv))])),
        #    "mean_l1_img": float(np.mean([r["l1_img"] for r in csv.DictReader(open(epoch_csv))])),
        #    "mean_ssim": float(np.mean([r["ssim"] for r in csv.DictReader(open(epoch_csv))])),
        #})

        #epoch_full = epoch_csv.replace(".csv", "_full.csv")
        # соберём файл как копию (можно пропустить — построчный уже есть)

        epoch_full = epoch_csv.replace(".csv", "_full.csv")
        rows = []
        with open(epoch_csv, 'r', newline='') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                # приводим числовые поля к float
                for k in (
                        "acc_c1_clean", "acc_c1_wm", "acc_c2_clean", "acc_c2_wm",
                        "loss_c2_clean", "loss_c2_wm", "adv_c2_loss",
                        "l1_img", "ssim", "payload_pixels", "payload_pct",
                        "mask_fraction", "intensity"
                ):
                    if k in r and r[k] != '':
                        try:
                            r[k] = float(r[k])
                        except:
                            pass
                rows.append(r)

        extra = {
            "cycle": cycle_id,
            "epoch": epoch,
            "mean_acc_c1_clean": mean_c1c,
            "mean_acc_c1_wm": mean_c1w,
            "mean_acc_c2_clean": mean_c2c,
            "mean_acc_c2_wm": mean_c2w,
        }

        if rows:
            all_px = []
            all_pct = []
            for r in rows:
                if "payload_px_list" in r and r["payload_px_list"]:
                    all_px.append(r["payload_px_list"])
                if "payload_pct_list" in r and r["payload_pct_list"]:
                    all_pct.append(r["payload_pct_list"])
            extra["payload_px_list"] = "||".join(all_px)
            extra["payload_pct_list"] = "||".join(all_pct)

        save_epoch_summary(rows, out_path=epoch_full, extra_meta=extra)

        # save C2 ckpt
        ckpt_path = os.path.join(C2_SAVE_DIR, f"C2_EFNB3_cycle{cycle_id}_epoch{epoch}.pth")
        torch.save(c2.state_dict(), ckpt_path)

        # track best by (wm_acc - clean_acc)
        margin = mean_c2w - mean_c2c
        if margin > best_margin:
            best_margin = margin
            best_ckpt = os.path.join(C2_SAVE_DIR, f"C2_EFNB3_BEST_cycle{cycle_id}.pth")
            torch.save(c2.state_dict(), best_ckpt)
            print(f"[BEST] Updated best C2 by margin={best_margin:.3f} → {best_ckpt}")

    print("Training complete.")

if __name__ == "__main__":
    main()

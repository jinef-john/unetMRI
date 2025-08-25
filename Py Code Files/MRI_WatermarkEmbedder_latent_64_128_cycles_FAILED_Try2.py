from torchvision.models import efficientnet_b3
import torch.optim as optim
import multiprocessing
from torchvision import transforms
import os

import cv2
from torch.amp import autocast, GradScaler
from MRI_C1_B3_CBAM import CBAM
from MRI_Encoder_Latent_64_128_Train import Decoder
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import os, csv, torch, numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import lpips
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch.nn as nn
import torch.nn.functional as F

# Source paths (update as needed):
DATA_ROOT = r"E:\MRI_LOWMEM\tiny_mri"
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Output folders (as requested):
SAL_ROOT = r"E:\MRI_LOWMEM\saliency_cycles_latent_skip64_skip128"
WM_ROOT  = r"E:\MRI_LOWMEM\watermarked_cycles_latent_skip64_skip128"
CSV_ROOT = r"E:\MRI_LOWMEM\metrics_logs_latent_skip64_skip128"
C1_PATH = r"E:\MRI_LOWMEM\C1-B3-CBAM\MRI-C1EfficientNet_B3_CBAM.pth"

os.makedirs(SAL_ROOT, exist_ok=True)
os.makedirs(WM_ROOT, exist_ok=True)
os.makedirs(CSV_ROOT, exist_ok=True)

# Curriculum config:
CYCLES = 5
EPOCHS_PER_CYCLE = 4
BATCH_SIZE = 1
INTENSITIES = [0.015, 0.02, 0.04, 0.06, 0.08]  # one per cycle
EARLY_STOP_C2_CLEAN = 0.20   # e.g., < 20% on clean triggers early stop
EARLY_STOP_C2_WM    = 0.90   # e.g., > 90% on watermarked triggers early stop

# Model config (adjust to match your setup):
PAYLOAD_DIM, SEED_DIM = 128, 16
LATENT_SHAPE, SKIP_SHAPE = (1024, 32, 32), (512, 64, 64)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EfficientNetB3_CBAM_Bottleneck(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.base = efficientnet_b3(weights=None, num_classes=num_classes)
        # === Патчим первый слой на grayscale (1 канал) ===
        old_conv = self.base.features[0][0]
        new_conv = nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False)
        # Если есть веса RGB — усредняем по каналам (рекомендуется при дообучении)
        if hasattr(old_conv, "weight") and old_conv.weight.shape[1] == 3:
            with torch.no_grad():
                gray_w = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(gray_w)
        self.base.features[0][0] = new_conv
        # === Теперь строим features1/features2 на основе уже патченной base ===
        self.features1 = nn.Sequential(*list(self.base.features.children())[:6])
        self.cbam = CBAM(136)
        self.features2 = nn.Sequential(*list(self.base.features.children())[6:])
        self.avgpool = self.base.avgpool
        self.classifier = self.base.classifier

        # DEBUG: убедись что оба слоя действительно на 1 канал
        print("base.features[0][0]:", self.base.features[0][0])
        print("features1[0][0]:", self.features1[0][0])


    def forward(self, x):
        x = self.features1(x)
        x = self.cbam(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class WatermarkDataset(Dataset):
    def __init__(self, root, classes):
        self.classes = classes
        self.samples = []
        for cname in classes:
            classdir = os.path.join(root, cname)
            for fname in os.listdir(classdir):
                if fname.endswith('.png') or fname.endswith('.jpg'):
                    self.samples.append((os.path.join(classdir, fname), cname))
    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, cname = self.samples[idx]
        # --- IMAGE ---
        img = Image.open(img_path).convert('L')
        img = img.resize((512, 512), Image.BICUBIC)
        img = np.array(img).astype(np.float32) / 255.0
        img = img[None, ...]  # [1, 512, 512]

        # --- NPZ ---
        latent_path = img_path.replace("tiny_mri", "MRI-NPZ_latent_skip64_skip128").replace(".jpg", ".npz").replace(
            ".png", ".npz")
        npz = np.load(latent_path)
        latent = torch.tensor(npz['latent']).float().unsqueeze(0)
        latent = F.interpolate(latent, size=(32, 32), mode='bilinear', align_corners=False)[0]
        skip64 = torch.tensor(npz['skip64']).float().unsqueeze(0)
        skip64 = F.interpolate(skip64, size=(64, 64), mode='bilinear', align_corners=False)[0]
        skip128 = torch.tensor(npz['skip128']).float().unsqueeze(0)
        skip128 = F.interpolate(skip128, size=(128, 128), mode='bilinear', align_corners=False)[0]

        # --- DEBUG ASSERTS ---
        assert img.shape == (1, 512, 512), f"Bad img shape: {img.shape}, {img_path}"
        assert latent.shape == (1024, 32, 32), f"Bad latent shape: {latent.shape}, {latent_path}"
        assert skip64.shape == (512, 64, 64), f"Bad skip64 shape: {skip64.shape}, {latent_path}"
        assert skip128.shape == (256, 128, 128), f"Bad skip128 shape: {skip128.shape}, {latent_path}"

        return img, latent, skip64, skip128, CLASSES.index(cname), os.path.basename(img_path)


# === Watermark NN Generator ===
class ContextAwareWatermarkGen(nn.Module):
    def __init__(self, payload_dim=128, seed_dim=16, latent_shape=(1024, 32, 32), skip_shape=(512, 64, 64)):
        super().__init__()
        self.payload_proj = nn.Linear(payload_dim + seed_dim, np.prod(latent_shape))

        self.fusion = nn.Sequential(
            nn.Conv2d(latent_shape[0] * 2, latent_shape[0], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_shape[0], latent_shape[0], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_shape[0], latent_shape[0], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_shape[0], latent_shape[0], 3, padding=1),
            nn.Tanh()
        )
        self.latent_shape = latent_shape

    def forward(self, payload, random_seed, x, _=None):
        device = next(self.parameters()).device  # Всегда device самой модели, а не только x!
        B = x.shape[0]
        # Переносим ВСЁ на device слоя
        payload = payload.to(device)
        random_seed = random_seed.to(device)
        x = x.to(device)
        full_payload = torch.cat([payload, random_seed], dim=1)
        # debug
        #print(f"full_payload: {full_payload.device}, Linear weight: {self.payload_proj.weight.device}, x: {x.device}")
        payload_feat = self.payload_proj(full_payload).view(B, *self.latent_shape)
        fused = torch.cat([x, payload_feat], dim=1)
        return self.fusion(fused)


class CrossLayerMaskHead(nn.Module):
    def __init__(self, ch_latent, ch_skip64, ch_skip128, out_size, p_dropout=0.1):
        super().__init__()
        total_ch = ch_latent + ch_skip64 + ch_skip128
        self.out_size = out_size
        self.layers = nn.Sequential(
            nn.Conv2d(total_ch, total_ch // 2, 3, padding=1),
            nn.BatchNorm2d(total_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p_dropout),
            nn.Conv2d(total_ch // 2, 1, 1),
        )
        nn.init.constant_(self.layers[-1].bias, -0.5)

    def forward(self, latent, skip64, skip128):
        device = next(self.parameters()).device
        latent = latent.to(device)
        skip64 = skip64.to(device)
        skip128 = skip128.to(device)
        latent_rs = F.interpolate(latent, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
        skip64_rs = F.interpolate(skip64, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
        skip128_rs = F.interpolate(skip128, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
        x = torch.cat([latent_rs, skip64_rs, skip128_rs], dim=1)
        return torch.sigmoid(self.layers(x))


def pick_free_gpu():
    import subprocess
    try:
        result = subprocess.check_output(
            'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits',
            shell=True)
        memory_free = [int(x) for x in result.decode('utf-8').strip().split('\n')]
        idx = int(np.argmax(memory_free))
        print(f"[INFO] Using GPU:{idx} (free mem: {memory_free[idx]} MB)")
        print(f"Device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
        print(f"Allocated memory: {torch.cuda.memory_allocated() // 1024 ** 2} MB")
        print(f"Reserved memory: {torch.cuda.memory_reserved() // 1024 ** 2} MB")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(idx)
    except Exception as e:
        print(f"[WARN] Could not automatically select GPU: {e}")
        pass


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


def main():


    def clean_state_dict(state_dict):
        # Для загрузки моделей, обученных с DataParallel
        if any(k.startswith('module.') for k in state_dict.keys()):
            from collections import OrderedDict
            return OrderedDict((k.replace('module.', '', 1), v) for k, v in state_dict.items())
        return state_dict

    # --- Models: Use your actual model classes here ---

    wm_gen_latent = ContextAwareWatermarkGen(PAYLOAD_DIM, SEED_DIM, LATENT_SHAPE, SKIP_SHAPE).to(device)
    wm_gen_skip64 = ContextAwareWatermarkGen(PAYLOAD_DIM, SEED_DIM, (512, 64, 64), (512, 64, 64)).to(device)
    wm_gen_skip128 = ContextAwareWatermarkGen(PAYLOAD_DIM, SEED_DIM, (256, 128, 128), (256, 128, 128)).to(device)

    c2 = EfficientNetB3_CBAM_Bottleneck(num_classes=4).to(device)

    state_dict = torch.load(r'E:\MRI_LOWMEM\Encoder_latent_64_128\autoencoder_epoch9.pth', map_location=device)
    # Извлечь только параметры decoder, убрать префикс
    #decoder_state = {k.replace('module.decoder.', ''): v for k, v in state_dict.items() if
                     #k.startswith('module.decoder.')} #use with nn.dataparralel
    #print("ALL KEYS IN STATE_DICT:", list(state_dict.keys()))  # Тут должны быть deconv1.weight, deconv2.weight и т.д.
    decoder_state = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}

    decoder = Decoder().to(device)
    decoder.load_state_dict(decoder_state)
    decoder.eval()

    # Проверка параметров decoder на nan/inf
    for n, p in decoder.named_parameters():
        if torch.isnan(p).any():
            print(f"[NAN WARNING] In decoder param: {n}")
        if torch.isinf(p).any():
            print(f"[INF WARNING] In decoder param: {n}")
    print("Parameter check completed.")

    optimizer_c2 = torch.optim.Adam(c2.parameters(), lr=1e-4)

    scaler = GradScaler()

    c1 = EfficientNetB3_CBAM_Bottleneck(num_classes=4)
    state_dict_c1 = torch.load(C1_PATH, map_location='cpu')
    state_dict_c1 = clean_state_dict(state_dict_c1)
    c1.load_state_dict(state_dict_c1)
    c1 = nn.DataParallel(c1).to(device)
    for p in c1.parameters():
        p.requires_grad = False
    c1.eval()

    mask_latent_head = CrossLayerMaskHead(
        ch_latent=1024, ch_skip64=512, ch_skip128=256, out_size=32, p_dropout=0.1
    ).to(device)
    mask_skip64_head = CrossLayerMaskHead(
        ch_latent=1024, ch_skip64=512, ch_skip128=256, out_size=64, p_dropout=0.1
    ).to(device)
    mask_skip128_head = CrossLayerMaskHead(
        ch_latent=1024, ch_skip64=512, ch_skip128=256, out_size=128, p_dropout=0.1
    ).to(device)

    print("Initial bias (latent mask):", mask_latent_head.layers[-1].bias.data)
    print("Initial bias (skip64 mask):", mask_skip64_head.layers[-1].bias.data)
    print("Initial bias (skip128 mask):", mask_skip128_head.layers[-1].bias.data)

    optimizer_gen = torch.optim.Adam([
        {'params': wm_gen_latent.parameters()},
        {'params': wm_gen_skip64.parameters()},
        {'params': wm_gen_skip128.parameters()},
        {'params': mask_latent_head.parameters(), 'lr': 3e-4},
        {'params': mask_skip64_head.parameters(), 'lr': 3e-4},
        {'params': mask_skip128_head.parameters(), 'lr': 3e-4},
    ], lr=1e-4)

    dataset = WatermarkDataset(DATA_ROOT, CLASSES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"[DIAG] Dataset size: {len(dataset)}")

    def entropy_loss(logits):
        prob = torch.softmax(logits, dim=1)
        log_prob = torch.log_softmax(logits, dim=1)
        entropy = - (prob * log_prob).sum(dim=1)
        return entropy.mean()


    def save_metrics_csv(metrics, filename):
        keys = sorted(metrics[0].keys())
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, keys)
            writer.writeheader()
            for row in metrics:
                writer.writerow(row)


    # Проверка параметров watermark generator (latent)
    for n, p in wm_gen_latent.named_parameters():
        if torch.isnan(p).any():
            print(f"[NAN WARNING] In wm_gen_latent param: {n}")
        if torch.isinf(p).any():
            print(f"[INF WARNING] In wm_gen_latent param: {n}")

    # Проверка параметров watermark generator (skip64)
    for n, p in wm_gen_skip64.named_parameters():
        if torch.isnan(p).any():
            print(f"[NAN WARNING] In wm_gen_skip64 param: {n}")
        if torch.isinf(p).any():
            print(f"[INF WARNING] In wm_gen_skip64 param: {n}")

    print("Watermark generator check completed .")

    # Проверка параметров mask_latent_head
    for n, p in mask_latent_head.named_parameters():
        if torch.isnan(p).any():
            print(f"[NAN WARNING] In mask_latent_head param: {n}")
        if torch.isinf(p).any():
            print(f"[INF WARNING] In mask_latent_head param: {n}")

    # Проверка параметров mask_skip64_head
    for n, p in mask_skip64_head.named_parameters():
        if torch.isnan(p).any():
            print(f"[NAN WARNING] In mask_skip64_head param: {n}")
        if torch.isinf(p).any():
            print(f"[INF WARNING] In mask_skip64_head param: {n}")

    print("Mask head check completed.")

    # Curriculum training loop
    for cycle in range(CYCLES):
        intensity = INTENSITIES[cycle]
        early_stop = False
        print(f"\n=== Cycle {cycle+1}/{CYCLES} | Intensity: {intensity} ===")
        cycle_metrics = []

        for epoch in range(EPOCHS_PER_CYCLE):
            print(f"  Epoch {epoch+1}/{EPOCHS_PER_CYCLE}")
            # Reset DataLoader iterator for full randomization
            epoch_metrics = []
            for batch in loader:
                try:
                    imgs, latents, skip64s, skip128s, ys, fnames = batch
                    imgs = imgs.to(device, non_blocking=True)
                    latents = latents.to(device, non_blocking=True)
                    skip64s = skip64s.to(device, non_blocking=True)
                    ys = ys.to(device, non_blocking=True)
                    B = imgs.shape[0]

                    print(f"[DIAG] New batch loaded!")
                    print(f"[DIAG] Batch fnames: {fnames}")
                    print(f"[DIAG] Batch size B: {B}")

                    payload = torch.randint(0, 2, (B, PAYLOAD_DIM), dtype=torch.float32, device=device)
                    random_seed = torch.randn(B, SEED_DIM, device=device)

                    imgs.requires_grad = True
                    # Compute C2 saliency mask
                    with torch.enable_grad():
                        out = c2(imgs.float())
                        class_idx = ys
                        score = out[range(B), class_idx].sum()
                        score.backward()
                        saliency = imgs.grad.abs().mean(dim=1)
                    saliency_norm = (saliency - saliency.view(B, -1).min(dim=1)[0][:, None, None]) / \
                                    (saliency.view(B, -1).max(dim=1)[0][:, None, None] -
                                     saliency.view(B, -1).min(dim=1)[0][:, None, None] + 1e-8)
                    threshold = torch.quantile(saliency_norm.view(B, -1), 0.9, dim=1, keepdim=True).unsqueeze(-1)
                    go_mask = (saliency_norm < threshold).float()

                    # Watermarking (mixed precision)
                    with autocast(device_type='cuda'):
                        watermark_latent = wm_gen_latent(payload, random_seed, latents, None)
                        watermark_skip64 = wm_gen_skip64(payload, random_seed, skip64s, None)
                        watermark_skip128 = wm_gen_skip128(payload, random_seed, skip128s, None)
                        go_mask_latent = mask_latent_head(latents, skip64s, skip128s)
                        go_mask_skip64 = mask_skip64_head(latents, skip64s, skip128s)
                        go_mask_skip128 = mask_skip128_head(latents, skip64s, skip128s)
                        mask_l2_penalty_latent = (go_mask_latent ** 2).mean()
                        mask_l2_penalty_skip64 = (go_mask_skip64 ** 2).mean()
                        mask_l2_penalty_skip128 = (go_mask_skip128 ** 2).mean()
                        beta_l2 = 0.05
                        entropy_loss_latent = -(go_mask_latent * torch.log(go_mask_latent + 1e-8) +
                                                (1 - go_mask_latent) * torch.log(1 - go_mask_latent + 1e-8)).mean()
                        entropy_loss_skip64 = -(go_mask_skip64 * torch.log(go_mask_skip64 + 1e-8) +
                                                (1 - go_mask_skip64) * torch.log(1 - go_mask_skip64 + 1e-8)).mean()
                        entropy_loss_skip128 = -(go_mask_skip128 * torch.log(go_mask_skip128 + 1e-8) +
                                                 (1 - go_mask_skip128) * torch.log(1 - go_mask_skip128 + 1e-8)).mean()
                        sharpness_weight = 0.15
                        go_mask_latent = go_mask_latent.clamp(0, 1)
                        go_mask_skip64 = go_mask_skip64.clamp(0, 1)
                        go_mask_skip128 = go_mask_skip128.clamp(0, 1)
                        mask_latent_effective = go_mask_latent
                        latents = latents.to(device)
                        watermark_latent = watermark_latent.to(device)
                        mask_latent_effective = mask_latent_effective.to(device)
                        watermarked_latent = latents + intensity * watermark_latent * mask_latent_effective
                        latent_up = F.interpolate(mask_latent_effective, size=(64, 64), mode='bilinear',
                                                  align_corners=False)
                        mask_skip64_effective = go_mask_skip64 * (1 - latent_up)
                        skip64s = skip64s.to(device)
                        watermark_skip64 = watermark_skip64.to(device)
                        mask_skip64_effective = mask_skip64_effective.to(device)
                        watermarked_skip64 = skip64s + intensity * watermark_skip64 * mask_skip64_effective
                        latent_up128 = F.interpolate(mask_latent_effective, size=(128, 128), mode='bilinear',
                                                     align_corners=False)
                        skip64_up128 = F.interpolate(mask_skip64_effective, size=(128, 128), mode='bilinear',
                                                     align_corners=False)
                        mask_skip128_effective = go_mask_skip128 * (1 - latent_up128) * (1 - skip64_up128)
                        skip128s = skip128s.to(device)
                        watermark_skip128 = watermark_skip128.to(device)
                        mask_skip128_effective = mask_skip128_effective.to(device)
                        watermarked_skip128 = skip128s + intensity * watermark_skip128 * mask_skip128_effective
                        watermarked_latent = watermarked_latent.clamp(0, 1)
                        watermarked_skip64 = watermarked_skip64.clamp(0, 1)
                        watermarked_skip128 = watermarked_skip128.clamp(0, 1)
                        wm_img = decoder(watermarked_latent, watermarked_skip64, watermarked_skip128)
                        clean_img = decoder(latents, skip64s, skip128s)
                        norm = lambda x: (x - 0.5) / 0.5
                        wm_img_norm = norm(wm_img).to(torch.float32)
                        clean_img_norm = norm(clean_img).to(torch.float32)

                    with torch.no_grad():
                        c1_clean_pred = c1(clean_img_norm.float()).argmax(1)
                        c1_wm_pred = c1(wm_img_norm.float()).argmax(1)
                        acc_c1_clean = (c1_clean_pred == ys).float().mean().item()
                        acc_c1_wm = (c1_wm_pred == ys).float().mean().item()
                        c2_clean_pred = c2(clean_img_norm.float()).argmax(1)
                        c2_wm_pred = c2(wm_img_norm.float()).argmax(1)
                        acc_c2_clean = (c2_clean_pred == ys).float().mean().item()
                        acc_c2_wm = (c2_wm_pred == ys).float().mean().item()
                        class_accs_clean, class_accs_wm = [], []
                        for c in range(len(CLASSES)):
                            idx = (ys == c)
                            if idx.sum() > 0:
                                acc_clean = (c2_clean_pred[idx] == ys[idx]).float().mean().item()
                                acc_wm = (c2_wm_pred[idx] == ys[idx]).float().mean().item()
                            else:
                                acc_clean = acc_wm = float('nan')
                            class_accs_clean.append(acc_clean)
                            class_accs_wm.append(acc_wm)
                        min_clean, max_clean, mean_clean = np.nanmin(class_accs_clean), np.nanmax(
                            class_accs_clean), np.nanmean(class_accs_clean)
                        min_wm, max_wm, mean_wm = np.nanmin(class_accs_wm), np.nanmax(class_accs_wm), np.nanmean(
                            class_accs_wm)

                    logits = c2(wm_img_norm)
                    if torch.isnan(logits).any():
                        print("NAN in C2 logits! Skipping this batch.")
                        continue

                    loss_wm = torch.nn.functional.cross_entropy(logits, ys)
                    loss_clean = -entropy_loss(c2(clean_img_norm))
                    energy_loss_latent = (watermark_latent.abs() * go_mask_latent).mean()
                    energy_loss_skip64 = (watermark_skip64.abs() * go_mask_skip64).mean()
                    energy_loss_skip128 = (watermark_skip128.abs() * go_mask_skip128).mean()
                    mask_sparsity = go_mask_latent.mean()
                    mask_sparsity_skip64 = go_mask_skip64.mean()
                    mask_sparsity_skip128 = go_mask_skip128.mean()
                    sparsity_weight = 0.15
                    sparsity_weight_skip64 = 0.005
                    sparsity_weight_skip128 = 0.005
                    l1_img = (wm_img - clean_img).abs().mean()
                    alpha_l1 = 0.075
                    drop = acc_c1_clean - acc_c1_wm
                    penalty_c1 = torch.tensor(0.0, device=device)
                    if acc_c1_wm < 0.97:
                        penalty_c1 = torch.tensor(abs(acc_c1_clean - acc_c1_wm) * 40.0, device=device)
                    total_loss = (
                            loss_wm + loss_clean
                            + 0.01 * energy_loss_latent
                            + 0.005 * energy_loss_skip64
                            + 0.002 * energy_loss_skip128
                            + sparsity_weight * mask_sparsity
                            + sparsity_weight_skip64 * mask_sparsity_skip64
                            + sparsity_weight_skip128 * mask_sparsity_skip128
                            + alpha_l1 * l1_img
                            + penalty_c1
                            + beta_l2 * mask_l2_penalty_latent
                            + beta_l2 * mask_l2_penalty_skip64
                            + beta_l2 * mask_l2_penalty_skip128
                            + sharpness_weight * entropy_loss_latent
                            + sharpness_weight * entropy_loss_skip64
                            + sharpness_weight * entropy_loss_skip128
                    )

                    nan_report = lambda name, tensor: print(
                        f"[NAN WARNING] {name} contains nan!" if torch.isnan(tensor).any()
                        else f"[OK] {name} min: {tensor.min().item()} max: {tensor.max().item()} mean: {tensor.mean().item()}"
                    )
                    nan_report("total_loss", total_loss)
                    nan_report("loss_wm", loss_wm)
                    nan_report("loss_clean", loss_clean)
                    nan_report("mask_sparsity", mask_sparsity)

                    # === MAIN backward and optimizer section with full error tracking ===
                    try:
                        with torch.autograd.set_detect_anomaly(True):
                            scaler.scale(total_loss).backward()
                    except Exception as e:
                        print(f"[ERROR] Exception in backward: {e}")
                        continue  # skip optimizer and collage if backward failed

                    try:
                        print("Running scalers and optimizers")
                        scaler.step(optimizer_c2)
                        scaler.step(optimizer_gen)
                        scaler.update()
                        optimizer_c2.zero_grad()
                        optimizer_gen.zero_grad()
                    except Exception as e:
                        print(f"[ERROR] Exception in optimizer step: {e}")
                        continue

                    # ========== COLLEGE CREATION BLOCK ==========
                    B = imgs.shape[0]
                    print(f"[DIAG] Entering collage save loop after init B, B = {B}")

                    for i in range(B):
                        print(f"[DIAG] Trying to save collage for image {i}, fname: {fnames[i]}")
                        try:
                            orig_img = imgs[i].detach().cpu().numpy()[0]
                            orig_img = (orig_img * 255).clip(0, 255).astype(np.uint8)
                            orig_img_rgb = np.stack([orig_img] * 3, axis=-1)
                            orig_img_rgb = cv2.resize(orig_img_rgb, (512, 512), interpolation=cv2.INTER_CUBIC)
                            orig_img_labeled = draw_label(orig_img_rgb, "Original")

                            wm_img_np = wm_img[i].detach().cpu().numpy()[0]
                            wm_img_np = (wm_img_np * 255).clip(0, 255).astype(np.uint8)
                            wm_img_rgb = np.stack([wm_img_np] * 3, axis=-1)
                            wm_img_rgb = cv2.resize(wm_img_rgb, (512, 512), interpolation=cv2.INTER_CUBIC)
                            wm_img_labeled = draw_label(wm_img_rgb, "Watermarked")

                            mask_latent_eff = F.interpolate(mask_latent_effective, size=(512, 512),
                                                            mode='bilinear').detach().cpu().numpy()
                            mask_skip64_eff = F.interpolate(mask_skip64_effective, size=(512, 512),
                                                            mode='bilinear').detach().cpu().numpy()
                            mask_skip128_eff = F.interpolate(mask_skip128_effective, size=(512, 512),
                                                             mode='bilinear').detach().cpu().numpy()
                            allowed_mask = \
                            ((mask_latent_eff > 0.5) | (mask_skip64_eff > 0.5) | (mask_skip128_eff > 0.5))[0]
                            allowed_mask_3c = np.repeat(allowed_mask[:, :, np.newaxis], 3, axis=2)
                            allowed_overlay = np.where(allowed_mask_3c, [255, 255, 255], wm_img_np)
                            allowed_overlay_labeled = draw_label(allowed_overlay, "Allowed Embedding Zones")

                            def mask_overlay(src_img, mask, label):
                                mask = np.nan_to_num(mask)
                                mask_norm = (mask - mask.min()) / (np.ptp(mask) + 1e-8)
                                color_overlay = (cm.jet(mask_norm)[..., :3] * 255).astype(np.uint8)
                                result = cv2.addWeighted(src_img, 0.7, color_overlay, 0.3, 0)
                                return draw_label(result, label)

                            overlay_latent = mask_overlay(orig_img, mask_latent_eff[0], "Latent Mask Overlay")
                            overlay_skip64 = mask_overlay(orig_img, mask_skip64_eff[0], "Skip64 Mask Overlay")
                            overlay_skip128 = mask_overlay(orig_img, mask_skip128_eff[0], "Skip128 Mask Overlay")

                            def wm_heatmap(wm_tensor, label):
                                arr = wm_tensor[i].detach().cpu().numpy()
                                arr_map = np.abs(arr).mean(axis=0)
                                arr_map = np.nan_to_num(arr_map)
                                arr_map = cv2.resize(arr_map, (512, 512), interpolation=cv2.INTER_CUBIC)
                                arr_norm = (arr_map - arr_map.min()) / (np.ptp(arr_map) + 1e-8)
                                heat = (cm.jet(arr_norm)[..., :3] * 255).astype(np.uint8)
                                return draw_label(heat, label)

                            heatmap_latent = wm_heatmap(watermark_latent, "Latent Heatmap")
                            heatmap_skip64 = wm_heatmap(watermark_skip64, "Skip64 Heatmap")
                            heatmap_skip128 = wm_heatmap(watermark_skip128, "Skip128 Heatmap")

                            collage = np.concatenate([
                                orig_img_labeled,
                                wm_img_labeled,
                                allowed_overlay_labeled,
                                overlay_latent,
                                overlay_skip64,
                                overlay_skip128,
                                heatmap_latent,
                                heatmap_skip64,
                                heatmap_skip128
                            ], axis=1)

                            cname = dataset.classes[ys[i].item()] if hasattr(dataset, 'classes') else CLASSES[
                                ys[i].item()]
                            epoch_dir = os.path.join(WM_ROOT, f"cycle{cycle + 1}", f"epoch{epoch + 1}", cname)
                            os.makedirs(epoch_dir, exist_ok=True)
                            collage_path = os.path.join(epoch_dir, f"{fnames[i]}_collage.png")
                            Image.fromarray(collage).save(collage_path)
                            print(f"[OK] Saved collage: {collage_path}")

                        except Exception as e:
                            print(f"[ERROR] Failed to save collage for image {i} ({fnames[i]}): {e}")
                            continue

                except Exception as e:
                    print(f"[BATCH FAIL] Unexpected error in batch: {e}")
                    continue

                print("loss_wm requires_grad:", loss_wm.requires_grad)
                print("total_loss requires_grad:", total_loss.requires_grad)
                print("loss_wm grad_fn:", loss_wm.grad_fn)

                # Логирование средних значений масок
                print(f"[Log] Mean go_mask_latent:   {go_mask_latent.mean().item():.5f}")
                print(f"[Log] Mean go_mask_skip64:   {go_mask_skip64.mean().item():.5f}")

                # Логирование penalty C1 и sparsity для каждой маски
                print(f"[Log] Penalty C1:            {penalty_c1.item():.5f}")
                print(f"[Log] Sparsity latent:       {mask_sparsity.item():.5f}")
                print(f"[Log] Sparsity skip64:       {mask_sparsity_skip64.item():.5f}")

                # Логирование loss'ов от C2
                print(f"[Log] loss_wm (C2 on wm):    {loss_wm.item():.5f}")
                print(f"[Log] loss_clean (C2 on cl): {loss_clean.item():.5f}")



                # Логирование L1 между wm_img и clean_img
                print(f"[Log] L1 diff (wm-clean img): {l1_img.item():.5f}")

                # Логирование нормы градиента mask head (latent)
                #if mask_latent_head.conv.weight.grad is not None:
                #    print(
                #        f"[Log] mask_latent_head grad norm: {mask_latent_head.conv.weight.grad.norm().item():.5f}")
                #else:
                #    print("[Log] mask_latent_head grad: None")

                # Логирование нормы градиента mask head (skip64)
                #if mask_skip64_head.conv.weight.grad is not None:
                #    print(
                #        f"[Log] mask_skip64_head grad norm: {mask_skip64_head.conv.weight.grad.norm().item():.5f}")
                #else:
                #    print("[Log] mask_skip64_head grad: None")

        # -- Save CSV metrics per epoch
        cycle_metrics.extend(epoch_metrics)
        csv_filename = os.path.join(CSV_ROOT, f"metrics_cycle{cycle+1}_epoch{epoch+1}.csv")
        save_metrics_csv(epoch_metrics, csv_filename)
        print(
            f"Cycle {cycle + 1} Epoch {epoch + 1} | "
            f"C2 clean acc: {mean_clean:.3f} | C2 wm acc: {mean_wm:.3f} | "
            f"C1 clean acc: {acc_c1_clean:.3f} | C1 wm acc: {acc_c1_wm:.3f} | "
            f"L1 img: {l1_img.item():.5f} | "
            f"Mask sparsity latent: {mask_sparsity.item():.5f} | "
            f"Mask sparsity skip64: {mask_sparsity_skip64.item():.5f} | "
            f"Latent Energy Loss: {energy_loss_latent.item():.5f} | "
            f"Skip64 Energy Loss: {energy_loss_skip64.item():.5f}"
        )

        # -- Early stopping: if all classes pass thresholds, break
        if (mean_clean < EARLY_STOP_C2_CLEAN) and (mean_wm > EARLY_STOP_C2_WM):
            print(f"Early stopping triggered for cycle {cycle+1} at epoch {epoch+1}!")
            early_stop = True
            break

    # -- Save cycle metrics at end of cycle
    cycle_csv_filename = os.path.join(CSV_ROOT, f"metrics_cycle{cycle+1}_full.csv")
    save_metrics_csv(cycle_metrics, cycle_csv_filename)


print("Training complete and all results saved.")

if __name__ == "__main__":
    pick_free_gpu()
   # lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    main()

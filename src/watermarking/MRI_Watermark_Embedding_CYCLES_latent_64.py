from torchvision.models import efficientnet_b3
import torch.optim as optim
import multiprocessing
from torchvision import transforms
import os

import cv2
from torch.amp import autocast, GradScaler
from MRI_C1_B3_CBAM import CBAM
#from MRI_Encoder_Latent_64_128_Train import Decoder
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


# Get project root directory (three levels up from current file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Source paths (update as needed):
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset", "brain-tumor-mri-dataset", "Training")
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Output folders (as requested):
SAL_ROOT = os.path.join(PROJECT_ROOT, "output", "saliency_cycles_latent_skip64_skip128")
WM_ROOT  = os.path.join(PROJECT_ROOT, "output", "watermarked_cycles_latent_skip64_skip128")
CSV_ROOT = os.path.join(PROJECT_ROOT, "output", "metrics_logs_latent_skip64_skip128")
C1_PATH = os.path.join(PROJECT_ROOT, "pt models", "MRI-C1EfficientNet_B3_CBAM.pth")

os.makedirs(SAL_ROOT, exist_ok=True)
os.makedirs(WM_ROOT, exist_ok=True)
os.makedirs(CSV_ROOT, exist_ok=True)

# Curriculum config:
CYCLES = 5
EPOCHS_PER_CYCLE = 5
BATCH_SIZE = 1
INTENSITIES = [0.015, 0.02, 0.04, 0.06, 0.08]  # one per cycle
EARLY_STOP_C2_CLEAN = 0.20   # e.g., < 20% on clean triggers early stop
EARLY_STOP_C2_WM    = 0.90   # e.g., > 90% on watermarked triggers early stop

# Model config (adjust to match your setup):
PAYLOAD_DIM, SEED_DIM = 128, 16
LATENT_SHAPE, SKIP_SHAPE = (1024, 32, 32), (512, 64, 64)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, bottleneck_channels=1024):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(bottleneck_channels, 512, 4, 2, 1)      # 32 -> 64
        self.deconv2 = nn.ConvTranspose2d(512+512, 256, 4, 2, 1)                  # 64 -> 128 (+skip64)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)                      # 128 -> 256 (без skip128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)                       # 256 -> 512
        self.deconv5 = nn.Conv2d(64, 1, 3, 1, 1)
        self.relu = nn.ReLU(inplace=False)
        self.tanh = nn.Tanh()

    def forward(self, latent, skip64):
        x = self.relu(self.deconv1(latent))         # [B, 512, 64, 64]
        x = torch.cat([x, skip64], dim=1)           # [B, 1024, 64, 64]
        x = self.relu(self.deconv2(x))              # [B, 256, 128, 128]
        x = self.relu(self.deconv3(x))              # [B, 128, 256, 256]
        x = self.relu(self.deconv4(x))              # [B, 64, 512, 512]
        x = self.tanh(self.deconv5(x))              # [B, 1, 512, 512]
        return (x + 1) / 2   # [-1,1] → [0,1]



class EfficientNetB3_CBAM_Bottleneck(nn.Module):
    def __init__(self, num_classes=4):
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
        img = Image.open(img_path).convert('L')
        img = img.resize((512, 512), resample=Image.BICUBIC)
        img = np.array(img).astype(np.float32) / 255.0  # [H, W]
        img = np.expand_dims(img, axis=0)  # [1, 512, 512]

        latent_path = img_path.replace("tiny_mri", "MRI-NPZ_latent_skip64_skip128").replace(".png", ".npz").replace(
            ".jpg", ".npz")
        npz = np.load(latent_path)
        latent = npz['latent']  # должен быть [1024, 32, 32]
        skip64 = npz['skip64']  # должен быть [512, 64, 64]

        # Проверяем именно на эти размеры!
        if img.shape != (1, 512, 512):
            print(f"[IMG BAD SHAPE] {img_path}: {img.shape}")
            return self.__getitem__((idx + 1) % len(self.samples))
        if latent.shape != (1024, 32, 32):
            print(f"[LATENT BAD SHAPE] {latent_path}: {latent.shape}")
            return self.__getitem__((idx + 1) % len(self.samples))
        if skip64.shape != (512, 64, 64):
            print(f"[SKIP64 BAD SHAPE] {latent_path}: {skip64.shape}")
            return self.__getitem__((idx + 1) % len(self.samples))

        return img, latent, skip64, CLASSES.index(cname), os.path.basename(img_path)


# === Watermark NN Generator ===
class ContextAwareWatermarkGen(nn.Module):
    def __init__(self, payload_dim=128, seed_dim=16, latent_shape=(1024, 32, 32), skip_shape=(512, 64, 64)):
        super().__init__()
        self.payload_proj = nn.Linear(payload_dim + seed_dim, np.prod(latent_shape))
        self.fusion = nn.Sequential(
            nn.Conv2d(latent_shape[0] * 2, latent_shape[0], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_shape[0], latent_shape[0], 3, padding=1),
            nn.Tanh()
        )
        self.latent_shape = latent_shape

    def forward(self, payload, random_seed, x, _=None):
        B = x.shape[0]
        full_payload = torch.cat([payload, random_seed], dim=1)
        payload_feat = self.payload_proj(full_payload).view(B, *self.latent_shape)
        fused = torch.cat([x, payload_feat], dim=1)
        return self.fusion(fused)


class MaskHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        nn.init.constant_(self.conv.bias, -0.5)  # стартовая sparse mask

    def forward(self, x):
        return torch.sigmoid(self.conv(x))



def main():

    def clean_state_dict(state_dict):
        # Для загрузки моделей, обученных с DataParallel
        if any(k.startswith('module.') for k in state_dict.keys()):
            from collections import OrderedDict
            return OrderedDict((k.replace('module.', '', 1), v) for k, v in state_dict.items())
        return state_dict

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # --- Models: Use your actual model classes here ---
    wm_gen_latent = ContextAwareWatermarkGen(PAYLOAD_DIM, SEED_DIM, LATENT_SHAPE, SKIP_SHAPE).to(device)
    wm_gen_skip64 = ContextAwareWatermarkGen(
        PAYLOAD_DIM, SEED_DIM, (512, 64, 64), (512, 64, 64)
    ).to(device)

    c2 = EfficientNetB3_CBAM_Bottleneck(num_classes=4).to(device)

    encoder_path = os.path.join(PROJECT_ROOT, "pt models", "autoencoder_epoch7.pth")
    state_dict = torch.load(encoder_path, map_location=device)
    # Извлечь только параметры decoder, убрать префикс
    decoder_state = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}
    decoder = Decoder().to(device)

    # === ПАТЧ: грузим только совпадающие веса ===
    model_state = decoder.state_dict()
    filtered_state = {}
    for k, v in decoder_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered_state[k] = v
        else:
            print(f'[SKIP] {k}: checkpoint shape {v.shape}, model shape {model_state.get(k, None)}')

    decoder.load_state_dict(filtered_state, strict=False)
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
    #lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

    c1 = EfficientNetB3_CBAM_Bottleneck(num_classes=4)
    state_dict_c1 = torch.load(C1_PATH, map_location='cpu')
    state_dict_c1 = clean_state_dict(state_dict_c1)
    c1.load_state_dict(state_dict_c1)
    c1 = nn.DataParallel(c1).to(device)
    for p in c1.parameters():
        p.requires_grad = False
    c1.eval()

    mask_latent_head = MaskHead(in_channels=1024).to(device)
    mask_skip64_head = MaskHead(in_channels=512).to(device)  # 512 каналов для skip64

    print("Initial bias (latent mask):", mask_latent_head.conv.bias.data)
    print("Initial bias (skip64 mask):", mask_skip64_head.conv.bias.data)

    optimizer_gen = torch.optim.Adam([
        {'params': wm_gen_latent.parameters()},
        {'params': wm_gen_skip64.parameters()},
        {'params': mask_latent_head.parameters(), 'lr': 3e-4},  # У mask_latent_head lr в 3 раза выше!
        {'params': mask_skip64_head.parameters(), 'lr': 3e-4},  # То же для skip64
    ], lr=1e-4)

    dataset = WatermarkDataset(DATA_ROOT, CLASSES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

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
                imgs, latents, skip64s, ys, fnames = batch
                imgs = imgs.to(device, non_blocking=True)
                latents = latents.to(device, non_blocking=True)
                skip64s = skip64s.to(device, non_blocking=True)
                ys = ys.to(device, non_blocking=True)

                #print("ys type:", type(ys))
                #print("ys dtype:", ys.dtype)
                #print("ys device:", ys.device)

                B = imgs.shape[0]

                # -- Payload + Seed
                payload = torch.randint(0, 2, (B, PAYLOAD_DIM), dtype=torch.float32, device=device)
                random_seed = torch.randn(B, SEED_DIM, device=device)


                imgs.requires_grad = True
                # -- Compute C2 saliency mask (per image, per batch)
                with torch.enable_grad():
                    out = c2(imgs)
                    class_idx = ys
                    score = out[range(B), class_idx].sum()
                    score.backward()
                    saliency = imgs.grad.abs().mean(dim=1)  # [B, H, W]
                # Normalize, threshold for mask
                saliency_norm = (saliency - saliency.view(B, -1).min(dim=1)[0][:, None, None]) / \
                                (saliency.view(B, -1).max(dim=1)[0][:, None, None] - saliency.view(B, -1).min(dim=1)[0][:, None, None] + 1e-8)
                threshold = torch.quantile(saliency_norm.view(B, -1), 0.9, dim=1, keepdim=True).unsqueeze(-1)
                go_mask = (saliency_norm < threshold).float()



                # -- Watermark generation and embedding
                # Watermark generation
                # latent watermark
                watermark_latent = wm_gen_latent(payload, random_seed, latents, None)
                watermark_skip64 = wm_gen_skip64(payload, random_seed, skip64s, None)
                go_mask_latent = mask_latent_head(latents)
                go_mask_skip64 = mask_skip64_head(skip64s)

                mask_l2_penalty_latent = (go_mask_latent ** 2).mean()
                mask_l2_penalty_skip64 = (go_mask_skip64 ** 2).mean()
                beta_l2 = 0.05  # можно варьировать (0.005...0.02)

                # Entropy penalty (sharpness) — стимулирует маску быть ближе к 0 или 1
                entropy_loss_latent = -(go_mask_latent * torch.log(go_mask_latent + 1e-8) +
                                        (1 - go_mask_latent) * torch.log(1 - go_mask_latent + 1e-8)).mean()
                entropy_loss_skip64 = -(go_mask_skip64 * torch.log(go_mask_skip64 + 1e-8) +
                                        (1 - go_mask_skip64) * torch.log(1 - go_mask_skip64 + 1e-8)).mean()
                sharpness_weight = 0.15 # обычно 0.01...0.05

                go_mask_latent = go_mask_latent.clamp(0, 1) # mask clamping to avoid runaway conditions
                go_mask_skip64 = go_mask_skip64.clamp(0, 1) # mask clamping to avoid runaway conditions

                #print("latents min/max/mean:", latents.min().item(), latents.max().item(), latents.mean().item())
                #print("watermark_latent min/max/mean:", watermark_latent.min().item(),
                #      watermark_latent.max().item(), watermark_latent.mean().item())
                #print("go_mask_latent min/max/mean:", go_mask_latent.min().item(), go_mask_latent.max().item(),
                #      go_mask_latent.mean().item())

                if torch.isnan(latents).any(): print("NAN IN latents")
                if torch.isnan(watermark_latent).any(): print("NAN IN watermark_latent")
                if torch.isnan(go_mask_latent).any(): print("NAN IN go_mask_latent")

                watermarked_latent = latents + intensity*0.5 * watermark_latent * go_mask_latent
                watermarked_skip64 = skip64s + intensity * watermark_skip64 * go_mask_skip64

                watermarked_latent = watermarked_latent.clamp(0, 1)
                watermarked_skip64 = watermarked_skip64.clamp(0, 1)

                wm_img = decoder(watermarked_latent, watermarked_skip64)
                clean_img = decoder(latents, skip64s)

                # Normalize images to [-1, 1] for LPIPS
                #wm_img_lpips = wm_img.clamp(0, 1) * 2 - 1
                #clean_img_lpips = clean_img.clamp(0, 1) * 2 - 1

                #lpips_val = lpips_loss_fn(wm_img_lpips, clean_img_lpips).mean()
                #alpha_lpips = 0.075  # you can tune this coefficient

                print("watermarked_latent min/max/mean:", watermarked_latent.min().item(),
                      watermarked_latent.max().item(), watermarked_latent.mean().item())
                print("watermarked_skip64 min/max/mean:", watermarked_skip64.min().item(),
                      watermarked_skip64.max().item(), watermarked_skip64.mean().item())
                if torch.isnan(watermarked_latent).any():
                    print("NAN in watermarked_latent!")
                if torch.isnan(watermarked_skip64).any():
                    print("NAN in watermarked_skip64!")

                # -- Normalize for classifier
                norm = lambda x: (x - 0.5) / 0.5
                wm_img_norm = norm(wm_img)
                clean_img_norm = norm(clean_img)

                with torch.no_grad():
                    # Compute C1 accuracy on clean and watermarked images
                    c1_clean_pred = c1(clean_img_norm).argmax(1)
                    c1_wm_pred = c1(wm_img_norm).argmax(1)
                    acc_c1_clean = (c1_clean_pred == ys).float().mean().item()
                    acc_c1_wm = (c1_wm_pred == ys).float().mean().item()

                # -- C2 metrics
                # -- C2 метрики и accuracy (можно с no_grad, это только для логов/анализа)
                with torch.no_grad():
                    c2_clean_pred = c2(clean_img_norm).argmax(1)
                    c2_wm_pred = c2(wm_img_norm).argmax(1)
                    acc_c2_clean = (c2_clean_pred == ys).float().mean().item()
                    acc_c2_wm = (c2_wm_pred == ys).float().mean().item()

                    # -- Per-class stats (тоже только для анализа)
                    class_accs_clean = []
                    class_accs_wm = []
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

                # -- Всё что идёт в loss (и участвует в backward) — ТОЛЬКО вне no_grad!
                logits = c2(wm_img_norm)
                #print("c2 logits min/max:", logits.min().item(), logits.max().item())
                if torch.isnan(logits).any():
                    print("NAN in C2 logits!")

                loss_wm = torch.nn.functional.cross_entropy(logits, ys)
                loss_clean = -entropy_loss(
                    c2(clean_img_norm))  # entropy_loss должен быть корректным, без .item()/.detach()
                energy_loss_latent = (watermark_latent.abs() * go_mask_latent).mean()
                energy_loss_skip64 = (watermark_skip64.abs() * go_mask_skip64).mean()
                mask_sparsity = go_mask_latent.mean()
                sparsity_weight = 0.3 #termporary, very large
                mask_sparsity_skip64 = go_mask_skip64.mean()
                sparsity_weight_skip64 = 0.005
                l1_img = (wm_img - clean_img).abs().mean()
                alpha_l1 = 0.05

                if torch.isnan(wm_img).any(): print("NAN IN wm_img")
                if torch.isinf(wm_img).any(): print("INF IN wm_img")

                # C1 accuracy drop penalty (only if acc drops below 0.90)
                drop = acc_c1_clean - acc_c1_wm
                penalty_c1 = torch.tensor(0.0, device=device)
                if acc_c1_wm < 0.97:
                    penalty_c1 = torch.tensor(abs(acc_c1_clean - acc_c1_wm) * 40.0,
                                              device=device)  # You can tune weight

                total_loss = (
                        loss_wm
                        + loss_clean
                        + 0.01 * energy_loss_latent
                        + 0.005 * energy_loss_skip64
                        + sparsity_weight * mask_sparsity
                        + sparsity_weight_skip64 * mask_sparsity_skip64
                        + alpha_l1 * l1_img
                        + penalty_c1
                        + beta_l2 * mask_l2_penalty_latent
                        + beta_l2 * mask_l2_penalty_skip64
                        + sharpness_weight * entropy_loss_latent
                        + sharpness_weight * entropy_loss_skip64
                        #+ alpha_lpips * lpips_val  # <-- NEW TERM
                )

                def nan_report(name, tensor):
                    if torch.isnan(tensor).any():
                        print(
                            f"[NAN WARNING] {name} contains nan! Min: {tensor.min().item()}, Max: {tensor.max().item()}")
                    else:
                        print(
                            f"[OK] {name} min: {tensor.min().item()} max: {tensor.max().item()} mean: {tensor.mean().item()}")

                nan_report("total_loss", total_loss)
                nan_report("loss_wm", loss_wm)
                nan_report("loss_clean", loss_clean)
                nan_report("mask_sparsity", mask_sparsity)
                nan_report("mask_sparsity_skip64", mask_sparsity_skip64)
                nan_report("l1_img", l1_img)
                nan_report("go_mask_latent", go_mask_latent)
                nan_report("go_mask_skip64", go_mask_skip64)
                nan_report("wm_img", wm_img)
                nan_report("clean_img", clean_img)

                watermark_latent = watermark_latent.clamp(-1, 1)
                watermark_skip64 = watermark_skip64.clamp(-1, 1)
                go_mask_latent = go_mask_latent.clamp(0, 1)
                go_mask_skip64 = go_mask_skip64.clamp(0, 1)

                print("wm_img.requires_grad:", wm_img.requires_grad)
                print("clean_img requires grad:", clean_img.requires_grad)
                print("go_mask_latent.requires_grad:", go_mask_latent.requires_grad)
                print("watermarked_latent.requires_grad:", watermarked_latent.requires_grad)
                c2_logits = c2(wm_img_norm)
                print("c2_logits.requires_grad:", c2_logits.requires_grad)
                print("logits grad_fn:", logits.grad_fn)

                print("loss_wm requires_grad:", loss_wm.requires_grad)
                print("total_loss requires_grad:", total_loss.requires_grad)

                print("loss_wm type:", type(loss_wm))
                print("loss_wm requires_grad:", loss_wm.requires_grad)
                print("total_loss type:", type(total_loss))
                print("total_loss requires_grad:", total_loss.requires_grad)

                try:
                    with torch.autograd.set_detect_anomaly(True):
                        scaler.scale(total_loss).backward()

                except RuntimeError as e:
                    print("CAUGHT ANOMALY IN BACKWARD:", e)
                    print("watermarked_latent min/max/mean:", watermarked_latent.min().item(),
                          watermarked_latent.max().item(), watermarked_latent.mean().item())
                    print("watermarked_skip64 min/max/mean:", watermarked_skip64.min().item(),
                          watermarked_skip64.max().item(), watermarked_skip64.mean().item())
                    print("go_mask_latent min/max/mean:", go_mask_latent.min().item(), go_mask_latent.max().item(),
                          go_mask_latent.mean().item())
                    print("go_mask_skip64 min/max/mean:", go_mask_skip64.min().item(), go_mask_skip64.max().item(),
                          go_mask_skip64.mean().item())
                    print("wm_img min/max/mean:", wm_img.min().item(), wm_img.max().item(), wm_img.mean().item())
                    print("clean_img min/max/mean:", clean_img.min().item(), clean_img.max().item(),
                          clean_img.mean().item())
                    raise

                for name, module in [("mask_latent_head", mask_latent_head), ("mask_skip64_head", mask_skip64_head)]:
                    grad_norm = 0.
                    for p in module.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm().item()
                    print(f"[Log] {name} grad norm: {grad_norm:.5f}" if grad_norm > 0 else f"[Log] {name} grad: None")

                # Логирование нормы градиента mask head (latent)
                if mask_latent_head.conv.weight.grad is not None:
                    print(f"[Log] mask_latent_head grad norm: {mask_latent_head.conv.weight.grad.norm().item():.5f}")
                else:
                    print("[Log] mask_latent_head grad: None")

                # Логирование нормы градиента mask head (skip64)
                if mask_skip64_head.conv.weight.grad is not None:
                    print(f"[Log] mask_skip64_head grad norm: {mask_skip64_head.conv.weight.grad.norm().item():.5f}")
                else:
                    print("[Log] mask_skip64_head grad: None")

                scaler.step(optimizer_c2)
                scaler.step(optimizer_gen)
                scaler.update()
                optimizer_c2.zero_grad()
                optimizer_gen.zero_grad()

                # -- Logging batch stats for CSV
                epoch_metrics.append({
                    'cycle': cycle + 1, 'epoch': epoch + 1,
                    'acc_c2_clean': acc_c2_clean,
                    'acc_c2_wm': acc_c2_wm,
                    'min_clean': min_clean, 'max_clean': max_clean, 'mean_clean': mean_clean,
                    'min_wm': min_wm, 'max_wm': max_wm, 'mean_wm': mean_wm,
                    'energy_loss_latent': energy_loss_latent.item(),
                    'energy_loss_skip64': energy_loss_skip64,
                    'intensity': intensity,
                    # --- Add C1
                    'acc_c1_clean': acc_c1_clean,
                    'acc_c1_wm': acc_c1_wm,
                    'c1_drop': drop
                })

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

                for i in range(B):
                    try:
                        # 1. Original
                        orig_img = imgs[i].detach().cpu().numpy().transpose(1, 2, 0)
                        orig_img = (orig_img * 255).clip(0, 255).astype(np.uint8)
                        orig_img = cv2.resize(orig_img, (512, 512), interpolation=cv2.INTER_CUBIC)
                        orig_img_labeled = draw_label(orig_img, "Original")

                        # 2. Watermarked
                        wm_img_np = wm_img[i].detach().cpu().numpy().transpose(1, 2, 0)
                        wm_img_np = (wm_img_np * 255).clip(0, 255).astype(np.uint8)
                        wm_img_np = cv2.resize(wm_img_np, (512, 512), interpolation=cv2.INTER_CUBIC)
                        wm_img_labeled = draw_label(wm_img_np, "Watermarked")

                        # 3. Latent mask overlay
                        softmask_latent = go_mask_latent[i][0].detach().cpu().numpy()
                        softmask_latent = np.nan_to_num(softmask_latent)
                        mask_lat = cv2.resize(softmask_latent, (512, 512), interpolation=cv2.INTER_CUBIC)
                        mask_lat_norm = (mask_lat - mask_lat.min()) / (np.ptp(mask_lat) + 1e-8) # temperature related binarization - increasing separation
                        overlay_lat = (cm.jet(mask_lat_norm)[..., :3] * 255).astype(np.uint8)
                        overlay_lat = cv2.addWeighted(orig_img, 0.7, overlay_lat, 0.3, 0)
                        overlay_lat_labeled = draw_label(overlay_lat, "Latent Mask Overlay")

                        # 4. Skip64 mask overlay
                        softmask_skip64 = go_mask_skip64[i][0].detach().cpu().numpy()
                        softmask_skip64 = np.nan_to_num(softmask_skip64)
                        mask_skip = cv2.resize(softmask_skip64, (512, 512), interpolation=cv2.INTER_CUBIC)
                        mask_skip_norm = (mask_skip - mask_skip.min()) / (np.ptp(mask_skip) + 1e-8)
                        overlay_skip = (cm.jet(mask_skip_norm)[..., :3] * 255).astype(np.uint8)
                        overlay_skip = cv2.addWeighted(orig_img, 0.7, overlay_skip, 0.3, 0)
                        overlay_skip_labeled = draw_label(overlay_skip, "Skip64 Mask Overlay")

                        # 5. Latent heatmap
                        wm_lat = watermark_latent[i].detach().cpu().numpy()
                        wm_lat_map = np.abs(wm_lat).mean(axis=0)
                        wm_lat_map = np.nan_to_num(wm_lat_map)
                        wm_lat_map = cv2.resize(wm_lat_map, (512, 512), interpolation=cv2.INTER_CUBIC)
                        wm_lat_map_norm = (wm_lat_map - wm_lat_map.min()) / (np.ptp(wm_lat_map) + 1e-8)
                        heatmap_lat = (cm.jet(wm_lat_map_norm)[..., :3] * 255).astype(np.uint8)
                        heatmap_lat_labeled = draw_label(heatmap_lat, "Latent Heatmap")

                        # 6. Skip64 heatmap
                        wm_skip = watermark_skip64[i].detach().cpu().numpy()
                        wm_skip_map = np.abs(wm_skip).mean(axis=0)
                        wm_skip_map = np.nan_to_num(wm_skip_map)
                        wm_skip_map = cv2.resize(wm_skip_map, (512, 512), interpolation=cv2.INTER_CUBIC)
                        wm_skip_map_norm = (wm_skip_map - wm_skip_map.min()) / (np.ptp(wm_skip_map) + 1e-8)
                        heatmap_skip = (cm.jet(wm_skip_map_norm)[..., :3] * 255).astype(np.uint8)
                        heatmap_skip_labeled = draw_label(heatmap_skip, "Skip64 Heatmap")

                        # Сборка коллажа
                        collage = np.concatenate([
                            orig_img_labeled,
                            wm_img_labeled,
                            overlay_lat_labeled,
                            overlay_skip_labeled,
                            heatmap_lat_labeled,
                            heatmap_skip_labeled
                        ], axis=1)

                        # Имя класса — строго по датасету!
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
                    #print(f"[Log] LPIPS (wm-clean): {lpips_val.item():.5f}")

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
        if early_stop:
            continue  # Proceed to next cycle (or break if last)

    print("Training complete and all results saved.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()


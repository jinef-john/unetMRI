

import os, csv, torch, random, numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image, ImageDraw, ImageFont
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3
from MRI_C1_B3_CBAM import CBAM
import matplotlib.cm as cm
import cv2

# ============ CONFIG ============
# Get project root directory (three levels up from current file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset", "brain-tumor-mri-dataset", "Training")
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
SAL_ROOT = os.path.join(PROJECT_ROOT, "output", "saliency_cycles_latent_skip64_skip128")
WM_ROOT  = os.path.join(PROJECT_ROOT, "output", "watermarked_cycles_latent_skip64_skip128")
CSV_ROOT = os.path.join(PROJECT_ROOT, "output", "metrics_logs_latent_skip64_skip128")
C1_PATH = os.path.join(PROJECT_ROOT, "pt models", "MRI-C1EfficientNet_B3_CBAM.pth")
NPZ_ROOT = os.path.join(PROJECT_ROOT, "output", "MRI-NPZ_latent_skip64_skip128")

CYCLES = 5
EPOCHS_PER_CYCLE = 5
BATCH_SIZE = 1
INTENSITIES = [0.015, 0.02, 0.04, 0.06, 0.08]
EARLY_STOP_C2_CLEAN = 0.20
EARLY_STOP_C2_WM    = 0.90

PAYLOAD_DIM, SEED_DIM = 128, 16
LATENT_SHAPE, SKIP_SHAPE = (1024, 32, 32), (512, 64, 64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(SAL_ROOT, exist_ok=True)
os.makedirs(WM_ROOT, exist_ok=True)
os.makedirs(CSV_ROOT, exist_ok=True)

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
        nn.init.constant_(self.conv.bias, -0.5)
    def forward(self, x):
        return torch.sigmoid(self.conv(x))

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


def save_watermark_collage(orig_img, wm_img, go_mask_latent, go_mask_skip64, fname, out_dir):
    import numpy as np
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.cm as cm

    orig = orig_img[0].detach().cpu().numpy()
    orig = (orig * 255).clip(0, 255).astype(np.uint8)[0]
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)

    wm = wm_img[0].detach().cpu().numpy()
    wm = (wm * 255).clip(0, 255).astype(np.uint8)[0]
    wm_rgb = cv2.cvtColor(wm, cv2.COLOR_GRAY2RGB)

    diff = np.abs(orig.astype(np.int16) - wm.astype(np.int16)).astype(np.uint8)
    diff_rgb = np.stack([diff, diff, diff], axis=-1)
    mask_nonzero = diff > 8
    white = np.ones_like(diff_rgb) * 255
    diff_rgb = np.where(mask_nonzero[..., None], diff_rgb, white)

    # Always use float32 for OpenCV
    latent_mask = go_mask_latent[0, 0].detach().cpu().numpy()
    latent_mask = np.nan_to_num(latent_mask, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if latent_mask.size == 0 or np.all(latent_mask == 0):
        latent_mask_resized = np.zeros((512, 512), dtype=np.float32)
        latent_mask_norm = latent_mask_resized
    else:
        try:
            latent_mask_resized = cv2.resize(latent_mask, (512, 512), interpolation=cv2.INTER_CUBIC)
            latent_mask_norm = (latent_mask_resized - latent_mask_resized.min()) / (np.ptp(latent_mask_resized) + 1e-8)
        except Exception as e:
            print(f"cv2.resize error (latent): {e}, shape={latent_mask.shape}, dtype={latent_mask.dtype}")
            latent_mask_norm = np.zeros((512, 512), dtype=np.float32)
    latent_heatmap = (cm.jet(latent_mask_norm)[..., :3] * 255).astype(np.uint8)

    skip_mask = go_mask_skip64[0, 0].detach().cpu().numpy()
    skip_mask = np.nan_to_num(skip_mask, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if skip_mask.size == 0 or np.all(skip_mask == 0):
        skip_mask_resized = np.zeros((512, 512), dtype=np.float32)
        skip_mask_norm = skip_mask_resized
    else:
        try:
            skip_mask_resized = cv2.resize(skip_mask, (512, 512), interpolation=cv2.INTER_CUBIC)
            skip_mask_norm = (skip_mask_resized - skip_mask_resized.min()) / (np.ptp(skip_mask_resized) + 1e-8)
        except Exception as e:
            print(f"cv2.resize error (skip64): {e}, shape={skip_mask.shape}, dtype={skip_mask.dtype}")
            skip_mask_norm = np.zeros((512, 512), dtype=np.float32)
    skip_heatmap = (cm.jet(skip_mask_norm)[..., :3] * 255).astype(np.uint8)

    def put_label(img, text):
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        text_height = bbox[3] - bbox[1]
        draw.rectangle([0, 0, img_pil.width, text_height + 4], fill=(255, 255, 255, 220))
        draw.text((5, 2), text, fill=(0, 0, 0), font=font)
        return np.array(img_pil)

    orig_labeled = put_label(orig_rgb, "Original")
    wm_labeled = put_label(wm_rgb, "Watermarked")
    diff_labeled = put_label(diff_rgb, "Difference (on white)")
    latent_labeled = put_label(latent_heatmap, "Latent Mask Heatmap")
    skip_labeled = put_label(skip_heatmap, "Skip64 Mask Heatmap")

    collage = np.concatenate([orig_labeled, wm_labeled, diff_labeled, latent_labeled, skip_labeled], axis=1)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{fname}_collage.png")
    Image.fromarray(collage).save(out_path)



# ============ TRAINING LOOP ============
def main():
    dataset = MRIWatermarkDataset(DATA_ROOT, NPZ_ROOT, CLASSES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=False, persistent_workers=True)
    # MODELS (FP16)
    wm_latent = WatermarkGen(PAYLOAD_DIM, SEED_DIM, LATENT_SHAPE).to(device)
    wm_skip64 = WatermarkGen(PAYLOAD_DIM, SEED_DIM, SKIP_SHAPE).to(device)
    decoder = Decoder().to(device)
    c2 = EfficientNetB3_CBAM_Bottleneck(num_classes=4).to(device)
    mask_latent = MaskHead(1024).to(device)
    mask_skip64 = MaskHead(512).to(device)
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

    optimizer_c2 = torch.optim.AdamW(c2.parameters(), lr=1e-4)
    optimizer_gen = torch.optim.AdamW(list(wm_latent.parameters()) + list(wm_skip64.parameters())
        + list(mask_latent.parameters()) + list(mask_skip64.parameters()), lr=1e-4)
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
                imgs, latents, skip64s, ys, fnames = batch
                imgs = imgs.to(device, non_blocking=True)
                latents = latents.to(device, non_blocking=True)
                skip64s = skip64s.to(device, non_blocking=True)
                ys = ys.to(device, non_blocking=True)
                B = imgs.shape[0]
                payload = torch.randint(0, 2, (B, PAYLOAD_DIM), dtype=torch.float32, device=device)
                random_seed = torch.randn(B, SEED_DIM, device=device).half()

                with autocast(device_type="cuda"):
                    watermark_latent = wm_latent(payload, random_seed, latents, None)
                    watermark_skip64 = wm_skip64(payload, random_seed, skip64s, None)
                    go_mask_latent = mask_latent(latents).clamp(0,1)
                    go_mask_skip64 = mask_skip64(skip64s).clamp(0,1)
                    watermarked_latent = (latents + intensity*0.5*watermark_latent*go_mask_latent).clamp(0,1)
                    watermarked_skip64 = (skip64s + intensity*watermark_skip64*go_mask_skip64).clamp(0,1)
                    wm_img = decoder(watermarked_latent, watermarked_skip64)
                    clean_img = decoder(latents, skip64s)
                    wm_img_norm = norm(wm_img).float()
                    clean_img_norm = norm(clean_img).float()

                save_watermark_collage(
                    imgs.cpu(),
                    wm_img.cpu(),
                    go_mask_latent.cpu(),
                    go_mask_skip64.cpu(),
                    fnames[0],
                    WM_ROOT
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
                loss_clean = -entropy_loss(c2(clean_img_norm))
                mask_l2_latent = (go_mask_latent ** 2).mean()
                mask_l2_skip64 = (go_mask_skip64 ** 2).mean()
                entropy_latent = -(go_mask_latent*torch.log(go_mask_latent+1e-8) + (1-go_mask_latent)*torch.log(1-go_mask_latent+1e-8)).mean()
                entropy_skip64 = -(go_mask_skip64*torch.log(go_mask_skip64+1e-8) + (1-go_mask_skip64)*torch.log(1-go_mask_skip64+1e-8)).mean()
                energy_loss_latent = (watermark_latent.abs() * go_mask_latent).mean()
                energy_loss_skip64 = (watermark_skip64.abs() * go_mask_skip64).mean()
                mask_sparsity_latent = go_mask_latent.mean()
                mask_sparsity_skip64 = go_mask_skip64.mean()
                l1_img = (wm_img - clean_img).abs().mean()
                penalty_c1 = torch.tensor(0.0, device=device)
                if acc_c1_wm < 0.97:
                    penalty_c1 = torch.tensor(abs(acc_c1_clean - acc_c1_wm) * 40.0, device=device)
                total_loss = (
                    loss_wm + loss_clean
                    + 0.01 * energy_loss_latent + 0.005 * energy_loss_skip64
                    + 0.3 * mask_sparsity_latent + 0.005 * mask_sparsity_skip64
                    + 0.05 * l1_img
                    + penalty_c1
                    + 0.05 * mask_l2_latent + 0.05 * mask_l2_skip64
                    + 0.15 * entropy_latent + 0.15 * entropy_skip64
                )
                scaler.scale(total_loss).backward()
                scaler.step(optimizer_c2)
                scaler.step(optimizer_gen)
                scaler.update()
                optimizer_c2.zero_grad(set_to_none=True)
                optimizer_gen.zero_grad(set_to_none=True)

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

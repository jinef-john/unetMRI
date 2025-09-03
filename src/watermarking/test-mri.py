"""
FIXED Adversarial Watermarking Pipeline:
- Mode-detection based training instead of confused classifier
- Clean images: random class predictions via entropy maximization
- Watermarked images: correct predictions via detectable watermark signal
- C2 learns to detect watermark presence, then classify accordingly
"""

import os, sys, csv, glob, time, math, random, pathlib
from datetime import datetime
import numpy as np
import cv2
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

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ====== IMPORTING LOCAL MODULES =====
sys.path.append(os.path.join(PROJECT_ROOT, "src", "models"))
from efficientnet_cbam import EfficientNetB3_CBAM_Bottleneck
from autoencoder import Encoder, Decoder

# ---------------- Paths & Config ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_ROOT = os.path.join(PROJECT_ROOT, "dataset", "brain-tumor-mri-dataset", "Training")
U2NET_ROOT = os.path.join(PROJECT_ROOT, "output", "u2net_masks_png")
WM_ROOT = os.path.join(PROJECT_ROOT, "output", "watermarked_cycles_FIXED")
CSV_ROOT = os.path.join(PROJECT_ROOT, "output", "csv_logs_FIXED")
C1_PATH = os.path.join(PROJECT_ROOT, "pt models", "MRI-C1EfficientNet_B3_CBAM.pth")
AE_PATH = os.path.join(PROJECT_ROOT, "pt models", "autoencoder_epoch7.pth")
C2_SAVE_DIR = os.path.join(PROJECT_ROOT, "output", "C2-FIXED")

os.makedirs(WM_ROOT, exist_ok=True)
os.makedirs(CSV_ROOT, exist_ok=True)
os.makedirs(C2_SAVE_DIR, exist_ok=True)

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS2IDX = {c:i for i,c in enumerate(CLASSES)}

# Training hyperparams
EPOCHS = 15                # Increased for mode-based training
BATCH_SIZE_WARMUP = 64
BATCH_SIZE_ADV = 16
NUM_WORKERS = 16
AMP = True

C2_WARMUP_EPOCHS = 1
ADVERSARIAL_EPOCHS = 14   # Extended for mode learning
FINE_TUNE_EPOCHS = 25

LR_C2_WARMUP = 1e-3
LR_C2_ADV = 5e-4
LR_GEN = 1e-3

# Loss weights - optimized for mode detection
ALPHA_MODE_DETECTION = 2.0   # Mode detection strength
ALPHA_ENTROPY_MAX = 3.0      # Entropy maximization for clean
BETA_CLASS_WM = 1.5          # Correct classification for watermarked
GAMMA_RECONSTRUCTION = 0.1   # Image quality
DELTA_EXCLUSION = 5.0        # Brain exclusion
EPSILON_DETECTION_GEN = 2.0  # Generator detectability

# Watermark parameters
INTENSITY_INIT = 0.5        # Increased for better detection
MASK_FRAC_INIT = 0.15
INTENSITY_SCALE = 2.5

# ---------------- Dataset ----------------
class MRIDataset(Dataset):
    def __init__(self, root_dir, classes):
        self.root_dir = root_dir
        self.classes = classes
        self.files = []
        self.labels = []
        
        for i, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            if os.path.exists(cls_dir):
                for f in os.listdir(cls_dir):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.files.append(os.path.join(cls_dir, f))
                        self.labels.append(i)
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]
        
        from PIL import Image
        img = Image.open(img_path).convert('L')
        img = self.transform(img)
        
        fname = os.path.basename(img_path)
        cls_name = self.classes[label]
        
        return img, label, fname, cls_name

# ---------------- Watermark Generator ----------------
class WatermarkGeneratorMiniUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels_latent=1024, out_channels_skip=512):
        super().__init__()
        
        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder paths
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output heads
        self.out_latent = nn.Conv2d(64, out_channels_latent, 1)
        self.out_skip = nn.Conv2d(64, out_channels_skip, 1)
        
    def forward(self, x, intensity=1.0):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder
        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))
        
        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))
        
        # Generate watermarks
        wm_full = self.out_latent(d2)
        wm_skip_full = self.out_skip(d2)
        
        # Resize to target dimensions
        wm_latent = F.interpolate(wm_full, size=(32, 32), mode='bilinear', align_corners=False)
        wm_skip = F.interpolate(wm_skip_full, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Apply intensity scaling
        wm_latent = torch.tanh(wm_latent) * intensity
        wm_skip = torch.tanh(wm_skip) * intensity
        
        return wm_latent, wm_skip

# ---------------- Mode Detector Model ----------------
class ModeDetector(nn.Module):
    """C2 that first detects watermark presence (0=clean, 1=wm) then classifies."""
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EfficientNetB3_CBAM_Bottleneck(num_classes=num_classes)
        # backbone’s final feature map channels
        feat_dim = self.backbone.base.classifier[1].in_features
        self.mode_head  = nn.Linear(feat_dim, 2)
        self.class_head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        return self.mode_head(feats), self.class_head(feats)

# ---------------- Model Loading Functions ----------------
def load_c1_frozen():
    """Load frozen C1 model"""
    print("Loading C1 model...")
    c1 = EfficientNetB3_CBAM_Bottleneck(num_classes=len(CLASSES)).to(DEVICE)
    
    # Modify first layer for 1-channel input
    first_conv = c1.base.features[0][0]
    c1.base.features[0][0] = nn.Conv2d(1, first_conv.out_channels, 
                                       kernel_size=first_conv.kernel_size,
                                       stride=first_conv.stride, 
                                       padding=first_conv.padding, 
                                       bias=False)
    
    # Load weights
    checkpoint = torch.load(C1_PATH, map_location=DEVICE)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Handle DataParallel prefix
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    c1.load_state_dict(state_dict)
    c1.eval()
    for param in c1.parameters():
        param.requires_grad = False
    return c1

def load_autoencoder_frozen():
    """Load frozen autoencoder"""
    checkpoint = torch.load(AE_PATH, map_location=DEVICE)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    encoder = Encoder()
    decoder = Decoder()
    
    encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}, strict=False)
    decoder.load_state_dict({k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}, strict=False)
    
    encoder.eval().requires_grad_(False)
    decoder.eval().requires_grad_(False)
    
    return encoder.to(DEVICE), decoder.to(DEVICE)

def build_c2():
    """Build mode detector C2"""
    c2 = ModeDetector(num_classes=len(CLASSES))
    return c2.to(DEVICE)

# ---------------- U2Net Mask Loading ----------------
def load_u2net_mask_batch(fnames, classes, size_512=(512,512)):
    batch_size = len(fnames)
    masks = torch.zeros(batch_size, 1, *size_512)
    
    for i, (f, cls) in enumerate(zip(fnames, classes)):
        mask_path = os.path.join(U2NET_ROOT, cls, os.path.splitext(f)[0] + ".png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, size_512)
            masks[i, 0] = torch.from_numpy(mask / 255.0)
    
    return masks

# ---------------- Utility Functions ----------------
def sobel_energy_map(img):
    """Compute Sobel edge energy for maskability"""
    img_np = img.detach().cpu().numpy()
    energy = torch.zeros_like(img)
    
    for b in range(img.shape[0]):
        gray = img_np[b, 0]
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_mag = (sobel_mag - sobel_mag.min()) / (sobel_mag.max() - sobel_mag.min() + 1e-8)
        energy[b, 0] = torch.from_numpy(sobel_mag)
    
    return energy.to(img.device)

def quantile_binary(maskability, exclusion_mask, target_fraction=0.15):
    """Create binary mask based on quantile thresholding"""
    valid_region = maskability * exclusion_mask
    batch_size = maskability.shape[0]
    binary_masks = torch.zeros_like(maskability)
    
    for b in range(batch_size):
        valid_vals = valid_region[b][valid_region[b] > 0]
        if len(valid_vals) > 0:
            threshold = torch.quantile(valid_vals, 1 - target_fraction)
            binary_mask = (maskability[b] >= threshold) & (exclusion_mask[b] > 0)
            binary_masks[b] = binary_mask.float()
    
    return binary_masks, []

def compute_exclusion_penalty(self, watermark, brain_mask, intensity):
    """Penalty for watermarking in critical regions"""
    brain_mask_resized = F.interpolate(brain_mask, size=watermark.shape[-2:], mode='nearest')
    penalty = torch.sum(torch.abs(watermark) * brain_mask_resized) * intensity
    return penalty

# ---------------- Fixed Adversarial Trainer ----------------
class AdversarialTrainer:
    def __init__(self, c1, c2, encoder, decoder, watermark_gen):
        self.c1 = c1
        self.c2 = c2 
        self.encoder = encoder
        self.decoder = decoder
        self.watermark_gen = watermark_gen
        
        self.norm_c1 = transforms.Normalize(mean=[0.5], std=[0.5])
        self.class_embed = nn.Embedding(len(CLASSES), 1024).to(DEVICE)
        
        self.c2_optimizer = None
        self.gen_optimizer = None
        self.setup_optimizers(stage="warmup")
        
    def setup_optimizers(self, stage):
        """Setup optimizers for different training stages"""
        if stage == "warmup":
            self.c2_optimizer = torch.optim.AdamW(self.c2.parameters(), lr=LR_C2_WARMUP, weight_decay=1e-4)
        elif stage == "adversarial":
            self.c2_optimizer = torch.optim.AdamW(self.c2.parameters(), lr=LR_C2_ADV, weight_decay=1e-4)
        
        gen_params = list(self.watermark_gen.parameters()) + list(self.class_embed.parameters())
        self.gen_optimizer = torch.optim.AdamW(gen_params, lr=LR_GEN, weight_decay=1e-4)
    
    def run_encoder_decoder(self, img):
        """Run encoder-decoder pipeline"""
        with torch.no_grad():
            img_norm = (img - 0.5) / 0.5
            latents, skip64s = self.encoder(img_norm)
            clean_img = self.decoder(latents, skip64s).clamp(0, 1)
        return latents, skip64s, clean_img
    
    def generate_watermark(self, img, labels, intensity):
        """Generate conditional watermark"""
        wm_latent, wm_skip = self.watermark_gen(img, intensity)
        
        # Strong class conditioning
        class_bias = self.class_embed(labels).view(labels.size(0), -1, 1, 1)
        class_bias = class_bias.expand(-1, -1, 32, 32)
        wm_latent_cond = wm_latent + class_bias
        
        return wm_latent_cond, wm_skip
    
    def embed_watermark_frequency_domain(self, latents, skip64s, wm_latent, wm_skip, exclusion_mask, intensity_scale=2.0):
        """Embed watermark in frequency domain with exclusion"""
        exclusion_lat = F.interpolate(exclusion_mask, size=(32, 32), mode='nearest')
        exclusion_skip = F.interpolate(exclusion_mask, size=(64, 64), mode='nearest')
        
        # Frequency domain embedding
        latent_freq = fft2(latents)
        wm_latent_masked = wm_latent * exclusion_lat
        latent_freq_wm = latent_freq + (wm_latent_masked * intensity_scale)
        latents_wm = ifft2(latent_freq_wm).real
        
        # Skip64 embedding
        skip_freq = fft2(skip64s)
        wm_skip_masked = wm_skip * exclusion_skip
        skip_freq_wm = skip_freq + (wm_skip_masked * intensity_scale * 0.5)
        skip64s_wm = ifft2(skip_freq_wm).real
        
        return latents_wm, skip64s_wm
    
    def train_step_warmup(self, batch, intensity):
        """Warmup training - train C2 normally on clean data"""
        img, labels, fnames, cls_names = batch
        img = img.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Get clean reconstruction
        latents, skip64s, clean_img = self.run_encoder_decoder(img)
        clean_norm = self.norm_c1(clean_img)
        
        # Train mode detector on clean images
        mode_logits, class_logits = self.c2(clean_norm)
        
        # Mode detection: all should be "clean" (0)
        mode_targets = torch.zeros(len(img), dtype=torch.long, device=DEVICE)
        mode_loss = F.cross_entropy(mode_logits, mode_targets)
        
        # Class prediction: normal training
        class_loss = F.cross_entropy(class_logits, labels)
        
        loss_c2 = mode_loss + class_loss
        
        self.c2_optimizer.zero_grad()
        loss_c2.backward()
        self.c2_optimizer.step()
        
        # Metrics
        with torch.no_grad():
            mode_acc = (mode_logits.argmax(1) == mode_targets).float().mean().item()
            class_acc = (class_logits.argmax(1) == labels).float().mean().item()
        
        return {
            'loss_c2': loss_c2.item(),
            'mode_acc': mode_acc,
            'class_acc': class_acc,
            'stage': 'warmup'
        }
    
    def train_step_adversarial(self, batch, intensity, mask_fraction):
        """Fixed adversarial training with mode detection"""
        torch.cuda.empty_cache()
        
        img, labels, fnames, cls_names = batch
        img = img.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Get clean reconstruction
        latents, skip64s, clean_img = self.run_encoder_decoder(img)
        
        # Generate exclusion masks
        u2_masks = load_u2net_mask_batch(fnames, cls_names).to(DEVICE)
        brain_mask = (u2_masks > 0.5).float()
        exclusion_mask = (1.0 - brain_mask)
        
        # Maskability
        maskability = sobel_energy_map(img)
        allow_mask, _ = quantile_binary(maskability, exclusion_mask, mask_fraction)
        
        # Generate watermark
        wm_latent, wm_skip = self.generate_watermark(img, labels, intensity)
        latents_wm, skip64s_wm = self.embed_watermark_frequency_domain(
            latents, skip64s, wm_latent, wm_skip, allow_mask, INTENSITY_SCALE)
        
        with torch.no_grad():
            wm_img = self.decoder(latents_wm, skip64s_wm).clamp(0, 1)
        
        clean_norm = self.norm_c1(clean_img)
        wm_norm = self.norm_c1(wm_img)
        
        # Create combined batch for efficient training
        combined_imgs = torch.cat([clean_norm, wm_norm])
        combined_labels = torch.cat([labels, labels])
        
        # Shuffle to prevent ordering bias
        perm = torch.randperm(len(combined_imgs))
        combined_imgs = combined_imgs[perm]
        combined_labels = combined_labels[perm]
        
        # Determine which are clean vs watermarked after shuffling
        clean_mask = perm < len(clean_norm)
        wm_mask = ~clean_mask
        
        # === Step 1: Train C2 (Mode Detector) ===
        self.c2_optimizer.zero_grad()
        
        mode_logits, class_logits = self.c2(combined_imgs)
        
        # Mode detection targets
        mode_targets = torch.zeros(len(combined_imgs), dtype=torch.long, device=DEVICE)
        mode_targets[wm_mask] = 1  # 1 = watermarked
        
        mode_loss = F.cross_entropy(mode_logits, mode_targets)
        
        # Class prediction strategy
        # Clean images: maximize entropy (random predictions)
        clean_probs = F.softmax(class_logits[clean_mask], dim=1)
        clean_entropy = -(clean_probs * torch.log(clean_probs + 1e-8)).sum(dim=1).mean()
        
        # Watermarked images: correct predictions
        wm_class_loss = F.cross_entropy(class_logits[wm_mask], combined_labels[wm_mask])
        
        # Total C2 loss
        loss_c2 = (mode_loss - 
                   ALPHA_ENTROPY_MAX * clean_entropy + 
                   BETA_CLASS_WM * wm_class_loss)
        
        loss_c2.backward()
        self.c2_optimizer.step()
        
        # === Step 2: Train Generator ===
        self.gen_optimizer.zero_grad()
        
        # Regenerate with gradients
        wm_latent, wm_skip = self.generate_watermark(img, labels, intensity)
        latents_wm, skip64s_wm = self.embed_watermark_frequency_domain(
            latents, skip64s, wm_latent, wm_skip, allow_mask, INTENSITY_SCALE)
        wm_img_gen = self.decoder(latents_wm, skip64s_wm).clamp(0, 1)
        wm_norm_gen = self.norm_c1(wm_img_gen)
        
        # Generator losses
        mode_logits_gen, class_logits_gen = self.c2(wm_norm_gen)
        
        # Ensure watermarks are detectable
        detection_loss = F.cross_entropy(mode_logits_gen, torch.ones(len(img), device=DEVICE))
        
        # Ensure correct classification for watermarked
        class_loss = F.cross_entropy(class_logits_gen, labels)
        
        # Quality preservation
        l1_loss = F.l1_loss(wm_img_gen, clean_img)
        
        # Exclusion penalty
        exclusion_penalty = torch.sum(torch.abs(wm_latent) * 
                                    F.interpolate(brain_mask, size=(32, 32), mode='nearest')) * intensity
        
        loss_gen = (EPSILON_DETECTION_GEN * detection_loss + 
                   0.5 * class_loss + 
                   GAMMA_RECONSTRUCTION * l1_loss + 
                   DELTA_EXCLUSION * exclusion_penalty)
        
        loss_gen.backward()
        self.gen_optimizer.step()
        
        # === Metrics ===
        with torch.no_grad():
            # C1 performance (quality check)
            logits_c1_clean = self.c1(clean_norm)
            logits_c1_wm = self.c1(wm_norm)
            acc_c1_clean = (logits_c1_clean.argmax(1) == labels).float().mean().item()
            acc_c1_wm = (logits_c1_wm.argmax(1) == labels).float().mean().item()
            
            # Mode detection accuracy
            mode_preds = mode_logits.argmax(1)
            mode_acc_clean = (mode_preds[clean_mask] == 0).float().mean().item()
            mode_acc_wm = (mode_preds[wm_mask] == 1).float().mean().item()
            
            # Class accuracy
            class_preds = class_logits.argmax(1)
            class_acc_clean = (class_preds[clean_mask] == combined_labels[clean_mask]).float().mean().item()
            class_acc_wm = (class_preds[wm_mask] == combined_labels[wm_mask]).float().mean().item()
            
            # Quality
            l1_metric = F.l1_loss(wm_img_gen, clean_img).item()
        
        return {
            'loss_c2': loss_c2.item(),
            'loss_gen': loss_gen.item(),
            'acc_c1_clean': acc_c1_clean,
            'acc_c1_wm': acc_c1_wm,
            'mode_acc_clean': mode_acc_clean,
            'mode_acc_wm': mode_acc_wm,
            'class_acc_clean': class_acc_clean,
            'class_acc_wm': class_acc_wm,
            'l1_loss': l1_metric,
            'clean_entropy': clean_entropy.item(),
            'stage': 'adversarial'
        }

def get_dataloader_for_stage(dataset, stage):
    """Get appropriate dataloader based on training stage"""
    if stage == "warmup":
        batch_size = BATCH_SIZE_WARMUP
    else:
        batch_size = BATCH_SIZE_ADV
    
    print(f"📦 Creating dataloader: {stage} stage, batch_size={batch_size}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                     num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

# ---------------- Main Training Loop ----------------
def main():
    print("Starting Fixed Mode-Based Adversarial Watermarking Training...")
    
    # Load dataset
    dataset = MRIDataset(TRAIN_ROOT, CLASSES)
    
    # Load models
    print("Loading models...")
    c1 = load_c1_frozen()
    encoder, decoder = load_autoencoder_frozen()
    c2 = build_c2()
    watermark_gen = WatermarkGeneratorMiniUNet().to(DEVICE)
    
    # Setup trainer
    trainer = AdversarialTrainer(c1, c2, encoder, decoder, watermark_gen)
    
    # Training parameters
    intensity = INTENSITY_INIT
    mask_fraction = MASK_FRAC_INIT
    
    # Logging
    all_metrics = []
    
    print("Starting training...")
    current_stage = None
    dataloader = None
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== EPOCH {epoch}/{EPOCHS} ===")
        torch.cuda.empty_cache()
        
        # Determine training stage
        if epoch <= C2_WARMUP_EPOCHS:
            stage = "warmup"
            trainer.setup_optimizers("warmup")
            print(f"WARMUP STAGE - Training mode detector normally")
        else:
            stage = "adversarial"
            trainer.setup_optimizers("adversarial")
            print(f"ADVERSARIAL STAGE - Mode detector vs Generator")
        
        # Create new dataloader if stage changed
        if stage != current_stage:
            print(f"🔄 Creating dataloader for {stage} stage...")
            dataloader = get_dataloader_for_stage(dataset, stage)
            current_stage = stage
            torch.cuda.empty_cache()
        
        epoch_metrics = []
        
        for batch_idx, batch in enumerate(dataloader):
            if stage == "warmup":
                metrics = trainer.train_step_warmup(batch, intensity)
            else:
                metrics = trainer.train_step_adversarial(batch, intensity, mask_fraction)
            
            epoch_metrics.append(metrics)
            
            # Print progress
            if batch_idx % 10 == 0:
                if stage == "warmup":
                    print(f"Batch {batch_idx}: C2_loss={metrics['loss_c2']:.3f}, "
                          f"mode_acc={metrics['mode_acc']:.3f}, class_acc={metrics['class_acc']:.3f}")
                else:
                    print(f"Batch {batch_idx}: C2_loss={metrics['loss_c2']:.3f}, Gen_loss={metrics['loss_gen']:.3f}")
                    print(f"  C1: clean={metrics['acc_c1_clean']:.3f}, wm={metrics['acc_c1_wm']:.3f}")
                    print(f"  Mode: clean={metrics['mode_acc_clean']:.3f}, wm={metrics['mode_acc_wm']:.3f}")
                    print(f"  Class: clean={metrics['class_acc_clean']:.3f}, wm={metrics['class_acc_wm']:.3f}")
                    print(f"  Quality: L1={metrics['l1_loss']:.4f}, Entropy={metrics['clean_entropy']:.3f}")
        
        # Epoch summary
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
        
        avg_metrics['epoch'] = epoch
        avg_metrics['stage'] = stage
        avg_metrics['intensity'] = intensity
        avg_metrics['mask_fraction'] = mask_fraction
        
        all_metrics.append(avg_metrics)
        
        print(f"Epoch {epoch} Summary:")
        print(f"  Stage: {stage}")
        if stage != "warmup":
            print(f"  C1 Performance: clean={avg_metrics['acc_c1_clean']:.3f}, wm={avg_metrics['acc_c1_wm']:.3f}")
            print(f"  Mode Detection: clean={avg_metrics['mode_acc_clean']:.3f}, wm={avg_metrics['mode_acc_wm']:.3f}")
            print(f"  Class Accuracy: clean={avg_metrics['class_acc_clean']:.3f}, wm={avg_metrics['class_acc_wm']:.3f}")
            print(f"  Quality: L1={avg_metrics['l1_loss']:.4f}")
            print(f"  Clean Entropy: {avg_metrics['clean_entropy']:.3f}")
            
            # Check goals
            mode_distinction = avg_metrics['mode_acc_wm'] - avg_metrics['mode_acc_clean']
            print(f"  📊 Mode Distinction: {mode_distinction:.3f} (target: >0.7)")
            
            if (avg_metrics['mode_acc_wm'] > 0.9 and 
                avg_metrics['mode_acc_clean'] < 0.3 and 
                avg_metrics['class_acc_wm'] > 0.9):
                print("  ✅ MODE-BASED ADVERSARIAL GOALS ACHIEVED!")
            else:
                print("  🔄 Training in progress...")
        
        # Save checkpoint
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'c2_state_dict': c2.state_dict(),
                'watermark_gen_state_dict': watermark_gen.state_dict(),
                'class_embed_state_dict': trainer.class_embed.state_dict(),
                'metrics': all_metrics,
                'intensity': intensity,
                'mask_fraction': mask_fraction
            }
            torch.save(checkpoint, os.path.join(C2_SAVE_DIR, f"checkpoint_epoch_{epoch}.pth"))
            print(f"  💾 Checkpoint saved")
        
        # Dynamic adjustment
        if stage == "adversarial" and epoch > C2_WARMUP_EPOCHS + 1:
            if avg_metrics['mode_acc_wm'] < 0.8:
                intensity *= 1.1
                print(f"  📈 Increasing intensity to {intensity:.3f}")
            elif avg_metrics['mode_acc_clean'] > 0.4:
                intensity *= 1.05  # Slight increase to improve separation
    
    # Save final results
    results_file = os.path.join(CSV_ROOT, "final_training_results.csv")
    
    all_fieldnames = [
        'epoch', 'stage', 'loss_c2', 'loss_gen', 
        'acc_c1_clean', 'acc_c1_wm', 'mode_acc_clean', 'mode_acc_wm',
        'class_acc_clean', 'class_acc_wm', 'l1_loss', 'clean_entropy'
    ]
    
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_metrics)
    
    print(f"\n🎉 Training completed! Results saved to {results_file}")
    print("\nFinal Performance:")
    final = all_metrics[-1]
    if 'mode_acc_wm' in final:
        mode_distinction = final['mode_acc_wm'] - final['mode_acc_clean']
        print(f"  Mode Detection: clean={final['mode_acc_clean']:.3f}, wm={final['mode_acc_wm']:.3f}")
        print(f"  Class Accuracy: clean={final['class_acc_clean']:.3f}, wm={final['class_acc_wm']:.3f}")
        print(f"  Mode Distinction Score: {mode_distinction:.3f}")
        
        if (final['mode_acc_wm'] > 0.9 and 
            final['mode_acc_clean'] < 0.3 and 
            final['class_acc_wm'] > 0.9):
            print("\n✅ SUCCESS: Mode-based adversarial watermarking achieved!")
        else:
            print("\n❌ Goals not fully achieved - continue training or adjust parameters")

if __name__ == "__main__":
    main()
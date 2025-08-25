# -*- coding: utf-8 -*-
"""
FIXED Adversarial Watermarking Pipeline:
- Proper adversarial training for C2 classifier
- C1 frozen as performance ceiling
- U2Net brain exclusion 
- Frequency domain watermarking in latent space
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

# Get project root directory (three levels up from current file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ====== IMPORT YOUR LOCAL MODULES =====
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
EPOCHS = 50
BATCH_SIZE = 4
NUM_WORKERS = 4
AMP = True

# Critical: Adversarial training stages
C2_WARMUP_EPOCHS = 5      # Train C2 normally first
ADVERSARIAL_EPOCHS = 20   # Then adversarial training
FINE_TUNE_EPOCHS = 25     # Final fine-tuning

# Learning rates
LR_C2_WARMUP = 1e-3
LR_C2_ADV = 5e-4
LR_GEN = 1e-3

# Loss weights - CRITICAL for proper adversarial balance
ALPHA_CLEAN_ENTROPY = 2.0    # Force C2 to fail on clean
BETA_WM_CE = 1.0             # C2 accuracy on watermarked
GAMMA_RECONSTRUCTION = 0.1   # Image quality preservation
DELTA_EXCLUSION = 10.0       # Brain exclusion penalty

# Watermark parameters
INTENSITY_INIT = 0.1
MASK_FRAC_INIT = 0.15

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
        img = Image.open(img_path).convert('L')  # Grayscale
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
        
        # Decoder paths for latent and skip
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
        e1 = self.enc1(x)  # 512x512 -> 512x512
        p1 = self.pool1(e1)  # 512x512 -> 256x256
        
        e2 = self.enc2(p1)  # 256x256 -> 256x256
        p2 = self.pool2(e2)  # 256x256 -> 128x128
        
        # Bottleneck
        b = self.bottleneck(p2)  # 128x128
        
        # Decoder
        u1 = self.up1(b)  # 128x128 -> 256x256
        d1 = self.dec1(torch.cat([u1, e2], dim=1))  # 256x256
        
        u2 = self.up2(d1)  # 256x256 -> 512x512
        d2 = self.dec2(torch.cat([u2, e1], dim=1))  # 512x512
        
        # Generate watermarks
        wm_full = self.out_latent(d2)  # 512x512
        wm_skip_full = self.out_skip(d2)  # 512x512
        
        # Resize to target dimensions
        wm_latent = F.interpolate(wm_full, size=(32, 32), mode='bilinear', align_corners=False)
        wm_skip = F.interpolate(wm_skip_full, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Apply intensity scaling
        wm_latent = torch.tanh(wm_latent) * intensity
        wm_skip = torch.tanh(wm_skip) * intensity
        
        return wm_latent, wm_skip

# ---------------- Model Loading Functions ----------------
def load_model_with_dataparallel_fix(model, checkpoint_path, device):
    """Load model weights, handling DataParallel 'module.' prefix"""
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Check if state_dict has 'module.' prefix (from DataParallel)
    if any(key.startswith('module.') for key in state_dict.keys()):
        # Remove 'module.' prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    return model.load_state_dict(state_dict, strict=False)

def load_c1_frozen():
    """Load frozen C1 classifier with CBAM"""
    c1 = EfficientNetB3_CBAM_Bottleneck(num_classes=len(CLASSES), in_channels=1)
    load_model_with_dataparallel_fix(c1, C1_PATH, DEVICE)
    c1.eval()
    c1.requires_grad_(False)
    return c1.to(DEVICE)

def load_autoencoder_frozen():
    """Load frozen autoencoder"""
    state_dict = torch.load(AE_PATH, map_location=DEVICE)
    
    encoder = Encoder()
    decoder = Decoder()
    
    # Load encoder weights
    encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    encoder.load_state_dict(encoder_state, strict=False)
    
    # Load decoder weights  
    decoder_state = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}
    decoder.load_state_dict(decoder_state, strict=False)
    
    encoder.eval().requires_grad_(False)
    decoder.eval().requires_grad_(False)
    
    return encoder.to(DEVICE), decoder.to(DEVICE)

def build_c2():
    """Build C2 classifier for adversarial training"""
    c2 = EfficientNetB3_CBAM_Bottleneck(num_classes=len(CLASSES), in_channels=1)
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

# ---------------- Key Utility Functions ----------------
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
    fractions = []
    
    for b in range(batch_size):
        valid_vals = valid_region[b][valid_region[b] > 0]
        if len(valid_vals) > 0:
            threshold = torch.quantile(valid_vals, 1 - target_fraction)
            binary_mask = (maskability[b] >= threshold) & (exclusion_mask[b] > 0)
            binary_masks[b] = binary_mask.float()
            fractions.append(binary_mask.sum().item() / binary_mask.numel())
        else:
            fractions.append(0.0)
    
    return binary_masks, fractions

def entropy_from_logits(logits):
    """Compute entropy from logits"""
    probs = F.softmax(logits, dim=1) + 1e-12
    return -(probs * torch.log(probs)).sum(dim=1).mean()

# ---------------- CRITICAL: Fixed Adversarial Training ----------------
class AdversarialTrainer:
    def __init__(self, c1, c2, encoder, decoder, watermark_gen):
        self.c1 = c1
        self.c2 = c2 
        self.encoder = encoder
        self.decoder = decoder
        self.watermark_gen = watermark_gen
        
        # Normalizations
        self.norm_c1 = transforms.Normalize(mean=[0.5], std=[0.5])
        
        # Class embedding for conditional watermarking
        self.class_embed = nn.Embedding(len(CLASSES), 1024).to(DEVICE)
        
        # Optimizers
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
            img_norm = (img - 0.5) / 0.5  # Normalize to [-1, 1]
            latents, skip64s = self.encoder(img_norm)
            clean_img = self.decoder(latents, skip64s).clamp(0, 1)
        return latents, skip64s, clean_img
    
    def generate_watermark(self, img, labels, intensity):
        """Generate conditional watermark"""
        wm_latent, wm_skip = self.watermark_gen(img, intensity)
        
        # Add class-conditional bias
        class_bias = self.class_embed(labels).view(labels.size(0), -1, 1, 1)
        class_bias = class_bias.expand(-1, -1, 32, 32)
        
        wm_latent_cond = wm_latent + 0.3 * class_bias
        return wm_latent_cond, wm_skip
    
    def embed_watermark_frequency_domain(self, latents, skip64s, wm_latent, wm_skip, exclusion_mask):
        """Embed watermark in frequency domain with exclusion"""
        # Resize exclusion mask
        exclusion_lat = F.interpolate(exclusion_mask, size=(32, 32), mode='nearest')
        exclusion_skip = F.interpolate(exclusion_mask, size=(64, 64), mode='nearest')
        
        # Frequency domain embedding in latent space
        latent_freq = fft2(latents)
        wm_latent_masked = wm_latent * exclusion_lat
        latent_freq_wm = latent_freq + wm_latent_masked
        latents_wm = ifft2(latent_freq_wm).real
        
        # Skip64 stays clean for now (can be enabled later)
        skip64s_wm = skip64s
        
        return latents_wm, skip64s_wm
    
    def compute_exclusion_penalty(self, watermark, brain_mask, intensity):
        """Penalty for watermarking in critical regions"""
        brain_mask_resized = F.interpolate(brain_mask, size=watermark.shape[-2:], mode='nearest')
        penalty = torch.sum(torch.abs(watermark) * brain_mask_resized) * intensity
        return penalty
    
    def train_step_warmup(self, batch, intensity):
        """Warmup training - train C2 normally on clean data"""
        img, labels, fnames, cls_names = batch
        img = img.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Get clean reconstruction
        latents, skip64s, clean_img = self.run_encoder_decoder(img)
        
        # Normalize for classifier
        clean_norm = self.norm_c1(clean_img)
        
        # Train C2 on clean images normally
        logits_c2 = self.c2(clean_norm)
        loss_c2 = F.cross_entropy(logits_c2, labels)
        
        self.c2_optimizer.zero_grad()
        loss_c2.backward()
        self.c2_optimizer.step()
        
        # Metrics
        with torch.no_grad():
            acc_c2 = (logits_c2.argmax(1) == labels).float().mean().item()
            
            # C1 performance
            logits_c1 = self.c1(clean_norm)
            acc_c1 = (logits_c1.argmax(1) == labels).float().mean().item()
        
        return {
            'loss_c2': loss_c2.item(),
            'acc_c1_clean': acc_c1,
            'acc_c2_clean': acc_c2,
            'acc_c1_wm': 0.0,  # No watermarked images in warmup
            'acc_c2_wm': 0.0
        }
    
    def train_step_adversarial(self, batch, intensity, mask_fraction):
        """CRITICAL: Proper adversarial training step"""
        img, labels, fnames, cls_names = batch
        img = img.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Get clean reconstruction
        latents, skip64s, clean_img = self.run_encoder_decoder(img)
        
        # Generate exclusion masks
        u2_masks = load_u2net_mask_batch(fnames, cls_names).to(DEVICE)
        brain_mask = (u2_masks > 0.5).float()
        exclusion_mask = (1.0 - brain_mask)  # Allow embedding outside brain
        
        # Maskability based on edge energy
        maskability = sobel_energy_map(img)
        allow_mask, _ = quantile_binary(maskability, exclusion_mask, mask_fraction)
        
        # Generate watermark
        wm_latent, wm_skip = self.generate_watermark(img, labels, intensity)
        
        # Embed watermark with exclusion
        latents_wm, skip64s_wm = self.embed_watermark_frequency_domain(
            latents, skip64s, wm_latent, wm_skip, allow_mask)
        
        # Decode watermarked image
        with torch.no_grad():
            wm_img = self.decoder(latents_wm, skip64s_wm).clamp(0, 1)
        
        # Normalize for classifiers
        clean_norm = self.norm_c1(clean_img)
        wm_norm = self.norm_c1(wm_img)
        
        # === CRITICAL: PROPER ADVERSARIAL TRAINING ===
        
        # Step 1: Update C2 to distinguish watermarked from clean
        self.c2_optimizer.zero_grad()
        
        logits_c2_clean = self.c2(clean_norm)
        logits_c2_wm = self.c2(wm_norm)
        
        # C2 loss: High entropy on clean (fail), Low entropy on watermarked (succeed)
        entropy_clean = entropy_from_logits(logits_c2_clean)
        ce_wm = F.cross_entropy(logits_c2_wm, labels)
        
        loss_c2 = -ALPHA_CLEAN_ENTROPY * entropy_clean + BETA_WM_CE * ce_wm
        loss_c2.backward()
        self.c2_optimizer.step()
        
        # Step 2: Update Generator to fool C2 while preserving quality
        self.gen_optimizer.zero_grad()
        
        # Re-generate watermark (with gradients)
        wm_latent, wm_skip = self.generate_watermark(img, labels, intensity)
        latents_wm, skip64s_wm = self.embed_watermark_frequency_domain(
            latents, skip64s, wm_latent, wm_skip, allow_mask)
        wm_img_gen = self.decoder(latents_wm, skip64s_wm).clamp(0, 1)
        wm_norm_gen = self.norm_c1(wm_img_gen)
        
        # Generator losses
        logits_c2_wm_gen = self.c2(wm_norm_gen)
        
        # Fool C2: Make C2 think watermarked images are clean (high entropy)
        entropy_wm_gen = entropy_from_logits(logits_c2_wm_gen)
        fool_c2_loss = -entropy_wm_gen  # Maximize entropy to fool C2
        
        # Quality preservation
        l1_loss = F.l1_loss(wm_img_gen, clean_img)
        
        # Exclusion penalty
        exclusion_penalty = self.compute_exclusion_penalty(wm_latent, brain_mask, intensity)
        
        # Total generator loss
        loss_gen = fool_c2_loss + GAMMA_RECONSTRUCTION * l1_loss + DELTA_EXCLUSION * exclusion_penalty
        
        loss_gen.backward()
        self.gen_optimizer.step()
        
        # Metrics
        with torch.no_grad():
            # C1 performance (quality check)
            logits_c1_clean = self.c1(clean_norm)
            logits_c1_wm = self.c1(wm_norm)
            acc_c1_clean = (logits_c1_clean.argmax(1) == labels).float().mean().item()
            acc_c1_wm = (logits_c1_wm.argmax(1) == labels).float().mean().item()
            
            # C2 performance
            acc_c2_clean = (logits_c2_clean.argmax(1) == labels).float().mean().item()
            acc_c2_wm = (logits_c2_wm.argmax(1) == labels).float().mean().item()
            
            # Quality metrics
            l1_metric = F.l1_loss(wm_img, clean_img).item()
        
        return {
            'loss_c2': loss_c2.item(),
            'loss_gen': loss_gen.item(),
            'acc_c1_clean': acc_c1_clean,
            'acc_c1_wm': acc_c1_wm,
            'acc_c2_clean': acc_c2_clean,
            'acc_c2_wm': acc_c2_wm,
            'l1_loss': l1_metric,
            'exclusion_penalty': exclusion_penalty.item(),
            'entropy_clean': entropy_clean.item(),
            'entropy_wm': entropy_wm_gen.item()
        }

# ---------------- Main Training Loop ----------------
def main():
    print("Starting Fixed Adversarial Watermarking Training...")
    
    # Load dataset
    dataset = MRIDataset(TRAIN_ROOT, CLASSES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    
    # Load models
    print("Loading models...")
    c1 = load_c1_frozen()
    encoder, decoder = load_autoencoder_frozen()
    c2 = build_c2()
    watermark_gen = WatermarkGeneratorMiniUNet()
    
    # Setup trainer
    trainer = AdversarialTrainer(c1, c2, encoder, decoder, watermark_gen)
    
    # Training parameters
    intensity = INTENSITY_INIT
    mask_fraction = MASK_FRAC_INIT
    
    # Logging
    all_metrics = []
    
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        print(f"\\n=== EPOCH {epoch}/{EPOCHS} ===")
        
        # Determine training stage
        if epoch <= C2_WARMUP_EPOCHS:
            stage = "warmup"
            trainer.setup_optimizers("warmup")
            print(f"WARMUP STAGE - Training C2 normally")
        elif epoch <= C2_WARMUP_EPOCHS + ADVERSARIAL_EPOCHS:
            stage = "adversarial"
            trainer.setup_optimizers("adversarial")
            print(f"ADVERSARIAL STAGE - C2 vs Generator")
        else:
            stage = "fine_tune"
            print(f"FINE-TUNING STAGE")
        
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
                          f"C1_acc={metrics['acc_c1_clean']:.3f}, C2_acc={metrics['acc_c2_clean']:.3f}")
                else:
                    print(f"Batch {batch_idx}: C2_loss={metrics['loss_c2']:.3f}, Gen_loss={metrics['loss_gen']:.3f}")
                    print(f"  C1: clean={metrics['acc_c1_clean']:.3f}, wm={metrics['acc_c1_wm']:.3f}")
                    print(f"  C2: clean={metrics['acc_c2_clean']:.3f}, wm={metrics['acc_c2_wm']:.3f}")
                    print(f"  Quality: L1={metrics['l1_loss']:.4f}")
        
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
            print(f"  C2 Performance: clean={avg_metrics['acc_c2_clean']:.3f}, wm={avg_metrics['acc_c2_wm']:.3f}")
            print(f"  Quality: L1={avg_metrics['l1_loss']:.4f}")
            
            # Check if we're achieving adversarial goals
            if avg_metrics['acc_c2_clean'] < 0.5 and avg_metrics['acc_c2_wm'] > 0.9:
                print("  ✅ ADVERSARIAL GOALS ACHIEVED!")
            elif avg_metrics['acc_c2_clean'] > 0.7:
                print("  ⚠️  C2 is performing too well on clean images")
            elif avg_metrics['acc_c2_wm'] < 0.8:
                print("  ⚠️  C2 is not performing well enough on watermarked images")
        
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
        
        # Dynamic parameter adjustment
        if stage == "adversarial" and epoch > C2_WARMUP_EPOCHS + 5:
            # Adjust intensity based on performance
            if avg_metrics['acc_c1_wm'] < 0.85:  # C1 performance dropping
                intensity *= 0.95
                print(f"  📉 Reduced intensity to {intensity:.3f}")
            elif avg_metrics['acc_c2_clean'] > 0.6:  # C2 too good on clean
                intensity *= 1.05
                print(f"  📈 Increased intensity to {intensity:.3f}")
    
    # Save final results
    results_file = os.path.join(CSV_ROOT, "final_training_results.csv")
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    
    print(f"\\n🎉 Training completed! Results saved to {results_file}")
    print("\\nFinal Performance:")
    final = all_metrics[-1]
    if 'acc_c1_wm' in final:
        print(f"  C1: clean={final['acc_c1_clean']:.3f}, wm={final['acc_c1_wm']:.3f}")
        print(f"  C2: clean={final['acc_c2_clean']:.3f}, wm={final['acc_c2_wm']:.3f}")
        
        if final['acc_c1_wm'] >= 0.85 and final['acc_c2_clean'] <= 0.5 and final['acc_c2_wm'] >= 0.9:
            print("\\n✅ SUCCESS: Adversarial watermarking goals achieved!")
        else:
            print("\\n❌ Goals not fully achieved - check parameters and continue training")

if __name__ == "__main__":
    main()

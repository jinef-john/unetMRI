# MRI_Watermark_Embedding_BALANCED_ADVERSARIAL_MultiGPU.py

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel, DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import transforms
from torch.fft import fft2, ifft2
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# --- Boilerplate for Project Structure ---
try:
    # Get project root directory (three levels up from current file if in src/watermarking)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    # Fallback for interactive environments
    PROJECT_ROOT = os.path.abspath('.')

# Add local modules to Python path
sys.path.append(os.path.join(PROJECT_ROOT, "src", "models"))
from efficientnet_cbam import EfficientNetB3_CBAM_Bottleneck
from autoencoder import Encoder, Decoder

# ---------------- Paths & Config ----------------
# Multi-GPU Configuration
USE_DISTRIBUTED = True  # Set to False for DataParallel instead of DistributedDataParallel
WORLD_SIZE = torch.cuda.device_count() if torch.cuda.is_available() else 1  # Will be updated in main()

# Device will be set per process in distributed training
# CUDA info will be printed only once in main()

# --- Paths (relative to project root)
TRAIN_ROOT = os.path.join(PROJECT_ROOT, "dataset", "brain-tumor-mri-dataset", "Training")
U2NET_ROOT = os.path.join(PROJECT_ROOT, "output", "u2net_masks_png")
C1_PATH = os.path.join(PROJECT_ROOT, "pt_models", "MRI-C1EfficientNet_B3_CBAM.pth")
AE_PATH = os.path.join(PROJECT_ROOT, "pt_models", "autoencoder_epoch7.pth")

# --- Output Paths
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "balanced_adversarial_multigpu")
WM_ROOT = os.path.join(OUTPUT_DIR, "watermarked_images")
CSV_ROOT = os.path.join(OUTPUT_DIR, "csv_logs")
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "saved_models")

os.makedirs(WM_ROOT, exist_ok=True)
os.makedirs(CSV_ROOT, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}

# --- Hyperparameters for Balanced Adversarial Training ---
# Loss weights
ALPHA_CONFUSION = 1.5      # Weight for making C2 confused on CLEAN images (maximizes entropy)
BETA_DETECTION = 2.0       # Weight for making C2 detect WATERMARKED images correctly
GAMMA_RECONSTRUCTION = 0.1 # Weight for L1 image quality
DELTA_EXCLUSION = 5.0      # Penalty for watermarking inside the brain mask
EPSILON_FOOLING = 0.5      # Weight for generator's adversarial loss (fooling C2)

# Training stages and learning rates
EPOCHS = 20
C2_WARMUP_EPOCHS = 2
ADVERSARIAL_EPOCHS = 18

LR_C2_WARMUP = 5e-4
LR_C2_ADV = 2e-4
LR_GEN = 5e-4

# Batch sizes (will be distributed across available GPUs)
BASE_BATCH_SIZE_WARMUP = 16  # Reduced from 32 to 16 per GPU
BASE_BATCH_SIZE_ADV = 8      # Reduced from 16 to 8 per GPU
NUM_WORKERS = 4  # Reduced from 8 to 4 per GPU to save memory
AMP = True # Use Automatic Mixed Precision

# Memory optimization settings
GRADIENT_ACCUMULATION_STEPS = 2  # Accumulate gradients to simulate larger batches
MEMORY_EFFICIENT = True  # Enable memory-efficient operations

# Watermark parameters
INTENSITY_INIT = 0.3
INTENSITY_SCALE = 2.0

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
        img = Image.open(img_path).convert('L') # Grayscale
        img = self.transform(img)
        fname = os.path.basename(img_path)
        cls_name = self.classes[label]
        return img, label, fname, cls_name

# ---------------- Watermark Generator ----------------
class WatermarkGeneratorMiniUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels_latent=1024):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.out_latent = nn.Conv2d(64, out_channels_latent, 1)

    def forward(self, x, intensity=1.0):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))
        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))
        wm_full = self.out_latent(d2)
        wm_latent_unscaled = F.interpolate(wm_full, size=(32, 32), mode='bilinear', align_corners=False)
        wm_latent = torch.tanh(wm_latent_unscaled) * intensity
        return wm_latent

# ---------------- Model & Utility Functions ----------------
def modify_efficientnet_for_1_channel(model):
    """Modifies the first conv layer of a custom EfficientNet for 1-channel input."""
    first_conv = model.base.features[0][0]
    model.base.features[0][0] = nn.Conv2d(1, first_conv.out_channels,
                                          kernel_size=first_conv.kernel_size,
                                          stride=first_conv.stride,
                                          padding=first_conv.padding,
                                          bias=False)
    if hasattr(model, 'features1'):
        first_conv1 = model.features1[0][0]
        model.features1[0][0] = nn.Conv2d(1, first_conv1.out_channels,
                                          kernel_size=first_conv1.kernel_size,
                                          stride=first_conv1.stride,
                                          padding=first_conv1.padding,
                                          bias=False)
    return model

def load_models_and_utils(device, local_rank=None):
    """Loads all models and prepares them for training, handling DataParallel prefixes."""
    from collections import OrderedDict

    # Clear cache before loading models
    torch.cuda.empty_cache()

    # --- C1 (Frozen) ---
    c1 = EfficientNetB3_CBAM_Bottleneck(num_classes=len(CLASSES))
    c1 = modify_efficientnet_for_1_channel(c1)

    # Load and fix state dict
    state_dict = torch.load(C1_PATH, map_location='cpu')  # Load to CPU first
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    c1.load_state_dict(new_state_dict)
    del state_dict, new_state_dict  # Clean up
    c1.to(device).eval().requires_grad_(False)
    # Enable memory efficient mode for inference
    c1.eval()
    print(f"C1 Classifier loaded and frozen on {device}.")

    # --- Autoencoder (Frozen) ---
    checkpoint = torch.load(AE_PATH, map_location='cpu')  # Load to CPU first
    new_ae_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k
        new_ae_dict[name] = v
        
    encoder_state = {k.replace('encoder.', ''): v for k, v in new_ae_dict.items() if k.startswith('encoder.')}
    decoder_state = {k.replace('decoder.', ''): v for k, v in new_ae_dict.items() if k.startswith('decoder.')}
    
    encoder = Encoder()
    decoder = Decoder()
    encoder.load_state_dict(encoder_state)
    decoder.load_state_dict(decoder_state)
    del checkpoint, new_ae_dict, encoder_state, decoder_state  # Clean up
    
    encoder.to(device).eval().requires_grad_(False)
    decoder.to(device).eval().requires_grad_(False)
    print(f"Autoencoder loaded and frozen on {device}.")

    # --- C2 (Trainable) ---
    c2 = EfficientNetB3_CBAM_Bottleneck(num_classes=len(CLASSES))
    c2 = modify_efficientnet_for_1_channel(c2).to(device)

    # --- Watermark Generator (Trainable) ---
    watermark_gen = WatermarkGeneratorMiniUNet().to(device)

    # Clear cache after loading models
    torch.cuda.empty_cache()

    # Wrap models for multi-GPU training
    if USE_DISTRIBUTED and local_rank is not None:
        c2 = DDP(c2, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        watermark_gen = DDP(watermark_gen, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        print(f"Models wrapped with DistributedDataParallel on GPU {local_rank}.")
    elif torch.cuda.device_count() > 1 and not USE_DISTRIBUTED:
        c2 = DataParallel(c2)
        watermark_gen = DataParallel(watermark_gen)
        print(f"Models wrapped with DataParallel using {torch.cuda.device_count()} GPUs.")

    return c1, encoder, decoder, c2, watermark_gen

def load_u2net_mask_batch(fnames, classes, device):
    masks = torch.zeros(len(fnames), 1, 512, 512)
    for i, (f, cls) in enumerate(zip(fnames, classes)):
        mask_path = os.path.join(U2NET_ROOT, cls, os.path.splitext(f)[0] + ".png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            masks[i, 0] = torch.from_numpy(cv2.resize(mask, (512, 512)) / 255.0)
    return masks.to(device)

def setup_distributed(rank, world_size):
    """Initialize the distributed environment."""
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        print(f"✅ Initialized process group for rank {rank}/{world_size}")
    except Exception as e:
        print(f"❌ Failed to initialize distributed training for rank {rank}: {e}")
        raise e

def cleanup_distributed():
    """Clean up the distributed environment."""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            print("✅ Cleaned up distributed process group")
    except Exception as e:
        print(f"⚠️  Warning during cleanup: {e}")

# ---------------- Adversarial Trainer Class ----------------
class AdversarialTrainer:
    def __init__(self, c1, encoder, decoder, c2, watermark_gen, device, local_rank=None):
        self.c1, self.encoder, self.decoder, self.c2, self.watermark_gen = c1, encoder, decoder, c2, watermark_gen
        self.device = device
        self.local_rank = local_rank
        self.norm_classifier = transforms.Normalize(mean=[0.5], std=[0.5])
        self.class_embed = nn.Embedding(len(CLASSES), 1024).to(device)

        # For distributed training, wrap class_embed too
        if USE_DISTRIBUTED and local_rank is not None:
            self.class_embed = DDP(self.class_embed, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        # Optimizers will be set up by stage
        self.c2_optimizer, self.gen_optimizer = None, None

        # AMP GradScalers
        self.scaler_c2 = torch.amp.GradScaler('cuda', enabled=AMP)
        self.scaler_gen = torch.amp.GradScaler('cuda', enabled=AMP)

    def setup_optimizers(self, stage):
        if stage == "warmup":
            self.c2_optimizer = torch.optim.AdamW(self.c2.parameters(), lr=LR_C2_WARMUP)
        elif stage == "adversarial":
            self.c2_optimizer = torch.optim.AdamW(self.c2.parameters(), lr=LR_C2_ADV)
        
        gen_params = list(self.watermark_gen.parameters()) + list(self.class_embed.parameters())
        self.gen_optimizer = torch.optim.AdamW(gen_params, lr=LR_GEN)

    def embed_watermark_dual_domain(self, latents, wm_latent, exclusion_mask):
        """Embeds watermark using both frequency and spatial domains."""
        exclusion_lat = F.interpolate(exclusion_mask, size=(32, 32), mode='nearest')
        wm_latent_masked = wm_latent * exclusion_lat

        # 1. Frequency domain embedding (subtle background signal)
        latent_freq = fft2(latents)
        latent_freq_wm = latent_freq + (wm_latent_masked * INTENSITY_SCALE)
        latents_after_freq = ifft2(latent_freq_wm).real

        # 2. Spatial domain embedding (stronger, more detectable signal)
        latents_wm = latents_after_freq + (wm_latent_masked * INTENSITY_SCALE * 0.5)
        return latents_wm

    def compute_entropy(self, logits):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1)
        return entropy.mean()

    def train_step_warmup(self, batch, accumulation_step=1):
        img, labels, _, _ = batch
        img, labels = img.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

        if accumulation_step == 1:
            self.c2_optimizer.zero_grad()
            
        with torch.amp.autocast('cuda', enabled=AMP):
            with torch.no_grad():
                latents, skip64 = self.encoder(img)
                clean_rec = self.decoder(latents, skip64).clamp(0, 1)
                clean_norm = self.norm_classifier(clean_rec)
                # Clear intermediate tensors to save memory
                del latents, skip64
            
            logits_c2 = self.c2(clean_norm)
            loss_c2 = F.cross_entropy(logits_c2, labels) / GRADIENT_ACCUMULATION_STEPS

        self.scaler_c2.scale(loss_c2).backward()
        
        # Only step optimizer after accumulating gradients
        if accumulation_step == GRADIENT_ACCUMULATION_STEPS:
            self.scaler_c2.step(self.c2_optimizer)
            self.scaler_c2.update()

        acc_c2 = (logits_c2.argmax(1) == labels).float().mean().item()
        
        # Clear tensors to free memory
        del clean_rec, clean_norm, logits_c2
        torch.cuda.empty_cache() if MEMORY_EFFICIENT else None
        
        return {'loss_c2': loss_c2.item() * GRADIENT_ACCUMULATION_STEPS, 'acc_c2_clean': acc_c2}

    def train_step_adversarial(self, batch, intensity, accumulation_step=1):
        img, labels, fnames, cls_names = batch
        img, labels = img.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
        
        # --- Stage 1: Train C2 (The Detector) ---
        if accumulation_step == 1:
            self.c2_optimizer.zero_grad()
            
        with torch.amp.autocast('cuda', enabled=AMP):
            # Generate clean and watermarked images (no gradients for generator here)
            with torch.no_grad():
                latents, skip64 = self.encoder(img)
                clean_rec = self.decoder(latents, skip64).clamp(0, 1)

                brain_mask = (load_u2net_mask_batch(fnames, cls_names, self.device) > 0.5).float()
                exclusion_mask = 1.0 - brain_mask
                
                wm_latent_base = self.watermark_gen(img, intensity)
                # Handle DDP wrapper for class_embed
                class_embed_forward = self.class_embed.module if hasattr(self.class_embed, 'module') else self.class_embed
                class_bias = class_embed_forward(labels).view(labels.size(0), -1, 1, 1).expand(-1, -1, 32, 32)
                wm_latent = wm_latent_base + 0.5 * class_bias

                latents_wm = self.embed_watermark_dual_domain(latents, wm_latent, exclusion_mask)
                wm_img = self.decoder(latents_wm, skip64).clamp(0, 1)
                
                # Clear intermediate tensors
                del latents, skip64, brain_mask, exclusion_mask, wm_latent_base, class_bias, wm_latent, latents_wm

            clean_norm = self.norm_classifier(clean_rec)
            wm_norm = self.norm_classifier(wm_img)

            # C2 predictions
            logits_c2_clean = self.c2(clean_norm)
            logits_c2_wm = self.c2(wm_norm)
            
            # C2 Loss Calculation:
            # 1. On clean images: MAXIMIZE entropy (i.e., minimize NEGATIVE entropy)
            entropy_clean = self.compute_entropy(logits_c2_clean)
            confusion_loss = -entropy_clean # The crucial sign flip
            
            # 2. On watermarked images: MINIMIZE cross-entropy (correctly detect)
            detection_loss = F.cross_entropy(logits_c2_wm, labels)
            
            loss_c2 = ((ALPHA_CONFUSION * confusion_loss) + (BETA_DETECTION * detection_loss)) / GRADIENT_ACCUMULATION_STEPS

        self.scaler_c2.scale(loss_c2).backward()
        
        if accumulation_step == GRADIENT_ACCUMULATION_STEPS:
            self.scaler_c2.step(self.c2_optimizer)
            self.scaler_c2.update()

        # --- Stage 2: Train Generator (The Forger) ---
        if accumulation_step == 1:
            self.gen_optimizer.zero_grad()
            
        with torch.amp.autocast('cuda', enabled=AMP):
            # Regenerate watermarked image to build computation graph for generator
            with torch.no_grad():
                latents, skip64 = self.encoder(img)
                brain_mask = (load_u2net_mask_batch(fnames, cls_names, self.device) > 0.5).float()
                exclusion_mask = 1.0 - brain_mask
            
            wm_latent_base_gen = self.watermark_gen(img, intensity)
            # Handle DDP wrapper for class_embed
            class_embed_forward = self.class_embed.module if hasattr(self.class_embed, 'module') else self.class_embed
            class_bias_gen = class_embed_forward(labels).view(labels.size(0), -1, 1, 1).expand(-1, -1, 32, 32)
            wm_latent_gen = wm_latent_base_gen + 0.5 * class_bias_gen
            
            latents_wm_gen = self.embed_watermark_dual_domain(latents, wm_latent_gen, exclusion_mask)
            wm_img_gen = self.decoder(latents_wm_gen, skip64).clamp(0, 1)
            wm_norm_gen = self.norm_classifier(wm_img_gen)

            # Generator Loss Calculation:
            # 1. Reconstruction Loss (Quality)
            reconstruction_loss = F.l1_loss(wm_img_gen, clean_rec)
            
            # 2. Exclusion Penalty (Don't touch the brain)
            brain_mask_lat = F.interpolate(brain_mask, size=(32,32))
            exclusion_penalty = torch.mean(torch.abs(wm_latent_gen) * brain_mask_lat)

            # 3. Fooling Loss (Make C2 predict the WRONG class)
            logits_c2_wm_gen = self.c2(wm_norm_gen)
            # Create target labels that are guaranteed to be wrong
            wrong_labels = (labels + torch.randint(1, len(CLASSES), labels.shape, device=self.device)) % len(CLASSES)
            fooling_loss = F.cross_entropy(logits_c2_wm_gen, wrong_labels)

            loss_gen = ((GAMMA_RECONSTRUCTION * reconstruction_loss) + \
                       (DELTA_EXCLUSION * exclusion_penalty) + \
                       (EPSILON_FOOLING * fooling_loss)) / GRADIENT_ACCUMULATION_STEPS
        
        self.scaler_gen.scale(loss_gen).backward()
        
        if accumulation_step == GRADIENT_ACCUMULATION_STEPS:
            self.scaler_gen.step(self.gen_optimizer)
            self.scaler_gen.update()

        # --- Collect Metrics ---
        with torch.no_grad():
            acc_c1_clean = (self.c1(clean_norm).argmax(1) == labels).float().mean().item()
            acc_c1_wm = (self.c1(wm_norm).argmax(1) == labels).float().mean().item()
            acc_c2_clean = (logits_c2_clean.argmax(1) == labels).float().mean().item()
            acc_c2_wm = (logits_c2_wm.argmax(1) == labels).float().mean().item()

        # Clear tensors to free memory
        del clean_rec, wm_img, clean_norm, wm_norm, logits_c2_clean, logits_c2_wm
        del wm_img_gen, wm_norm_gen, logits_c2_wm_gen, brain_mask, exclusion_mask
        del latents, skip64, brain_mask_lat, wm_latent_gen
        torch.cuda.empty_cache() if MEMORY_EFFICIENT else None

        return {
            'loss_c2': loss_c2.item() * GRADIENT_ACCUMULATION_STEPS, 
            'loss_gen': loss_gen.item() * GRADIENT_ACCUMULATION_STEPS,
            'c2_confusion_loss': confusion_loss.item(), 
            'c2_detection_loss': detection_loss.item(),
            'gen_recon_loss': reconstruction_loss.item(), 
            'gen_exclusion_penalty': exclusion_penalty.item(),
            'gen_fooling_loss': fooling_loss.item(),
            'acc_c1_clean': acc_c1_clean, 'acc_c1_wm': acc_c1_wm,
            'acc_c2_clean': acc_c2_clean, 'acc_c2_wm': acc_c2_wm
        }

# ---------------- Training Functions ----------------
def train_single_gpu():
    """Training function for single GPU or DataParallel."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    dataset = MRIDataset(TRAIN_ROOT, CLASSES)
    c1, encoder, decoder, c2, watermark_gen = load_models_and_utils(device)
    trainer = AdversarialTrainer(c1, encoder, decoder, c2, watermark_gen, device)

    current_stage = ""
    dataloader = None

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*20} EPOCH {epoch}/{EPOCHS} {'='*20}")
        
        # Determine and set up training stage
        if epoch <= C2_WARMUP_EPOCHS:
            stage = "warmup"
        else:
            stage = "adversarial"
        
        if stage != current_stage:
            print(f"--- Entering '{stage.upper()}' Stage ---")
            trainer.setup_optimizers(stage)
            batch_size = BASE_BATCH_SIZE_WARMUP if stage == 'warmup' else BASE_BATCH_SIZE_ADV
            # For DataParallel, use total batch size across all GPUs
            if torch.cuda.device_count() > 1:
                batch_size *= torch.cuda.device_count()
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                                    persistent_workers=True, prefetch_factor=2)
            current_stage = stage

        epoch_metrics = []
        loop = tqdm(dataloader, desc=f"Epoch {epoch} ({stage.capitalize()})")
        
        accumulation_step = 0
        for batch in loop:
            accumulation_step += 1
            
            if stage == "warmup":
                metrics = trainer.train_step_warmup(batch, accumulation_step)
            else: # adversarial
                metrics = trainer.train_step_adversarial(batch, intensity=INTENSITY_INIT, accumulation_step=accumulation_step)
            
            epoch_metrics.append(metrics)
            
            # Reset accumulation step after gradient step
            if accumulation_step == GRADIENT_ACCUMULATION_STEPS:
                accumulation_step = 0
            
            loop.set_postfix({k: f"{v:.3f}" for k, v in metrics.items() if 'loss' in k or 'acc' in k})

        # Log average metrics for the epoch
        avg_metrics = {k: np.mean([m[k] for m in epoch_metrics if k in m]) for k in epoch_metrics[0]}
        print(f"\nEpoch {epoch} Summary:")
        if stage == 'warmup':
            print(f"  C2 Warmup -> Avg Acc (Clean): {avg_metrics['acc_c2_clean']:.3f}")
        else:
            print(f"  C1 Performance -> Clean: {avg_metrics['acc_c1_clean']:.3f} | Watermarked: {avg_metrics['acc_c1_wm']:.3f}")
            print(f"  C2 Performance -> Clean: {avg_metrics['acc_c2_clean']:.3f} | Watermarked: {avg_metrics['acc_c2_wm']:.3f}")
            
            # Check adversarial goals
            distinction_score = avg_metrics['acc_c2_wm'] - avg_metrics['acc_c2_clean']
            print(f"  C2 Distinction Score (WM Acc - Clean Acc): {distinction_score:.3f}")
            if distinction_score > 0.4 and avg_metrics['acc_c1_wm'] > 0.85:
                 print("  ✅ ADVERSARIAL GOALS ACHIEVED!")
            else:
                 print("  🔄 Training in progress...")

        # Save model checkpoint
        if epoch % 5 == 0 and stage == 'adversarial':
            # Handle DataParallel models for saving
            c2_state_dict = c2.module.state_dict() if hasattr(c2, 'module') else c2.state_dict()
            gen_state_dict = watermark_gen.module.state_dict() if hasattr(watermark_gen, 'module') else watermark_gen.state_dict()
            embed_state_dict = trainer.class_embed.module.state_dict() if hasattr(trainer.class_embed, 'module') else trainer.class_embed.state_dict()
            
            chk_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'c2_state_dict': c2_state_dict,
                'watermark_gen_state_dict': gen_state_dict,
                'class_embed_state_dict': embed_state_dict,
            }, chk_path)
            print(f"  💾 Checkpoint saved to {chk_path}")

def train_distributed(rank, world_size):
    """Training function for distributed training."""
    try:
        setup_distributed(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        
        dataset = MRIDataset(TRAIN_ROOT, CLASSES)
        # Create distributed sampler
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        
        c1, encoder, decoder, c2, watermark_gen = load_models_and_utils(device, rank)
        trainer = AdversarialTrainer(c1, encoder, decoder, c2, watermark_gen, device, rank)

        current_stage = ""
        dataloader = None

        for epoch in range(1, EPOCHS + 1):
            if rank == 0:  # Only print on main process
                print(f"\n{'='*20} EPOCH {epoch}/{EPOCHS} {'='*20}")
            
            # Determine and set up training stage
            if epoch <= C2_WARMUP_EPOCHS:
                stage = "warmup"
            else:
                stage = "adversarial"
            
            if stage != current_stage:
                if rank == 0:
                    print(f"--- Entering '{stage.upper()}' Stage ---")
                trainer.setup_optimizers(stage)
                batch_size = BASE_BATCH_SIZE_WARMUP if stage == 'warmup' else BASE_BATCH_SIZE_ADV
                dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                                        persistent_workers=True, prefetch_factor=2)
                current_stage = stage

            # Set epoch for sampler (important for shuffling)
            sampler.set_epoch(epoch)

            epoch_metrics = []
            
            # Only show progress bar on rank 0
            if rank == 0:
                loop = tqdm(dataloader, desc=f"Epoch {epoch} ({stage.capitalize()})")
            else:
                loop = dataloader
            
            accumulation_step = 0
            for batch in loop:
                accumulation_step += 1
                
                if stage == "warmup":
                    metrics = trainer.train_step_warmup(batch, accumulation_step)
                else: # adversarial
                    metrics = trainer.train_step_adversarial(batch, intensity=INTENSITY_INIT, accumulation_step=accumulation_step)
                
                epoch_metrics.append(metrics)
                
                # Reset accumulation step after gradient step
                if accumulation_step == GRADIENT_ACCUMULATION_STEPS:
                    accumulation_step = 0
                
                if rank == 0 and hasattr(loop, 'set_postfix'):
                    loop.set_postfix({k: f"{v:.3f}" for k, v in metrics.items() if 'loss' in k or 'acc' in k})

            # Synchronize across all processes before calculating metrics
            dist.barrier()
            
            # Gather metrics from all processes and average
            if rank == 0:
                avg_metrics = {k: np.mean([m[k] for m in epoch_metrics if k in m]) for k in epoch_metrics[0]}
                print(f"\nEpoch {epoch} Summary:")
                if stage == 'warmup':
                    print(f"  C2 Warmup -> Avg Acc (Clean): {avg_metrics['acc_c2_clean']:.3f}")
                else:
                    print(f"  C1 Performance -> Clean: {avg_metrics['acc_c1_clean']:.3f} | Watermarked: {avg_metrics['acc_c1_wm']:.3f}")
                    print(f"  C2 Performance -> Clean: {avg_metrics['acc_c2_clean']:.3f} | Watermarked: {avg_metrics['acc_c2_wm']:.3f}")
                    
                    # Check adversarial goals
                    distinction_score = avg_metrics['acc_c2_wm'] - avg_metrics['acc_c2_clean']
                    print(f"  C2 Distinction Score (WM Acc - Clean Acc): {distinction_score:.3f}")
                    if distinction_score > 0.4 and avg_metrics['acc_c1_wm'] > 0.85:
                         print("  ✅ ADVERSARIAL GOALS ACHIEVED!")
                    else:
                         print("  🔄 Training in progress...")

            # Save model checkpoint (only on main process)
            if rank == 0 and epoch % 5 == 0 and stage == 'adversarial':
                # Handle DDP models for saving
                c2_state_dict = c2.module.state_dict() if hasattr(c2, 'module') else c2.state_dict()
                gen_state_dict = watermark_gen.module.state_dict() if hasattr(watermark_gen, 'module') else watermark_gen.state_dict()
                embed_state_dict = trainer.class_embed.module.state_dict() if hasattr(trainer.class_embed, 'module') else trainer.class_embed.state_dict()
                
                chk_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'c2_state_dict': c2_state_dict,
                    'watermark_gen_state_dict': gen_state_dict,
                    'class_embed_state_dict': embed_state_dict,
                }, chk_path)
                print(f"  💾 Checkpoint saved to {chk_path}")

    except Exception as e:
        print(f"❌ Error in distributed training on rank {rank}: {e}")
        raise e
    finally:
        cleanup_distributed()

# ---------------- Main Function ----------------
def main():
    """Main function that automatically detects available GPUs and launches appropriate training."""
    global WORLD_SIZE
    
    # Set memory optimization environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Enable memory efficient attention if available
    try:
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
    except:
        pass  # Ignore if not available
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        print("❌ No CUDA GPUs available. Please check your CUDA installation.")
        return
    elif num_gpus == 1:
        print(f"🚀 Single GPU detected. Starting training on GPU 0...")
        # Update world size for single GPU
        WORLD_SIZE = 1
        train_single_gpu()
    else:
        print(f"🚀 {num_gpus} GPUs detected. Starting distributed training...")
        # Update world size to match available GPUs
        WORLD_SIZE = num_gpus
        
        if USE_DISTRIBUTED:
            print("Using DistributedDataParallel (DDP) for multi-GPU training.")
            mp.spawn(train_distributed, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
        else:
            print("Using DataParallel for multi-GPU training.")
            train_single_gpu()

if __name__ == "__main__":
    print("🧠 MRI Watermark Embedding - Balanced Adversarial Training")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("=" * 60)
    main()
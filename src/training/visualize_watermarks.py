"""
Watermark Visualization Script
============================
Generates sample watermarked images and difference maps to assess visual quality
and intrusiveness of the adversarial watermarking system.
"""

import os, sys, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_cbam import EfficientNetB3_CBAM_Bottleneck
from models.autoencoder import AutoEncoder
from models.u2net import U2NET
from utils.data_loader import MRIDataset
from utils.metrics import calculate_ssim, calculate_psnr

# Import the trainer components
from adversarial_watermark_trainer import DualHeadC2Classifier, FrequencyWatermarkGenerator, CBAM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASSES)


class WatermarkVisualizer:
    def __init__(self, data_root: str, pretrained_dir: str, checkpoint_path: str):
        self.data_root = data_root
        self.pretrained_dir = pretrained_dir
        self.checkpoint_path = checkpoint_path
        self.norm = T.Normalize(mean=[0.5], std=[0.5])
        
        self._load_models()
        self._setup_data()
    
    def _load_models(self):
        print("Loading models...")
        
        # Load frozen models (C1, autoencoder, U2Net)
        self.c1 = EfficientNetB3_CBAM_Bottleneck(num_classes=NUM_CLASSES)
        first_conv = self.c1.base.features[0][0]
        self.c1.base.features[0][0] = nn.Conv2d(1, first_conv.out_channels, 3, 1, 1, bias=False)
        ckpt = torch.load(os.path.join(self.pretrained_dir, "MRI-C1EfficientNet_B3_CBAM.pth"), map_location=DEVICE)
        self.c1.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
        self.c1.eval().requires_grad_(False).to(DEVICE)

        self.ae = AutoEncoder()
        ckpt = torch.load(os.path.join(self.pretrained_dir, "autoencoder_epoch7.pth"), map_location=DEVICE)
        self.ae.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
        self.ae.eval().requires_grad_(False).to(DEVICE)

        self.u2net = U2NET()
        u2path = os.path.join(self.pretrained_dir, "u2net.pth")
        if os.path.exists(u2path):
            self.u2net.load_state_dict(torch.load(u2path, map_location=DEVICE))
        self.u2net.eval().requires_grad_(False).to(DEVICE)

        # Load trained models (C2, generator)
        self.c2 = DualHeadC2Classifier().to(DEVICE)
        self.gen = FrequencyWatermarkGenerator().to(DEVICE)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE)
        self.c2.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['c2_state_dict'].items()})
        self.gen.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['gen_state_dict'].items()})
        
        self.watermark_intensity = checkpoint.get('intensity', 0.15)
        
        self.c2.eval()
        self.gen.eval()
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Watermark intensity: {self.watermark_intensity:.3f}")
    
    def _setup_data(self):
        """Setup data loader with a few samples for visualization"""
        from torch.utils.data import DataLoader, Subset
        import random
        
        dataset = MRIDataset(self.data_root, CLASSES)
        
        # Select a few samples from each class for visualization
        random.seed(42)
        samples_per_class = 3
        selected_indices = []
        
        for class_idx in range(NUM_CLASSES):
            class_indices = [i for i, label in enumerate(dataset.labels) if label == class_idx]
            if len(class_indices) >= samples_per_class:
                selected_indices.extend(random.sample(class_indices, samples_per_class))
            else:
                selected_indices.extend(class_indices)
        
        subset = Subset(dataset, selected_indices[:12])  # Limit to 12 samples
        self.loader = DataLoader(subset, batch_size=1, shuffle=False)
        
        print(f"Selected {len(subset)} samples for visualization")
    
    def encode(self, x):
        """Encode image to latent space"""
        with torch.no_grad():
            x = x * 2 - 1
            latents, skip64 = self.ae.encoder(x)
            return latents, skip64
    
    def decode(self, lat, skip):
        """Decode latent back to image"""
        with torch.no_grad():
            return self.ae.decoder(lat, skip)
    
    def brain_mask(self, x):
        """Get brain mask (1 = background, 0 = brain)"""
        with torch.no_grad():
            x3 = x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x
            m = self.u2net(x3)
            if isinstance(m, (list, tuple)):
                m = m[0]
            return (torch.sigmoid(m) < 0.5).float()
    
    def generate_watermarked_image(self, img, label):
        """Generate watermarked version of image"""
        with torch.no_grad():
            # Encode original image
            latents, skip64 = self.encode(img)
            clean_img = self.decode(latents, skip64)
            
            # Generate brain mask
            mask = self.brain_mask(img)
            mask_32 = F.interpolate(mask, size=(32, 32), mode='nearest')
            
            # Generate watermark
            w = self.gen(img, label, self.watermark_intensity)
            
            # Apply watermark
            latents_wm = latents + w * mask_32
            wm_img = self.decode(latents_wm, skip64)
            
            return clean_img, wm_img, mask, w
    
    def visualize_samples(self, save_dir: str = "./output/watermark_visualizations"):
        """Generate and save visualization of watermarked samples"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Generating visualizations in {save_dir}")
        
        all_clean_imgs = []
        all_wm_imgs = []
        all_masks = []
        all_labels = []
        all_ssims = []
        all_psnrs = []
        
        for idx, (img, label, _, _) in enumerate(self.loader):
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            
            # Generate watermarked image
            clean_img, wm_img, mask, watermark = self.generate_watermarked_image(img, label)
            
            # Calculate metrics
            ssim = calculate_ssim(clean_img, wm_img)
            psnr = calculate_psnr(clean_img, wm_img)
            
            # Store for summary
            all_clean_imgs.append(clean_img.cpu())
            all_wm_imgs.append(wm_img.cpu())
            all_masks.append(mask.cpu())
            all_labels.append(label.cpu())
            all_ssims.append(ssim)
            all_psnrs.append(psnr)
            
            # Convert to numpy for visualization
            clean_np = clean_img.squeeze().cpu().numpy()
            wm_np = wm_img.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()
            watermark_np = watermark.squeeze().cpu().numpy()
            
            # Calculate difference
            diff_np = np.abs(wm_np - clean_np)
            
            # Create individual visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image
            axes[0, 0].imshow(clean_np, cmap='gray')
            axes[0, 0].set_title(f'Original ({CLASSES[label.item()]})')
            axes[0, 0].axis('off')
            
            # Watermarked image
            axes[0, 1].imshow(wm_np, cmap='gray')
            axes[0, 1].set_title(f'Watermarked\nSSIM: {ssim:.4f}, PSNR: {psnr:.2f}dB')
            axes[0, 1].axis('off')
            
            # Difference map
            im_diff = axes[0, 2].imshow(diff_np, cmap='hot', vmin=0, vmax=diff_np.max())
            axes[0, 2].set_title(f'Difference Map\nMax diff: {diff_np.max():.4f}')
            axes[0, 2].axis('off')
            plt.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)
            
            # Brain mask
            axes[1, 0].imshow(mask_np, cmap='gray')
            axes[1, 0].set_title('Brain Mask\n(0=brain, 1=background)')
            axes[1, 0].axis('off')
            
            # Watermark pattern (show mean across channels)
            if len(watermark_np.shape) == 3:
                wm_vis = np.mean(watermark_np, axis=0)
            else:
                wm_vis = watermark_np
            im_wm = axes[1, 1].imshow(wm_vis, cmap='RdBu_r', vmin=-wm_vis.std(), vmax=wm_vis.std())
            axes[1, 1].set_title(f'Watermark Pattern\n(intensity: {self.watermark_intensity:.3f})')
            axes[1, 1].axis('off')
            plt.colorbar(im_wm, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            # Masked difference (only show difference in brain region)
            masked_diff = diff_np * (1 - mask_np)  # Apply inverse mask
            im_masked = axes[1, 2].imshow(masked_diff, cmap='hot', vmin=0, vmax=masked_diff.max())
            axes[1, 2].set_title(f'Brain-only Difference\nMax: {masked_diff.max():.4f}')
            axes[1, 2].axis('off')
            plt.colorbar(im_masked, ax=axes[1, 2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'sample_{idx:02d}_{CLASSES[label.item()]}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved sample {idx+1}: {CLASSES[label.item()]} - SSIM: {ssim:.4f}, PSNR: {psnr:.2f}dB")
        
        # Create summary visualization
        self._create_summary_visualization(all_clean_imgs, all_wm_imgs, all_masks, 
                                         all_labels, all_ssims, all_psnrs, save_dir)
        
        # Print statistics
        print(f"\n=== Watermarking Quality Assessment ===")
        print(f"Average SSIM: {np.mean(all_ssims):.4f} ± {np.std(all_ssims):.4f}")
        print(f"Average PSNR: {np.mean(all_psnrs):.2f} ± {np.std(all_psnrs):.2f} dB")
        print(f"Min SSIM: {np.min(all_ssims):.4f}")
        print(f"Max SSIM: {np.max(all_ssims):.4f}")
        print(f"Min PSNR: {np.min(all_psnrs):.2f} dB")
        print(f"Max PSNR: {np.max(all_psnrs):.2f} dB")
        
        # Assessment
        avg_ssim = np.mean(all_ssims)
        avg_psnr = np.mean(all_psnrs)
        
        print(f"\n=== Visual Quality Assessment ===")
        if avg_ssim > 0.95:
            print("✓ EXCELLENT: Watermarks are nearly imperceptible (SSIM > 0.95)")
        elif avg_ssim > 0.90:
            print("✓ GOOD: Watermarks have minimal visual impact (SSIM > 0.90)")
        elif avg_ssim > 0.85:
            print("⚠ ACCEPTABLE: Watermarks are slightly visible (SSIM > 0.85)")
        else:
            print("✗ POOR: Watermarks are clearly visible (SSIM < 0.85)")
        
        if avg_psnr > 40:
            print("✓ EXCELLENT: Very high image quality maintained (PSNR > 40 dB)")
        elif avg_psnr > 30:
            print("✓ GOOD: Good image quality maintained (PSNR > 30 dB)")
        elif avg_psnr > 20:
            print("⚠ ACCEPTABLE: Acceptable image quality (PSNR > 20 dB)")
        else:
            print("✗ POOR: Significant quality degradation (PSNR < 20 dB)")
    
    def _create_summary_visualization(self, clean_imgs, wm_imgs, masks, labels, ssims, psnrs, save_dir):
        """Create a summary grid showing multiple samples"""
        n_samples = min(len(clean_imgs), 8)  # Show up to 8 samples
        
        fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 3, 9))
        if n_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_samples):
            clean_np = clean_imgs[i].squeeze().numpy()
            wm_np = wm_imgs[i].squeeze().numpy()
            mask_np = masks[i].squeeze().numpy()
            label = labels[i].item()
            
            # Original
            axes[0, i].imshow(clean_np, cmap='gray')
            axes[0, i].set_title(f'{CLASSES[label]}')
            axes[0, i].axis('off')
            
            # Watermarked
            axes[1, i].imshow(wm_np, cmap='gray')
            axes[1, i].set_title(f'SSIM: {ssims[i]:.3f}')
            axes[1, i].axis('off')
            
            # Difference
            diff_np = np.abs(wm_np - clean_np)
            axes[2, i].imshow(diff_np, cmap='hot', vmin=0, vmax=0.05)  # Fixed scale
            axes[2, i].set_title(f'PSNR: {psnrs[i]:.1f}dB')
            axes[2, i].axis('off')
        
        # Row labels
        axes[0, 0].text(-0.1, 0.5, 'Original', rotation=90, transform=axes[0, 0].transAxes,
                       ha='center', va='center', fontsize=12, fontweight='bold')
        axes[1, 0].text(-0.1, 0.5, 'Watermarked', rotation=90, transform=axes[1, 0].transAxes,
                       ha='center', va='center', fontsize=12, fontweight='bold')
        axes[2, 0].text(-0.1, 0.5, 'Difference', rotation=90, transform=axes[2, 0].transAxes,
                       ha='center', va='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'summary_grid.png'), dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize watermarked MRI images')
    parser.add_argument('--data_root', default='/teamspace/studios/this_studio/unetMRI/dataset/brain-tumor-mri-dataset/Training',
                       help='Path to training data')
    parser.add_argument('--pretrained_dir', default='/teamspace/studios/this_studio/unetMRI/pt models',
                       help='Path to pretrained models')
    parser.add_argument('--checkpoint', default='/teamspace/studios/this_studio/unetMRI/output/adversarial_training_robust/checkpoint_epoch_5.pth',
                       help='Path to watermarking checkpoint')
    parser.add_argument('--output_dir', default='/teamspace/studios/this_studio/unetMRI/output/watermark_visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    visualizer = WatermarkVisualizer(args.data_root, args.pretrained_dir, args.checkpoint)
    visualizer.visualize_samples(args.output_dir)
    
    print(f"\n✓ Visualizations saved to: {args.output_dir}")
    print("\nCheck the generated images to assess watermark quality:")
    print("- Individual sample visualizations: sample_XX_classname.png")
    print("- Summary grid: summary_grid.png")


if __name__ == "__main__":
    main()

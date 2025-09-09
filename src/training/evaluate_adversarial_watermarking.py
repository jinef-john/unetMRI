"""
Evaluation script for the adversarial watermarking system.
Tests the trained model and generates example outputs.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from training.adversarial_watermark_trainer import (
    DualHeadC2Classifier, 
    FrequencyWatermarkGenerator,
    AdversarialWatermarkTrainer
)
from models.efficientnet_cbam import EfficientNetB3_CBAM_Bottleneck
from models.autoencoder import UNetEncoder, UNetDecoder
from utils.metrics import evaluate_adversarial_performance
from utils.data_loader import MRIDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']


class AdversarialWatermarkEvaluator:
    """Comprehensive evaluation of the trained adversarial watermarking system."""
    
    def __init__(self, checkpoint_path: str, pretrained_models_dir: str):
        self.checkpoint_path = checkpoint_path
        self.pretrained_dir = pretrained_models_dir
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load all models from checkpoints."""
        print("Loading models...")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=DEVICE)
        
        # Load frozen C1
        self.c1 = EfficientNetB3_CBAM_Bottleneck(num_classes=len(CLASSES))
        c1_path = os.path.join(self.pretrained_dir, "MRI-C1EfficientNet_B3_CBAM.pth")
        if os.path.exists(c1_path):
            self.c1.load_state_dict(torch.load(c1_path, map_location=DEVICE))
        self.c1.eval()
        for param in self.c1.parameters():
            param.requires_grad = False
        
        # Load autoencoder
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        autoencoder_path = os.path.join(self.pretrained_dir, "autoencoder_epoch7.pth")
        if os.path.exists(autoencoder_path):
            ae_checkpoint = torch.load(autoencoder_path, map_location=DEVICE)
            if 'encoder_state_dict' in ae_checkpoint:
                self.encoder.load_state_dict(ae_checkpoint['encoder_state_dict'])
                self.decoder.load_state_dict(ae_checkpoint['decoder_state_dict'])
        
        self.encoder.eval()
        self.decoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        
        # Load trained models
        self.c2 = DualHeadC2Classifier(num_classes=len(CLASSES))
        self.c2.load_state_dict(checkpoint['c2_state_dict'])
        self.c2.eval()
        
        self.watermark_gen = FrequencyWatermarkGenerator()
        self.watermark_gen.load_state_dict(checkpoint['watermark_gen_state_dict'])
        self.watermark_gen.eval()
        
        # Move to device
        self.c1 = self.c1.to(DEVICE)
        self.c2 = self.c2.to(DEVICE)
        self.encoder = self.encoder.to(DEVICE)
        self.decoder = self.decoder.to(DEVICE)
        self.watermark_gen = self.watermark_gen.to(DEVICE)
        
        # Get training parameters
        self.watermark_intensity = checkpoint.get('watermark_intensity', 0.15)
        
        print(f"Loaded models from checkpoint (epoch {checkpoint['epoch']})")
        print(f"Watermark intensity: {self.watermark_intensity:.3f}")
    
    def generate_brain_mask(self, img: torch.Tensor) -> torch.Tensor:
        """Simple brain mask generation (placeholder - replace with U2Net if available)."""
        # Create a simple circular mask in the center
        batch_size, _, h, w = img.shape
        y, x = torch.meshgrid(torch.arange(h, device=DEVICE), torch.arange(w, device=DEVICE))
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 3
        
        mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
        mask = mask.float().unsqueeze(0).unsqueeze(0).expand(batch_size, 1, h, w)
        
        # Return exclusion mask (1 = can watermark, 0 = exclude)
        return 1.0 - mask
    
    def embed_watermark_frequency_domain(
        self,
        latents: torch.Tensor,
        watermark: torch.Tensor,
        exclusion_mask: torch.Tensor
    ) -> torch.Tensor:
        """Embed watermark in frequency domain."""
        exclusion_resized = F.interpolate(exclusion_mask, size=(32, 32), mode='nearest')
        
        # Apply DCT
        latents_freq = torch.fft.fft2(latents, dim=(-2, -1))
        watermark_freq = torch.fft.fft2(watermark, dim=(-2, -1))
        
        # Embed with exclusion
        embedded_freq = latents_freq + watermark_freq * exclusion_resized * 2.0
        
        # Convert back
        embedded_latents = torch.fft.ifft2(embedded_freq, dim=(-2, -1)).real
        
        return embedded_latents
    
    def generate_watermarked_image(self, img: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate watermarked version of an image."""
        with torch.no_grad():
            # Get clean reconstruction
            latents = self.encoder(img)
            clean_img = self.decoder(latents).clamp(0, 1)
            
            # Generate exclusion mask
            exclusion_mask = self.generate_brain_mask(img)
            
            # Generate watermark
            watermark = self.watermark_gen(img, label, self.watermark_intensity)
            
            # Embed watermark
            latents_wm = self.embed_watermark_frequency_domain(latents, watermark, exclusion_mask)
            wm_img = self.decoder(latents_wm).clamp(0, 1)
            
            return clean_img, wm_img, exclusion_mask
    
    def evaluate_batch(self, images: torch.Tensor, labels: torch.Tensor) -> Dict:
        """Evaluate a batch of images."""
        clean_images = []
        watermarked_images = []
        exclusion_masks = []
        
        # Generate watermarked versions
        for i in range(len(images)):
            img = images[i:i+1]
            label = labels[i:i+1]
            
            clean_img, wm_img, excl_mask = self.generate_watermarked_image(img, label)
            
            clean_images.append(clean_img)
            watermarked_images.append(wm_img)
            exclusion_masks.append(excl_mask)
        
        clean_images = torch.cat(clean_images, dim=0)
        watermarked_images = torch.cat(watermarked_images, dim=0)
        exclusion_masks = torch.cat(exclusion_masks, dim=0)
        
        # Normalize for classifiers
        norm_transform = transforms.Normalize(mean=[0.5], std=[0.5])
        clean_norm = norm_transform(clean_images)
        wm_norm = norm_transform(watermarked_images)
        
        # Evaluate performance
        metrics = evaluate_adversarial_performance(
            self.c1, self.c2, clean_norm, wm_norm, labels, DEVICE
        )
        
        return {
            'metrics': metrics,
            'clean_images': clean_images,
            'watermarked_images': watermarked_images,
            'exclusion_masks': exclusion_masks
        }
    
    def create_visualization_collage(
        self,
        clean_img: torch.Tensor,
        wm_img: torch.Tensor,
        exclusion_mask: torch.Tensor,
        class_name: str,
        save_path: str = None
    ) -> np.ndarray:
        """Create a visualization collage showing the watermarking process."""
        # Convert tensors to numpy
        clean_np = clean_img.squeeze().cpu().numpy()
        wm_np = wm_img.squeeze().cpu().numpy()
        mask_np = exclusion_mask.squeeze().cpu().numpy()
        
        # Calculate difference map
        diff_np = np.abs(wm_np - clean_np)
        diff_np = diff_np / diff_np.max() if diff_np.max() > 0 else diff_np
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Adversarial Watermarking Results - {class_name}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(clean_np, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Watermarked image
        axes[0, 1].imshow(wm_np, cmap='gray')
        axes[0, 1].set_title('Watermarked Image')
        axes[0, 1].axis('off')
        
        # Difference map
        axes[0, 2].imshow(diff_np, cmap='hot')
        axes[0, 2].set_title('Difference Map')
        axes[0, 2].axis('off')
        
        # Exclusion mask
        axes[1, 0].imshow(mask_np, cmap='viridis')
        axes[1, 0].set_title('Exclusion Mask\n(Yellow = Can Watermark)')
        axes[1, 0].axis('off')
        
        # Watermarked region overlay
        overlay = np.zeros((*clean_np.shape, 3))
        overlay[:, :, 0] = clean_np  # Red channel = original
        overlay[:, :, 1] = clean_np * (1 - mask_np) + wm_np * mask_np  # Green = watermarked where allowed
        overlay[:, :, 2] = clean_np  # Blue channel = original
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Watermarked Overlay\n(Green = Watermarked Regions)')
        axes[1, 1].axis('off')
        
        # Statistics
        stats_text = f"""
        SSIM: {calculate_ssim(clean_img.unsqueeze(0), wm_img.unsqueeze(0)):.4f}
        PSNR: {calculate_psnr(clean_img.unsqueeze(0), wm_img.unsqueeze(0)):.2f} dB
        Max Diff: {diff_np.max():.4f}
        Mean Diff: {diff_np.mean():.4f}
        Watermarked Area: {mask_np.mean():.1%}
        """
        
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        axes[1, 2].set_title('Quality Metrics')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        # Convert to numpy array for return
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return img_array
    
    def comprehensive_evaluation(self, data_root: str, num_samples: int = 50, save_dir: str = "./evaluation_results"):
        """Run comprehensive evaluation on a dataset."""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Running comprehensive evaluation on {num_samples} samples...")
        
        # Load test data
        dataset = MRIDataset(data_root, CLASSES)
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        all_metrics = []
        example_count = 0
        
        for i, idx in enumerate(indices):
            img, label, filename, class_name = dataset[idx]
            img = img.unsqueeze(0).to(DEVICE)
            label = torch.tensor([label]).to(DEVICE)
            
            # Evaluate this sample
            results = self.evaluate_batch(img, label)
            metrics = results['metrics']
            
            all_metrics.append(metrics)
            
            # Save examples for first few samples
            if example_count < 5:
                clean_img = results['clean_images'][0]
                wm_img = results['watermarked_images'][0]
                excl_mask = results['exclusion_masks'][0]
                
                collage_path = os.path.join(save_dir, f"example_{example_count}_{class_name}.png")
                self.create_visualization_collage(clean_img, wm_img, excl_mask, class_name, collage_path)
                
                example_count += 1
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(indices)} samples...")
        
        # Aggregate metrics
        print("\n" + "="*60)
        print("ADVERSARIAL WATERMARKING EVALUATION RESULTS")
        print("="*60)
        
        # Calculate average metrics
        avg_metrics = {}
        for key in ['c1_clean', 'c1_watermarked', 'c2_mode_clean', 'c2_mode_watermarked', 
                   'c2_class_clean', 'c2_class_watermarked']:
            avg_metrics[key] = np.mean([m[key]['accuracy'] for m in all_metrics])
        
        avg_metrics['ssim'] = np.mean([m['image_quality']['ssim'] for m in all_metrics])
        avg_metrics['psnr'] = np.mean([m['image_quality']['psnr'] for m in all_metrics])
        avg_metrics['c2_distinction'] = np.mean([m['c2_class_distinction'] for m in all_metrics])
        
        success_rate = np.mean([m['adversarial_success'] for m in all_metrics])
        
        # Print results
        print(f"\nPerformance Summary (averaged over {len(indices)} samples):")
        print(f"  C1 Clean Accuracy:        {avg_metrics['c1_clean']:.3f}")
        print(f"  C1 Watermarked Accuracy:  {avg_metrics['c1_watermarked']:.3f}")
        print(f"  C2 Mode Detection (Clean): {avg_metrics['c2_mode_clean']:.3f}")
        print(f"  C2 Mode Detection (WM):    {avg_metrics['c2_mode_watermarked']:.3f}")
        print(f"  C2 Class Clean Accuracy:   {avg_metrics['c2_class_clean']:.3f}")
        print(f"  C2 Class WM Accuracy:      {avg_metrics['c2_class_watermarked']:.3f}")
        print(f"  C2 Class Distinction:      {avg_metrics['c2_distinction']:.3f}")
        print(f"  Image Quality (SSIM):      {avg_metrics['ssim']:.4f}")
        print(f"  Image Quality (PSNR):      {avg_metrics['psnr']:.2f} dB")
        print(f"  Adversarial Success Rate:  {success_rate:.1%}")
        
        # Objective check
        print(f"\nObjective Achievement:")
        objectives = {
            "C1 Performance (≥85%)": avg_metrics['c1_clean'] >= 0.85 and avg_metrics['c1_watermarked'] >= 0.85,
            "C2 Mode Detection (≥90%)": avg_metrics['c2_mode_clean'] >= 0.90 and avg_metrics['c2_mode_watermarked'] >= 0.90,
            "C2 Watermark Detection (≥80%)": avg_metrics['c2_class_watermarked'] >= 0.80,
            "C2 Clean Confusion (≤50%)": avg_metrics['c2_class_clean'] <= 0.50,
            "Image Quality (SSIM ≥0.98)": avg_metrics['ssim'] >= 0.98,
            "High Distinction (≥0.3)": avg_metrics['c2_distinction'] >= 0.30
        }
        
        all_achieved = True
        for objective, achieved in objectives.items():
            status = "✅ ACHIEVED" if achieved else "❌ NOT ACHIEVED"
            print(f"  {objective}: {status}")
            if not achieved:
                all_achieved = False
        
        if all_achieved:
            print(f"\n🎯 ALL ADVERSARIAL OBJECTIVES ACHIEVED!")
            print(f"   The watermarking system successfully prevents adversarial drift.")
        else:
            print(f"\n⚠️  Some objectives not met. Consider:")
            print(f"   - Adjusting watermark intensity")
            print(f"   - Extending training epochs")
            print(f"   - Tuning loss weights")
        
        # Save detailed results
        results_file = os.path.join(save_dir, "evaluation_summary.txt")
        with open(results_file, 'w') as f:
            f.write("Adversarial Watermarking Evaluation Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Dataset: {data_root}\n")
            f.write(f"Samples evaluated: {len(indices)}\n")
            f.write(f"Watermark intensity: {self.watermark_intensity:.3f}\n\n")
            
            f.write("Performance Metrics:\n")
            for key, value in avg_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
            
            f.write(f"\nAdversarial Success Rate: {success_rate:.1%}\n")
            f.write(f"All Objectives Achieved: {'Yes' if all_achieved else 'No'}\n")
        
        print(f"\nDetailed results saved to {results_file}")
        print(f"Example visualizations saved in {save_dir}")
        
        return avg_metrics, success_rate


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate adversarial watermarking system")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--pretrained_dir", required=True, help="Directory containing pretrained models")
    parser.add_argument("--data_root", required=True, help="Root directory of test dataset")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--save_dir", default="./evaluation_results", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = AdversarialWatermarkEvaluator(args.checkpoint, args.pretrained_dir)
    
    # Run evaluation
    avg_metrics, success_rate = evaluator.comprehensive_evaluation(
        args.data_root, args.num_samples, args.save_dir
    )
    
    return avg_metrics, success_rate


if __name__ == "__main__":
    # For direct execution, use default paths
    checkpoint_path = "/teamspace/studios/this_studio/unetMRI/output/adversarial_training_robust/checkpoint_epoch_10.pth"
    pretrained_dir = "/teamspace/studios/this_studio/unetMRI/pt models"
    data_root = "/teamspace/studios/this_studio/unetMRI/dataset/brain-tumor-mri-dataset/Training"
    
    if os.path.exists(checkpoint_path):
        evaluator = AdversarialWatermarkEvaluator(checkpoint_path, pretrained_dir)
        avg_metrics, success_rate = evaluator.comprehensive_evaluation(data_root, num_samples=20)
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using adversarial_watermark_trainer.py")

"""
Metrics utilities for adversarial watermarking evaluation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1, img2: Input images of shape (B, C, H, W)
        data_range: Maximum possible pixel value
    
    Returns:
        Average SSIM value across the batch
    """
    # Convert to grayscale if needed
    if img1.dim() == 4 and img1.size(1) == 3:
        img1 = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
        img1 = img1.unsqueeze(1)
    if img2.dim() == 4 and img2.size(1) == 3:
        img2 = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
        img2 = img2.unsqueeze(1)
    
    # Constants for SSIM calculation
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    
    # Compute local means
    mu1 = F.avg_pool2d(img1, 3, 1, padding=1)
    mu2 = F.avg_pool2d(img2, 3, 1, padding=1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = F.avg_pool2d(img1.pow(2), 3, 1, padding=1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2.pow(2), 3, 1, padding=1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, padding=1) - mu1_mu2
    
    # Compute SSIM
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    
    ssim_map = numerator / denominator
    return ssim_map.mean().item()


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1, img2: Input images of shape (B, C, H, W)
        data_range: Maximum possible pixel value
    
    Returns:
        Average PSNR value across the batch
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr.item()


def calculate_lpips(img1: torch.Tensor, img2: torch.Tensor, lpips_model=None) -> float:
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS).
    
    Args:
        img1, img2: Input images of shape (B, C, H, W)
        lpips_model: Pre-initialized LPIPS model
    
    Returns:
        Average LPIPS value across the batch
    """
    if lpips_model is None:
        # Return MSE as fallback if LPIPS not available
        return F.mse_loss(img1, img2).item()
    
    try:
        # Ensure images are in correct format for LPIPS
        if img1.size(1) == 1:  # Grayscale to RGB
            img1 = img1.repeat(1, 3, 1, 1)
        if img2.size(1) == 1:  # Grayscale to RGB
            img2 = img2.repeat(1, 3, 1, 1)
        
        # Normalize to [-1, 1] if needed
        if img1.max() <= 1.0:
            img1 = img1 * 2.0 - 1.0
        if img2.max() <= 1.0:
            img2 = img2 * 2.0 - 1.0
        
        lpips_val = lpips_model(img1, img2)
        return lpips_val.mean().item()
    except Exception:
        # Fallback to MSE if LPIPS fails
        return F.mse_loss(img1, img2).item()


def calculate_entropy(logits: torch.Tensor) -> float:
    """
    Calculate entropy of logits (useful for measuring confusion).
    
    Args:
        logits: Model output logits of shape (B, num_classes)
    
    Returns:
        Average entropy across the batch
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy.mean().item()


def calculate_classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Calculate classification metrics including accuracy, precision, recall, F1.
    
    Args:
        logits: Model output logits of shape (B, num_classes)
        targets: Ground truth labels of shape (B,)
    
    Returns:
        Dictionary containing various metrics
    """
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).float()
    accuracy = correct.mean().item()
    
    # Per-class metrics
    num_classes = logits.size(1)
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for cls in range(num_classes):
        # True positives, false positives, false negatives
        tp = ((predictions == cls) & (targets == cls)).float().sum().item()
        fp = ((predictions == cls) & (targets != cls)).float().sum().item()
        fn = ((predictions != cls) & (targets == cls)).float().sum().item()
        
        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
    
    return {
        'accuracy': accuracy,
        'precision_macro': np.mean(precision_per_class),
        'recall_macro': np.mean(recall_per_class),
        'f1_macro': np.mean(f1_per_class),
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }


def calculate_watermark_strength(clean_img: torch.Tensor, watermarked_img: torch.Tensor) -> dict:
    """
    Calculate various measures of watermark strength and visibility.
    
    Args:
        clean_img: Original clean image
        watermarked_img: Watermarked image
    
    Returns:
        Dictionary containing watermark strength metrics
    """
    # Absolute difference
    diff = torch.abs(watermarked_img - clean_img)
    l1_distance = diff.mean().item()
    l2_distance = torch.sqrt((diff ** 2).mean()).item()
    linf_distance = diff.max().item()
    
    # Signal-to-noise ratio
    signal_power = (clean_img ** 2).mean().item()
    noise_power = (diff ** 2).mean().item()
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    # Normalized cross-correlation
    clean_flat = clean_img.view(-1)
    wm_flat = watermarked_img.view(-1)
    correlation = F.cosine_similarity(clean_flat.unsqueeze(0), wm_flat.unsqueeze(0)).item()
    
    return {
        'l1_distance': l1_distance,
        'l2_distance': l2_distance,
        'linf_distance': linf_distance,
        'snr_db': snr,
        'correlation': correlation
    }


def evaluate_adversarial_performance(
    c1_model,
    c2_model,
    clean_images: torch.Tensor,
    watermarked_images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device
) -> dict:
    """
    Comprehensive evaluation of adversarial watermarking performance.
    
    Args:
        c1_model: Frozen C1 classifier
        c2_model: Dual-head C2 classifier
        clean_images: Clean images tensor
        watermarked_images: Watermarked images tensor
        labels: Ground truth labels
        device: Computation device
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    c1_model.eval()
    c2_model.eval()
    
    with torch.no_grad():
        # C1 performance
        c1_clean_logits = c1_model(clean_images)
        c1_wm_logits = c1_model(watermarked_images)
        
        c1_clean_metrics = calculate_classification_metrics(c1_clean_logits, labels)
        c1_wm_metrics = calculate_classification_metrics(c1_wm_logits, labels)
        
        # C2 performance
        c2_clean_mode, c2_clean_class = c2_model(clean_images)
        c2_wm_mode, c2_wm_class = c2_model(watermarked_images)
        
        # Mode detection targets
        mode_targets_clean = torch.zeros(len(clean_images), dtype=torch.long, device=device)
        mode_targets_wm = torch.ones(len(watermarked_images), dtype=torch.long, device=device)
        
        c2_mode_clean_metrics = calculate_classification_metrics(c2_clean_mode, mode_targets_clean)
        c2_mode_wm_metrics = calculate_classification_metrics(c2_wm_mode, mode_targets_wm)
        
        c2_class_clean_metrics = calculate_classification_metrics(c2_clean_class, labels)
        c2_class_wm_metrics = calculate_classification_metrics(c2_wm_class, labels)
        
        # Image quality metrics
        ssim_val = calculate_ssim(clean_images, watermarked_images)
        psnr_val = calculate_psnr(clean_images, watermarked_images)
        
        # Watermark strength
        watermark_strength = calculate_watermark_strength(clean_images, watermarked_images)
        
        # Adversarial objectives check
        c2_class_distinction = c2_class_wm_metrics['accuracy'] - c2_class_clean_metrics['accuracy']
        
        adversarial_success = (
            c1_clean_metrics['accuracy'] >= 0.85 and
            c1_wm_metrics['accuracy'] >= 0.85 and
            c2_mode_clean_metrics['accuracy'] >= 0.90 and
            c2_mode_wm_metrics['accuracy'] >= 0.90 and
            c2_class_wm_metrics['accuracy'] >= 0.80 and
            c2_class_clean_metrics['accuracy'] <= 0.50 and
            ssim_val >= 0.98
        )
    
    return {
        'c1_clean': c1_clean_metrics,
        'c1_watermarked': c1_wm_metrics,
        'c2_mode_clean': c2_mode_clean_metrics,
        'c2_mode_watermarked': c2_mode_wm_metrics,
        'c2_class_clean': c2_class_clean_metrics,
        'c2_class_watermarked': c2_class_wm_metrics,
        'c2_class_distinction': c2_class_distinction,
        'image_quality': {
            'ssim': ssim_val,
            'psnr': psnr_val
        },
        'watermark_strength': watermark_strength,
        'adversarial_success': adversarial_success
    }

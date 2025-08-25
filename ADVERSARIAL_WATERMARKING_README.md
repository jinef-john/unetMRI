# Fixed Adversarial Watermarking System

## 🚨 Problem Analysis & Solution

### Original Issues Identified:

1. **Adversarial Drift**: C2 was learning to classify both clean and watermarked images correctly, defeating the purpose
2. **Conflicting Loss Functions**: Generator was trying to fool C2 while C2 was learning from the same examples
3. **Improper Training Stages**: No clear separation between warmup and adversarial phases
4. **Loss Balance**: Incorrect weighting of entropy vs classification losses

### ✅ Fixed Implementation

The new `MRI_Watermark_Embedding_CYCLES_FIXED_ADVERSARIAL.py` provides a proper adversarial training framework:

## 🎯 Key Features

### 1. **Staged Training Pipeline**
```
Phase 1: Warmup (5 epochs)     - Train C2 normally on clean images
Phase 2: Adversarial (20 epochs) - C2 vs Generator competition  
Phase 3: Fine-tuning (25 epochs) - Final optimization
```

### 2. **Proper Adversarial Loss Design**
```python
# C2 Training:
entropy_clean = entropy_from_logits(logits_c2_clean)
ce_wm = F.cross_entropy(logits_c2_wm, labels)
loss_c2 = -ALPHA_CLEAN_ENTROPY * entropy_clean + BETA_WM_CE * ce_wm

# Generator Training:
entropy_wm_gen = entropy_from_logits(logits_c2_wm_gen)
fool_c2_loss = -entropy_wm_gen  # Maximize entropy to fool C2
```

### 3. **Brain Exclusion with U2Net**
- Automatic brain segmentation using U2Net
- Frequency domain watermarking outside critical regions
- Adaptive maskability based on edge energy

### 4. **Performance Targets**
- **C1**: ≥85% accuracy on both clean and watermarked images (quality ceiling)
- **C2**: ≤50% accuracy on clean images (failure), ≥90% on watermarked (success)
- **Quality**: SSIM >0.99, minimal visual artifacts

## 🛠️ Usage

### Quick Start
```bash
cd /teamspace/studios/this_studio/unetMRI
python src/watermarking/MRI_Watermark_Embedding_CYCLES_FIXED_ADVERSARIAL.py
```

### Generate U2Net Masks First
```bash
python src/utils/u2net_segmentation.py
```

### Monitor Training
Check `output/csv_logs_FIXED/` for detailed metrics per epoch.

## 📊 Expected Results

### Training Progress:
```
Epoch 1-5 (Warmup):
  C2 accuracy on clean: 0.25 → 0.90 (learning normally)

Epoch 6-25 (Adversarial):
  C2 accuracy on clean: 0.90 → 0.30 (forgetting clean)
  C2 accuracy on watermarked: 0.25 → 0.95 (learning watermarked)
  C1 accuracy: maintained >0.85 (quality preserved)

Epoch 26-50 (Fine-tuning):
  Stable adversarial equilibrium achieved
```

### Success Criteria:
✅ **C1 ≥85%** on clean & watermarked (quality preserved)  
✅ **C2 ≤50%** on clean (properly confused)  
✅ **C2 ≥90%** on watermarked (detects watermark)  
✅ **Visual quality**: Watermarks invisible in brain regions  

## 🔧 Technical Details

### Adversarial Training Strategy

**The key insight**: C2 must learn to **distinguish** watermarked from clean, but **fail** on clean images when asked to classify the medical condition.

1. **C2 Update**: 
   - Maximize entropy on clean images (random guessing)
   - Minimize cross-entropy on watermarked images (correct classification)

2. **Generator Update**:
   - Make watermarked images look "clean" to C2 (high entropy)
   - Preserve image quality (L1, SSIM losses)
   - Avoid critical brain regions (exclusion penalty)

### Frequency Domain Embedding
```python
# Embed in latent space (32x32x1024) via FFT
latent_freq = fft2(latents)
wm_latent_masked = wm_latent * exclusion_mask
latent_freq_wm = latent_freq + wm_latent_masked  
latents_wm = ifft2(latent_freq_wm).real
```

### U2Net Integration
- Pre-trained U2Net segments brain regions
- Exclusion masks prevent watermarking in critical areas
- Works for both MRI (brains) and AFHQ (animal faces)

## 📁 File Structure

```
src/watermarking/
├── MRI_Watermark_Embedding_CYCLES_FIXED_ADVERSARIAL.py  # Main training script
└── (legacy files for reference)

src/utils/
├── u2net_segmentation.py                               # U2Net brain segmentation
└── (other utilities)

output/
├── csv_logs_FIXED/                                     # Training metrics
├── watermarked_cycles_FIXED/                           # Output images  
├── C2-FIXED/                                           # Model checkpoints
└── u2net_masks_png/                                    # Brain masks
```

## 🔍 Debugging Guide

### If C2 accuracy on clean images is too high (>60%):
- Increase `ALPHA_CLEAN_ENTROPY` (try 3.0-5.0)
- Reduce learning rate for C2 in adversarial phase
- Increase watermark intensity

### If C2 accuracy on watermarked images is too low (<80%):
- Increase `BETA_WM_CE` weight
- Reduce watermark intensity
- Ensure sufficient training data

### If C1 performance drops below 85%:
- Reduce watermark intensity
- Increase `DELTA_EXCLUSION` penalty
- Check U2Net mask quality

### If watermarks are too visible:
- Reduce intensity parameter
- Increase mask fraction (embed in more areas with lower intensity)
- Tune frequency domain embedding parameters

## 🎨 Extending to AFHQ Dataset

The framework is designed to work with AFHQ animal faces:

1. **Change dataset path** in config
2. **Update class names** to AFHQ categories  
3. **U2Net masks** will segment animal faces instead of brains
4. **Same adversarial training** principles apply

## 📝 Citation & References

This implementation addresses the adversarial watermarking challenges described in:
- Papers on adversarial training for watermarking
- U2Net segmentation methodology
- Frequency domain embedding techniques

## 🤝 Contributing

Found issues or improvements? The modular design allows easy experimentation with:
- Different watermark generators
- Alternative adversarial loss functions  
- Various embedding domains (spatial, frequency, hybrid)
- Custom exclusion strategies

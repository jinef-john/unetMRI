# Path Update Summary

This document summarizes all the path updates made to convert hardcoded Windows paths to relative project paths.

## Changes Made

### 1. Model Paths (`pt models/`)
Updated references to `.pth` model files to use the `pt models/` directory:

- **Old**: `E:\MRI_LOWMEM\C1-B3-CBAM\MRI-C1EfficientNet_B3_CBAM.pth`
- **New**: `os.path.join(PROJECT_ROOT, "pt models", "MRI-C1EfficientNet_B3_CBAM.pth")`

- **Old**: `E:\MRI_LOWMEM\Encoder_latent_64_128\autoencoder_epoch9.pth`
- **New**: `os.path.join(PROJECT_ROOT, "pt models", "autoencoder_epoch7.pth")`

### 2. Dataset Paths (`dataset/`)
Updated references to training/testing data:

- **Old**: `E:\MRI_LOWMEM\Training` / `E:\MRI_LOWMEM\Testing`
- **New**: `os.path.join(PROJECT_ROOT, "dataset", "brain-tumor-mri-dataset", "Training")`
- **New**: `os.path.join(PROJECT_ROOT, "dataset", "brain-tumor-mri-dataset", "Testing")`

### 3. Output Paths (`output/`)
All output directories now use the centralized `output/` directory:

- Saliency maps: `output/saliency_cycles_latent_*`
- Watermarked images: `output/watermarked_cycles_latent_*`
- Metrics logs: `output/metrics_logs_latent_*` and `output/csv_logs`
- NPZ files: `output/MRI-NPZ_latent_*`
- U2Net masks: `output/u2net_masks_png`
- Model checkpoints: `output/C2-B3`
- Various other outputs

### 4. PROJECT_ROOT Definition
Added consistent project root calculation in all updated files:

```python
# For files in src/ subdirectories (3 levels up)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# For files in scripts/ or experiments/ (2 levels up)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

## Files Updated

### Core Scripts
- `scripts/PTH Encoder checker.py`
- `scripts/niifti launcher.py`
- `scripts/niiftii file check.py`

### Watermarking Scripts
- `src/watermarking/MRI_Watermark_Embedding_CYCLES_latent_64.py`
- `src/watermarking/MRI_Watermark_Embedding_CYCLES_latent_64_FULL REWRITE.py`
- `src/watermarking/MRI_Watermark_Embedding_CYCLES_latent_64_FULL REWRITE_Try1.py`
- `src/watermarking/MRI_Watermark_Embedding_CYCLES_latent_64_FULL REWRITE_Try2_skip128.py`
- `src/watermarking/MRI_Watermark_Embedding_CYCLES_latent_64_skip128_NOCLAMP_NOLPIPS.py`
- `src/watermarking/MRI_Watermark_Embedding_CYCLES_latent_64_U2Net_AUTOTUNE_FREQUENCYDOMAIN.py`
- `src/watermarking/MRI_Watermark_Embedding_CYCLES_latent_64_U2Net_INVERT.py`
- `src/watermarking/ASCC_Watermark_Embedding_CYCLES_latent_64_FULL REWRITE_Try2_skip128.py`

### Utility Scripts
- `src/utils/HD-BET.py`
- `src/utils/ASCC-Collage Builder.py`
- `src/utils/MRI_Dataset and NPZ Check.py`
- `src/utils/MRI_NFTI_MASKS_NNUNET.py`
- `src/utils/MRI_NFTI_MASKS_U2NET.py`

### Experiment Scripts (MRI only)
- `experiments/MRI/MRI_C1_B3_Train.py`
- `experiments/MRI/MRI_Encoder_Latent_64_Train.py`
- `experiments/MRI/MRI_Encoder_Latent_64_128_Train.py`
- `experiments/MRI/MRI_Tiny_Train.py`

## Notes

1. **AFHQ and ASCC experiments**: These still reference their original dataset paths and may need manual updates based on your specific dataset locations.

2. **Model files**: Make sure the actual model files exist in the `pt models/` directory with the correct filenames.

3. **Directory creation**: Most scripts now include automatic directory creation with `os.makedirs(dir, exist_ok=True)`.

4. **Cross-platform compatibility**: All paths now use `os.path.join()` for cross-platform compatibility.

## File Structure After Updates

```
unetMRI/
├── dataset/
│   └── brain-tumor-mri-dataset/
│       ├── Training/
│       │   ├── glioma/
│       │   ├── meningioma/
│       │   ├── notumor/
│       │   └── pituitary/
│       └── Testing/
│           ├── glioma/
│           ├── meningioma/
│           ├── notumor/
│           └── pituitary/
├── pt models/
│   ├── MRI-C1EfficientNet_B3.pth
│   ├── MRI-C1EfficientNet_B3_CBAM.pth
│   └── autoencoder_epoch7.pth
├── output/
│   ├── (automatically created subdirectories)
│   └── README.md
├── src/
├── scripts/
├── experiments/
└── (other project files)
```

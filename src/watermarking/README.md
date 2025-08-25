# Watermarking Module

This directory contains implementations of watermarking techniques in neural networks, particularly focusing on cycle-consistent watermarking in latent space.

## Key Concepts

- **Latent Space Watermarking**: Embedding watermarks in the latent representations rather than directly in images
- **Cycle Consistency**: Ensuring that watermarked images can be properly classified while maintaining image quality
- **Multi-scale Approach**: Working with different latent dimensions (64x64, 128x128) for various trade-offs

## Files

Files follow naming convention: `{DATASET}_Watermark_Embedding_CYCLES_latent_{SIZE}_{VARIANT}.py`

### MRI Dataset Watermarking

- Multiple variants exploring different approaches
- Integration with U2-Net for saliency-guided watermarking
- Frequency domain techniques
- Dynamic saliency approaches

### ASCC Dataset Watermarking

- Adapted techniques for manufacturing surface images
- Similar architecture but dataset-specific optimizations

## Techniques Used

1. **Curriculum Learning**: Progressive intensity watermarking across cycles
2. **Saliency-Guided Embedding**: Using attention maps to guide watermark placement
3. **Perceptual Loss Integration**: Maintaining visual quality while embedding watermarks
4. **Early Stopping**: Preventing overfitting during watermark training

## Usage

Each watermarking script is typically self-contained and includes:

- Dataset loading
- Model initialization (classifier + encoder/decoder)
- Training loop with multiple cycles
- Evaluation and logging
- Output generation (watermarked images, metrics)

## Requirements

- Pre-trained classifier models
- Pre-trained autoencoder models
- Appropriate dataset organization
- Sufficient GPU memory for batch processing

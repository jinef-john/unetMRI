# Neural Network Models

This directory contains the core neural network architectures used in the project.

## Files

- **efficientnet_cbam.py**: EfficientNet-B3 with Convolutional Block Attention Module (CBAM)

  - Used for classification tasks
  - Includes channel and spatial attention mechanisms
  - Supports both 1-channel (grayscale) and 3-channel (RGB) inputs

- **autoencoder.py**: Autoencoder implementation with skip connections

  - Used for latent space representation learning
  - Includes perceptual loss using VGG features
  - CBAM attention integrated in encoder/decoder
  - Multiple latent dimensions supported (64x64, 128x128)

- **u2net.py**: U2-Net model for saliency detection
  - Used for generating attention masks
  - Helps in watermarking applications
  - Provides fine-grained spatial attention

## Usage

These models are imported and used by the experiment scripts in the `experiments/` directory. Each model includes both training and inference capabilities.

## Dependencies

- PyTorch
- torchvision
- pytorch-msssim (for SSIM loss)
- timm (for some utilities)

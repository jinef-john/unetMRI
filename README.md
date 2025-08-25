# Medical Image Analysis and Watermarking Project

This project focuses on medical image analysis, particularly MRI data, with advanced neural network architectures and watermarking techniques.

## Project Structure

```
uNETMRI/
├── src/                    # Source code
│   ├── models/            # Neural network models
│   │   ├── efficientnet_cbam.py    # EfficientNet with CBAM attention
│   │   ├── autoencoder.py          # Autoencoder for latent space work
│   │   └── u2net.py                # U2-Net model
│   ├── training/          # Training utilities and base classes
│   ├── data/              # Data processing and utilities
│   ├── watermarking/      # Watermarking implementations
│   └── utils/             # General utilities
├── experiments/           # Experiment-specific code
│   ├── AFHQ/             # Animal Faces-HQ dataset experiments
│   ├── MRI/              # MRI dataset experiments
│   └── ASCC/             # ASCC dataset experiments
├── scripts/              # Standalone scripts for testing and utilities
└── Py Code Files/        # Original unorganized files (deprecated)
```

## Datasets

- **MRI**: Medical brain imaging data for tumor classification
- **AFHQ**: Animal Faces-HQ dataset for general computer vision tasks
- **ASCC**: Additive Manufacturing Surface Condition Classification

## Key Components

### Models

- **EfficientNet-B3 with CBAM**: Convolutional Block Attention Module integration
- **Autoencoder**: For latent space representation and reconstruction
- **U2-Net**: For saliency detection and segmentation

### Techniques

- **Watermarking**: Cycle-consistent watermarking in latent space
- **Attention Mechanisms**: CBAM for improved feature learning
- **Multi-scale Processing**: Various latent dimensions (64, 128, etc.)

## Getting Started

1. Install dependencies (see requirements.txt when available)
2. Choose your experiment from the `experiments/` directory
3. Configure paths in the experiment scripts
4. Run training scripts

## Note

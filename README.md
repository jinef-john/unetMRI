
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
└── Py Code Files/        # Original 
```

# Experiments Directory

This directory contains dataset-specific experiment implementations. Each subdirectory focuses on a particular dataset with its own training scripts and configurations.

## Structure

### AFHQ (Animal Faces-HQ)

- **Dataset**: High-quality animal face images
- **Classes**: Typically 3 classes (cat, dog, wild)
- **Focus**: General computer vision and classification
- **Scripts**: Training scripts for EfficientNet-B3 with and without CBAM

### MRI (Medical Imaging)

- **Dataset**: Brain MRI scans for tumor classification
- **Classes**: 4 classes (glioma, meningioma, notumor, pituitary)
- **Focus**: Medical image analysis and classification
- **Special considerations**: Grayscale images, medical domain expertise required
- **Scripts**: Various training approaches including encoder-decoder architectures

### ASCC (Additive Manufacturing Surface Condition Classification)

- **Dataset**: Manufacturing surface condition images
- **Classes**: 7 classes (various surface conditions and anomalies)
- **Focus**: Industrial quality control and defect detection
- **Image size**: Typically smaller than other datasets (120x160)
- **Scripts**: Adapted training for manufacturing domain

## Common Patterns

All experiment directories typically contain:

- **Training scripts**: `*_Train.py` files for different model configurations
- **Encoder training**: `*_Encoder_*_Train.py` for autoencoder training
- **Model definitions**: Dataset-specific model adaptations when needed

## Usage

1. Choose the appropriate dataset directory
2. Configure data paths in the training scripts
3. Ensure proper data organization (typically ImageFolder structure)
4. Run training scripts with appropriate hyperparameters

## Dependencies

- PyTorch and torchvision
- sklearn for metrics
- pandas for logging
- tqdm for progress bars
- timm for advanced training utilities (Mixup, etc.)

## Note

Each experiment may have different:

- Image sizes and preprocessing
- Number of classes
- Batch sizes and learning rates
- Training epochs and stopping criteria

Adjust these parameters based on your specific dataset and computational resources.

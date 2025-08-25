# Data Processing Module

This directory contains scripts for data preprocessing, dataset creation, and data manipulation tasks.

## Categories

### NPZ Builders

Files like `*_NPZBuilder_*.py` create compressed numpy archives from image datasets:

- Efficient storage and loading during training
- Preprocessed data with consistent formats
- Latent space representations when using encoder models

### Dataset Utilities

- **Train-Val Split**: Scripts for splitting datasets into training and validation sets
- **Tiny Dataset Creation**: Creating smaller subsets for testing and development
- **Format Conversion**: Converting between different image formats (e.g., NIFTI to standard formats)

### Processing Scripts

- **Data sanitization**: Cleaning and validating datasets
- **File organization**: Organizing files into proper directory structures
- **Format standardization**: Ensuring consistent image sizes and formats

## File Naming Convention

- `{DATASET}_NPZBuilder_*.py`: Create NPZ archives for specific datasets
- `{DATASET}_Tiny_*.py`: Create small subsets of datasets
- `*_Split.py`: Training/validation split utilities

## Usage

1. **NPZ Creation**: Run NPZ builders after organizing your raw image data
2. **Dataset Splitting**: Use split scripts to create train/val/test divisions
3. **Format Conversion**: Use conversion scripts for medical imaging formats

## Dependencies

- numpy for array operations
- PIL/Pillow for image processing
- torchvision for transforms
- nibabel for NIFTI medical imaging format (when applicable)

## Important Notes

- Ensure proper data organization before running processing scripts
- NPZ files can significantly speed up training by reducing I/O overhead
- Always backup original data before running processing scripts
- Check image dimensions and formats match your model requirements

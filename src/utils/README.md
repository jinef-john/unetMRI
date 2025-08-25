# Utilities Module

This directory contains various utility scripts and helper functions used throughout the project.

## Categories

### Visualization and Analysis

- **Collage Builders**: Create visual collages of images for analysis and presentation
- **Saliency Map Generators**: Generate and visualize attention/saliency maps
- **Model Checkers**: Scripts to validate model architectures and weights

### Data Processing Utilities

- **File Renamers**: Batch rename files with consistent naming conventions
- **Data Sanitizers**: Clean and validate datasets
- **Format Converters**: Handle different image and data formats

### Medical Imaging Specific

- **HD-BET**: Brain extraction tools for MRI preprocessing
- **NIFTI Mask Generators**: Create masks for medical imaging data
- **U2Net Utilities**: Utilities for working with U2-Net segmentation

### Quality Control

- **Dataset Checkers**: Validate dataset integrity and structure
- **Model Validators**: Check model performance and outputs
- **RGB/Grayscale Converters**: Handle color space conversions

## Usage

These utilities are typically standalone scripts that can be run independently:

```bash
python script_name.py
```

Most utilities require configuration of input/output paths at the top of the script.

## Key Files

- **HD-BET.py**: Brain extraction for MRI preprocessing
- **MRI*SALMAP_Generator*\*.py**: Generate saliency maps for MRI data
- **_Collage_.py**: Create image collages for visualization
- **Grayscale dataset sanitizer.py**: Convert and clean grayscale datasets
- Various checker scripts for data validation

## Dependencies

- OpenCV for image processing
- matplotlib for visualization
- nibabel for medical imaging formats
- PIL/Pillow for general image operations
- torch/torchvision for model-related utilities

## Notes

- Many scripts contain hardcoded paths that need to be updated for your environment
- Some utilities are dataset-specific and may need adaptation
- Check script documentation for specific requirements and usage instructions

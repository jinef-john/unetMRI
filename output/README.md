# Output Directory

This directory contains all the generated outputs from the unetMRI project:

## Generated Directories
- `saliency_cycles_latent_*` - Saliency maps generated during training
- `watermarked_cycles_latent_*` - Watermarked images 
- `metrics_logs_latent_*` - CSV logs with training metrics
- `MRI-NPZ_latent_*` - NPZ files for processed data
- `u2net_masks_png` - U2Net generated mask images
- `resultant_Masks` - Final segmentation masks
- `csv_logs` - Various CSV logging outputs
- `C2-B3` - Model checkpoints for C2 EfficientNet-B3
- `nnunet_*` - nnU-Net specific directories

## Structure
The output directory is automatically created when scripts are run and will contain subdirectories based on the specific experiment configurations.

Note: All hardcoded Windows paths (E:\...) have been updated to use this centralized output directory.

# Scripts Directory

This directory contains standalone scripts for testing, validation, and quick utilities that don't fit into the main source code structure.

## Types of Scripts

### Testing Scripts

- **lpips test.py**: Test LPIPS (Learned Perceptual Image Patch Similarity) functionality
- **pytorch gen chk.py**: Check PyTorch generation and GPU functionality
- Various model and functionality testers

### Validation Scripts

- **PTH Encoder checker.py**: Validate saved PyTorch model files
- Model weight and architecture validation scripts

### Launcher Scripts

- **niifti launcher.py**: Launch NIFTI-related processing
- Quick execution scripts for common tasks

### File Management

- **niiftii file check.py**: Validate NIFTI file integrity
- File system and data validation utilities

## Usage

These scripts are typically run directly from the command line:

```bash
cd scripts/
python script_name.py
```

## Characteristics

- **Self-contained**: Each script should run independently
- **Quick tasks**: Usually for testing or one-off operations
- **Development aids**: Help during development and debugging
- **Validation tools**: Check system setup and data integrity

## Notes

- Scripts may contain hardcoded paths that need updating
- Some scripts are experimental or for debugging purposes
- Check each script's header comments for specific usage instructions
- These scripts are generally not part of the main training pipeline

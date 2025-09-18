#!/bin/bash

# Full 50-epoch adversarial watermarking training script
# This script runs the training with output logging to files

# Set up directories
OUTPUT_DIR="/teamspace/studios/this_studio/unetMRI/output/adversarial_training_full_50epochs"
LOG_DIR="/teamspace/studios/this_studio/unetMRI/logs"

# Create directories if they don't exist
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Set timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_full_50epochs_$TIMESTAMP.log"

echo "Starting 50-epoch adversarial watermarking training..."
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Started at: $(date)"

# Change to project directory
cd /teamspace/studios/this_studio/unetMRI

# Run training with output piped to both file and console
python src/training/adversarial_watermark_trainer.py 2>&1 | tee $LOG_FILE

# Check if training completed successfully
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Training completed successfully at $(date)"
    echo "✓ Checkpoints saved in: $OUTPUT_DIR"
    echo "✓ Full log available at: $LOG_FILE"
    
    # Create a summary of the training
    echo "=== Training Summary ===" >> $LOG_FILE
    echo "Completed at: $(date)" >> $LOG_FILE
    echo "Total epochs: 50" >> $LOG_FILE
    echo "Final checkpoint: $OUTPUT_DIR/checkpoint_epoch_50.pth" >> $LOG_FILE
    
    # Show final results
    echo ""
    echo "Training completed! Check the following:"
    echo "1. Checkpoints: $OUTPUT_DIR"
    echo "2. Full log: $LOG_FILE"
    echo "3. Latest checkpoint: $OUTPUT_DIR/checkpoint_epoch_50.pth"
else
    echo "✗ Training failed. Check log file: $LOG_FILE"
    exit 1
fi
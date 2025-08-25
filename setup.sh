#!/bin/bash

# Setup script for uNETMRI project
echo "Setting up uNETMRI project environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories if they don't exist
echo "Creating output directories..."
mkdir -p outputs/models
mkdir -p outputs/logs  
mkdir -p outputs/results
mkdir -p outputs/checkpoints

# Copy config template if config doesn't exist
if [ ! -f "config.ini" ]; then
    echo "Creating config.ini from template..."
    cp config.ini.template config.ini
    echo "Please edit config.ini with your specific paths and settings!"
fi

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit config.ini with your data paths"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Choose an experiment from the experiments/ directory"
echo "4. Run the training script"
echo ""
echo "For more information, see README.md"

"""
Data loader for MRI dataset compatible with the adversarial watermarking system.
"""

import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import List, Tuple


class MRIDataset(Dataset):
    """
    MRI Dataset loader for brain tumor classification.
    
    Expected directory structure:
    data_root/
    ├── glioma/
    │   ├── image1.jpg
    │   └── ...
    ├── meningioma/
    │   ├── image1.jpg
    │   └── ...
    ├── notumor/
    │   ├── image1.jpg
    │   └── ...
    └── pituitary/
        ├── image1.jpg
        └── ...
    """
    
    def __init__(self, data_root: str, classes: List[str], transform=None):
        self.data_root = data_root
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.float())
            ])
        else:
            self.transform = transform
        
        # Collect all image paths
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
        for class_name in classes:
            class_dir = os.path.join(data_root, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} not found")
                continue
            
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(class_dir, filename))
                    self.labels.append(self.class_to_idx[class_name])
                    self.class_names.append(class_name)
        
        print(f"Loaded {len(self.image_paths)} images from {len(classes)} classes")
        for class_name in classes:
            count = self.class_names.count(class_name)
            print(f"  {class_name}: {count} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        try:
            # Try PIL first
            image = Image.open(image_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path} with PIL: {e}")
            # Fallback to cv2/numpy
            import cv2
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                # Create dummy image if loading fails
                image = np.zeros((512, 512), dtype=np.uint8)
            image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Ensure correct shape and type
        if image.dim() == 2:  # Add channel dimension if missing
            image = image.unsqueeze(0)
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        label = self.labels[idx]
        filename = os.path.basename(self.image_paths[idx])
        class_name = self.class_names[idx]
        
        return image, label, filename, class_name


def create_dataloaders(data_root: str, classes: List[str], batch_size: int = 16, 
                      num_workers: int = 4, train_split: float = 0.8):
    """
    Create train and validation dataloaders.
    
    Args:
        data_root: Root directory containing class subdirectories
        classes: List of class names
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        train_split: Fraction of data to use for training
    
    Returns:
        train_loader, val_loader
    """
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float())
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float())
    ])
    
    # Create full dataset
    full_dataset = MRIDataset(data_root, classes, transform=None)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, dataset_size))
    
    # Create subset datasets with different transforms
    train_dataset = torch.utils.data.Subset(
        MRIDataset(data_root, classes, transform=train_transform),
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        MRIDataset(data_root, classes, transform=val_transform),
        val_indices
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def test_dataloader():
    """Test the dataloader with sample data."""
    # Test paths - adjust these to your actual data
    data_root = "/teamspace/studios/this_studio/unetMRI/dataset/brain-tumor-mri-dataset/Training"
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    if not os.path.exists(data_root):
        print(f"Test data not found at {data_root}")
        return
    
    # Create dataset
    dataset = MRIDataset(data_root, classes)
    
    if len(dataset) == 0:
        print("No images found in dataset")
        return
    
    # Test loading a few samples
    print(f"Testing dataset with {len(dataset)} samples...")
    
    for i in range(min(3, len(dataset))):
        image, label, filename, class_name = dataset[i]
        print(f"Sample {i}: {filename}")
        print(f"  Class: {class_name} (label: {label})")
        print(f"  Image shape: {image.shape}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print()
    
    # Test dataloader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch_idx, (images, labels, filenames, class_names) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels: {labels}")
        print(f"  Classes: {class_names}")
        
        if batch_idx >= 1:  # Only test first 2 batches
            break
    
    print("Dataloader test completed successfully!")


if __name__ == "__main__":
    test_dataloader()

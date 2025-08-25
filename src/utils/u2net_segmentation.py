# -*- coding: utf-8 -*-
"""
U2Net Integration for Brain/Face Segmentation
Provides pre-trained U2Net model and training utilities
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class U2NET(nn.Module):
    """U2Net model for segmentation"""
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        
        # Encoder stages
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage6 = RSU4F(512, 256, 512)
        
        # Decoder stages
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        
        # Output layers
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)
    
    def forward(self, x):
        hx = x
        
        # Encoder
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        
        hx6 = self.stage6(hx)
        hx6up = F.interpolate(hx6, size=hx5.shape[2:], mode='bilinear', align_corners=False)
        
        # Decoder
        hx5d = self.stage5d(torch.cat([hx6up, hx5], 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        
        hx4d = self.stage4d(torch.cat([hx5dup, hx4], 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        
        hx3d = self.stage3d(torch.cat([hx4dup, hx3], 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        
        hx2d = self.stage2d(torch.cat([hx3dup, hx2], 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        
        hx1d = self.stage1d(torch.cat([hx2dup, hx1], 1))
        
        # Side outputs
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, size=d1.shape[2:], mode='bilinear', align_corners=False)
        
        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, size=d1.shape[2:], mode='bilinear', align_corners=False)
        
        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, size=d1.shape[2:], mode='bilinear', align_corners=False)
        
        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, size=d1.shape[2:], mode='bilinear', align_corners=False)
        
        d6 = self.side6(hx6)
        d6 = F.interpolate(d6, size=d1.shape[2:], mode='bilinear', align_corners=False)
        
        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], 1))
        
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)

# RSU blocks for U2Net
class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        
        hx6 = self.rebnconv6(hx)
        
        hx7 = self.rebnconv7(hx6)
        
        hx6d = self.rebnconv6d(torch.cat([hx7, hx6], 1))
        hx6dup = F.interpolate(hx6d, size=hx5.shape[2:], mode='bilinear', align_corners=False)
        
        hx5d = self.rebnconv5d(torch.cat([hx6dup, hx5], 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        
        hx4d = self.rebnconv4d(torch.cat([hx5dup, hx4], 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        
        hx3d = self.rebnconv3d(torch.cat([hx4dup, hx3], 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        
        hx2d = self.rebnconv2d(torch.cat([hx3dup, hx2], 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        
        hx1d = self.rebnconv1d(torch.cat([hx2dup, hx1], 1))
        
        return hx1d + hxin

# Similar implementations for RSU6, RSU5, RSU4, RSU4F...
class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        
        hx5 = self.rebnconv5(hx)
        
        hx6 = self.rebnconv6(hx5)
        
        hx5d = self.rebnconv5d(torch.cat([hx6, hx5], 1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=False)
        
        hx4d = self.rebnconv4d(torch.cat([hx5dup, hx4], 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        
        hx3d = self.rebnconv3d(torch.cat([hx4dup, hx3], 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        
        hx2d = self.rebnconv2d(torch.cat([hx3dup, hx2], 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        
        hx1d = self.rebnconv1d(torch.cat([hx2dup, hx1], 1))
        
        return hx1d + hxin

class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        
        hx5 = self.rebnconv5(hx4)
        
        hx4d = self.rebnconv4d(torch.cat([hx5, hx4], 1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=False)
        
        hx3d = self.rebnconv3d(torch.cat([hx4dup, hx3], 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        
        hx2d = self.rebnconv2d(torch.cat([hx3dup, hx2], 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        
        hx1d = self.rebnconv1d(torch.cat([hx2dup, hx1], 1))
        
        return hx1d + hxin

class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        
        hx4 = self.rebnconv4(hx3)
        
        hx3d = self.rebnconv3d(torch.cat([hx4, hx3], 1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=False)
        
        hx2d = self.rebnconv2d(torch.cat([hx3dup, hx2], 1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=False)
        
        hx1d = self.rebnconv1d(torch.cat([hx2dup, hx1], 1))
        
        return hx1d + hxin

class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        
        hx4 = self.rebnconv4(hx3)
        
        hx3d = self.rebnconv3d(torch.cat([hx4, hx3], 1))
        hx2d = self.rebnconv2d(torch.cat([hx3d, hx2], 1))
        hx1d = self.rebnconv1d(torch.cat([hx2d, hx1], 1))
        
        return hx1d + hxin

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        
        return xout

# U2Net utility functions
class U2NetSegmenter:
    def __init__(self, model_path=None, device='cuda'):
        self.device = device
        self.model = U2NET(3, 1).to(device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded U2Net model from {model_path}")
        else:
            print("Warning: No U2Net model path provided. Using random weights.")
        
        self.model.eval()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def segment_batch(self, images, target_size=(512, 512)):
        """
        Segment a batch of images
        Args:
            images: torch.Tensor [B, C, H, W] in range [0, 1]
            target_size: tuple for output mask size
        Returns:
            masks: torch.Tensor [B, 1, H, W] binary masks
        """
        batch_size = images.shape[0]
        masks = torch.zeros(batch_size, 1, *target_size, device=self.device)
        
        with torch.no_grad():
            for i in range(batch_size):
                img = images[i]  # [C, H, W]
                
                # Convert grayscale to RGB if needed
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
                
                # Preprocess
                img_resized = F.interpolate(img.unsqueeze(0), size=(320, 320), mode='bilinear', align_corners=False)
                img_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_resized[0])
                
                # Segment
                d0, d1, d2, d3, d4, d5, d6 = self.model(img_norm.unsqueeze(0))
                pred = d0[0, 0]  # [H, W]
                
                # Resize to target size
                mask = F.interpolate(pred.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
                masks[i, 0] = (mask[0, 0] > 0.5).float()
        
        return masks
    
    def save_masks(self, images, output_dir, filenames, class_names):
        """Save segmentation masks as PNG files"""
        os.makedirs(output_dir, exist_ok=True)
        
        masks = self.segment_batch(images)
        
        for i, (fname, cls_name) in enumerate(zip(filenames, class_names)):
            cls_dir = os.path.join(output_dir, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            
            mask = masks[i, 0].cpu().numpy() * 255
            mask_path = os.path.join(cls_dir, os.path.splitext(fname)[0] + ".png")
            cv2.imwrite(mask_path, mask.astype(np.uint8))

def download_u2net_pretrained():
    """Download pre-trained U2Net model"""
    import urllib.request
    
    model_dir = os.path.join(PROJECT_ROOT, "pt models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "u2net.pth")
    
    if not os.path.exists(model_path):
        print("Downloading pre-trained U2Net model...")
        url = "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ"
        
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"U2Net model downloaded to {model_path}")
        except Exception as e:
            print(f"Failed to download U2Net model: {e}")
            print("Please manually download from: https://github.com/xuebinqin/U-2-Net")
    
    return model_path

def generate_u2net_masks_for_dataset():
    """Generate U2Net masks for the entire MRI dataset"""
    print("Generating U2Net masks for MRI dataset...")
    
    # Download model if needed
    model_path = download_u2net_pretrained()
    
    # Initialize segmenter
    segmenter = U2NetSegmenter(model_path)
    
    # Dataset paths
    train_dir = os.path.join(PROJECT_ROOT, "dataset", "brain-tumor-mri-dataset", "Training")
    output_dir = os.path.join(PROJECT_ROOT, "output", "u2net_masks_png")
    
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        
        print(f"Processing class: {cls}")
        
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Process in batches
        batch_size = 8
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i+batch_size]
            batch_images = []
            
            for fname in batch_files:
                img_path = os.path.join(cls_dir, fname)
                img = Image.open(img_path).convert('L')
                img = transforms.Resize((512, 512))(img)
                img_tensor = transforms.ToTensor()(img)
                batch_images.append(img_tensor)
            
            if batch_images:
                batch_tensor = torch.stack(batch_images)
                batch_classes = [cls] * len(batch_files)
                
                segmenter.save_masks(batch_tensor, output_dir, batch_files, batch_classes)
                
                print(f"  Processed {i+len(batch_files)}/{len(files)} files")
    
    print(f"U2Net masks saved to {output_dir}")

if __name__ == "__main__":
    # Generate masks for the dataset
    generate_u2net_masks_for_dataset()

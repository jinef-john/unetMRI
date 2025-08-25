#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation Script for Fixed Adversarial Watermarking Setup
Tests all components before running full training
"""

import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "models"))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "utils"))

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from efficientnet_cbam import EfficientNetB3_CBAM_Bottleneck
        print("  ✅ EfficientNet with CBAM imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import EfficientNet CBAM: {e}")
        return False
    
    try:
        from autoencoder import Encoder, Decoder
        print("  ✅ Autoencoder imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import Autoencoder: {e}")
        return False
    
    try:
        from u2net_segmentation import U2NetSegmenter, U2NET
        print("  ✅ U2Net imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import U2Net: {e}")
        return False
    
    return True

def test_model_files():
    """Test if required model files exist"""
    print("\\n📁 Testing model files...")
    
    model_dir = os.path.join(PROJECT_ROOT, "pt models")
    required_files = [
        "MRI-C1EfficientNet_B3_CBAM.pth",
        "autoencoder_epoch7.pth",
        "MRI-C1EfficientNet_B3.pth"
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✅ {filename} exists ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {filename} missing")
            all_exist = False
    
    return all_exist

def test_dataset():
    """Test if dataset is properly structured"""
    print("\\n📊 Testing dataset structure...")
    
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset", "brain-tumor-mri-dataset", "Training")
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    if not os.path.exists(dataset_dir):
        print(f"  ❌ Dataset directory not found: {dataset_dir}")
        return False
    
    total_images = 0
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        if os.path.exists(cls_dir):
            images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(images)
            print(f"  ✅ {cls}: {len(images)} images")
        else:
            print(f"  ❌ Class directory missing: {cls}")
            return False
    
    print(f"  📈 Total images: {total_images}")
    return total_images > 0

def test_model_loading():
    """Test if models can be loaded successfully"""
    print("\\n🔧 Testing model loading...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  🖥️  Using device: {device}")
    
    try:
        # Test C1 loading
        from efficientnet_cbam import EfficientNetB3_CBAM_Bottleneck
        c1_path = os.path.join(PROJECT_ROOT, "pt models", "MRI-C1EfficientNet_B3_CBAM.pth")
        
        if os.path.exists(c1_path):
            c1 = EfficientNetB3_CBAM_Bottleneck(num_classes=4)
            c1.load_state_dict(torch.load(c1_path, map_location=device))
            c1.eval()
            print("  ✅ C1 (EfficientNet-B3 CBAM) loaded successfully")
        else:
            print("  ⚠️  C1 model file missing - will affect training")
        
        # Test autoencoder loading
        from autoencoder import Encoder, Decoder
        ae_path = os.path.join(PROJECT_ROOT, "pt models", "autoencoder_epoch7.pth")
        
        if os.path.exists(ae_path):
            state_dict = torch.load(ae_path, map_location=device)
            encoder = Encoder()
            decoder = Decoder()
            
            # Load encoder weights
            encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
            encoder.load_state_dict(encoder_state, strict=False)
            
            # Load decoder weights  
            decoder_state = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}
            decoder.load_state_dict(decoder_state, strict=False)
            
            print("  ✅ Autoencoder loaded successfully")
        else:
            print("  ⚠️  Autoencoder model file missing - will affect training")
    
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False
    
    return True

def test_forward_pass():
    """Test a forward pass through the models"""
    print("\\n⚡ Testing forward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create dummy input
        dummy_img = torch.randn(2, 1, 512, 512).to(device)
        print(f"  📥 Input shape: {dummy_img.shape}")
        
        # Test autoencoder if available
        ae_path = os.path.join(PROJECT_ROOT, "pt models", "autoencoder_epoch7.pth")
        if os.path.exists(ae_path):
            from autoencoder import Encoder, Decoder
            state_dict = torch.load(ae_path, map_location=device)
            
            encoder = Encoder().to(device)
            decoder = Decoder().to(device)
            
            encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
            encoder.load_state_dict(encoder_state, strict=False)
            decoder_state = {k.replace('decoder.', ''): v for k, v in state_dict.items() if k.startswith('decoder.')}
            decoder.load_state_dict(decoder_state, strict=False)
            
            encoder.eval()
            decoder.eval()
            
            with torch.no_grad():
                dummy_norm = (dummy_img - 0.5) / 0.5
                latents, skip64s = encoder(dummy_norm)
                reconstructed = decoder(latents, skip64s)
                
            print(f"  🔄 Latents shape: {latents.shape}")
            print(f"  🔄 Skip64s shape: {skip64s.shape}")
            print(f"  📤 Reconstructed shape: {reconstructed.shape}")
            print("  ✅ Autoencoder forward pass successful")
        
        # Test classifier if available
        c1_path = os.path.join(PROJECT_ROOT, "pt models", "MRI-C1EfficientNet_B3_CBAM.pth")
        if os.path.exists(c1_path):
            from efficientnet_cbam import EfficientNetB3_CBAM_Bottleneck
            c1 = EfficientNetB3_CBAM_Bottleneck(num_classes=4).to(device)
            c1.load_state_dict(torch.load(c1_path, map_location=device))
            c1.eval()
            
            with torch.no_grad():
                # Normalize for classifier
                norm_transform = transforms.Normalize(mean=[0.5], std=[0.5])
                dummy_norm = norm_transform(dummy_img)
                logits = c1(dummy_norm)
                
            print(f"  🎯 Classifier output shape: {logits.shape}")
            print("  ✅ Classifier forward pass successful")
    
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        return False
    
    return True

def test_output_directories():
    """Test if output directories can be created"""
    print("\\n📂 Testing output directories...")
    
    output_dirs = [
        "output",
        "output/csv_logs_FIXED",
        "output/watermarked_cycles_FIXED", 
        "output/C2-FIXED",
        "output/u2net_masks_png"
    ]
    
    for dir_name in output_dirs:
        dir_path = os.path.join(PROJECT_ROOT, dir_name)
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"  ✅ {dir_name} directory ready")
        except Exception as e:
            print(f"  ❌ Failed to create {dir_name}: {e}")
            return False
    
    return True

def generate_sample_u2net_masks():
    """Generate sample U2Net masks for testing"""
    print("\\n🧠 Generating sample U2Net masks...")
    
    try:
        # Create dummy masks for testing
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset", "brain-tumor-mri-dataset", "Training")
        output_dir = os.path.join(PROJECT_ROOT, "output", "u2net_masks_png")
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        total_generated = 0
        for cls in classes:
            cls_input_dir = os.path.join(dataset_dir, cls)
            cls_output_dir = os.path.join(output_dir, cls)
            
            if not os.path.exists(cls_input_dir):
                continue
                
            os.makedirs(cls_output_dir, exist_ok=True)
            
            # Get a few sample images
            images = [f for f in os.listdir(cls_input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
            
            for img_name in images:
                # Create dummy brain mask (circular region in center)
                mask = np.zeros((512, 512), dtype=np.uint8)
                center = (256, 256)
                radius = 180
                y, x = np.ogrid[:512, :512]
                brain_region = (x - center[0])**2 + (y - center[1])**2 <= radius**2
                mask[brain_region] = 255
                
                mask_name = os.path.splitext(img_name)[0] + ".png"
                mask_path = os.path.join(cls_output_dir, mask_name)
                
                import cv2
                cv2.imwrite(mask_path, mask)
                total_generated += 1
        
        print(f"  ✅ Generated {total_generated} sample brain masks")
        print("  ⚠️  Note: These are dummy circular masks. Use actual U2Net for real training.")
        
    except Exception as e:
        print(f"  ❌ Failed to generate sample masks: {e}")
        return False
    
    return True

def main():
    """Run all validation tests"""
    print("🚀 Adversarial Watermarking System Validation")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Model Files", test_model_files),
        ("Dataset Structure", test_dataset), 
        ("Model Loading", test_model_loading),
        ("Forward Pass", test_forward_pass),
        ("Output Directories", test_output_directories),
        ("Sample U2Net Masks", generate_sample_u2net_masks)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  💥 {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if not passed:
            all_passed = False
    
    print("\\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Ready to run adversarial training.")
        print("\\nNext steps:")
        print("1. Run: python src/watermarking/MRI_Watermark_Embedding_CYCLES_FIXED_ADVERSARIAL.py")
        print("2. Monitor progress in: output/csv_logs_FIXED/")
        print("3. Check results in: output/watermarked_cycles_FIXED/")
    else:
        print("⚠️  SOME TESTS FAILED. Please fix issues before training.")
        print("\\nCommon fixes:")
        print("- Ensure model files are in 'pt models/' directory")
        print("- Check dataset is in 'dataset/brain-tumor-mri-dataset/'")
        print("- Install missing dependencies: torch, torchvision, opencv-python, pillow")
    
    print("=" * 50)

if __name__ == "__main__":
    main()

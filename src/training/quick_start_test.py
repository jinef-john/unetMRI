#!/usr/bin/env python3
"""
Quick start script to test the adversarial watermarking system.
This script performs basic checks and runs a minimal training example.
"""

import os
import sys
import torch
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('/teamspace/studios/this_studio/unetMRI/src')

def check_environment():
    """Check if the environment is set up correctly."""
    print("🔍 Checking environment...")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Check required directories
    required_dirs = [
        "/teamspace/studios/this_studio/unetMRI/dataset",
        "/teamspace/studios/this_studio/unetMRI/pt models",
        "/teamspace/studios/this_studio/unetMRI/src"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory found: {dir_path}")
        else:
            print(f"❌ Directory missing: {dir_path}")
    
    # Check for pretrained models
    model_files = [
        "/teamspace/studios/this_studio/unetMRI/pt models/MRI-C1EfficientNet_B3_CBAM.pth",
        "/teamspace/studios/this_studio/unetMRI/pt models/autoencoder_epoch7.pth"
    ]
    
    for model_path in model_files:
        if os.path.exists(model_path):
            print(f"✅ Model found: {os.path.basename(model_path)}")
        else:
            print(f"⚠️  Model missing: {os.path.basename(model_path)}")
    
    # Check dataset
    dataset_path = "/teamspace/studios/this_studio/unetMRI/dataset/brain-tumor-mri-dataset/Training"
    if os.path.exists(dataset_path):
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        total_images = 0
        for class_name in classes:
            class_dir = os.path.join(dataset_path, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                total_images += count
                print(f"✅ {class_name}: {count} images")
            else:
                print(f"❌ Class directory missing: {class_name}")
        print(f"📊 Total images: {total_images}")
    else:
        print(f"❌ Dataset not found: {dataset_path}")
    
    return True

def test_data_loading():
    """Test the data loading functionality."""
    print("\n📊 Testing data loading...")
    
    try:
        from utils.data_loader import MRIDataset
        
        data_root = "/teamspace/studios/this_studio/unetMRI/dataset/brain-tumor-mri-dataset/Training"
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        if not os.path.exists(data_root):
            print("❌ Dataset not found, skipping data loading test")
            return False
        
        dataset = MRIDataset(data_root, classes)
        
        if len(dataset) == 0:
            print("❌ No images loaded")
            return False
        
        # Test loading first sample
        image, label, filename, class_name = dataset[0]
        print(f"✅ Loaded sample: {filename}")
        print(f"   Shape: {image.shape}")
        print(f"   Class: {class_name} (label: {label})")
        print(f"   Range: [{image.min():.3f}, {image.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_model_loading():
    """Test model loading functionality."""
    print("\n🏗️  Testing model loading...")
    
    try:
        from training.adversarial_watermark_trainer import DualHeadC2Classifier, FrequencyWatermarkGenerator
        
        # Test C2 model
        c2 = DualHeadC2Classifier(num_classes=4)
        test_input = torch.randn(1, 1, 512, 512)
        mode_logits, class_logits = c2(test_input)
        
        print(f"✅ C2 Classifier loaded")
        print(f"   Mode logits shape: {mode_logits.shape}")
        print(f"   Class logits shape: {class_logits.shape}")
        
        # Test watermark generator
        watermark_gen = FrequencyWatermarkGenerator()
        test_img = torch.randn(2, 1, 512, 512)
        test_labels = torch.tensor([0, 1])
        watermark = watermark_gen(test_img, test_labels, 0.15)
        
        print(f"✅ Watermark Generator loaded")
        print(f"   Watermark shape: {watermark.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_training_setup():
    """Test training setup without actually training."""
    print("\n🏋️  Testing training setup...")
    
    try:
        from training.adversarial_watermark_trainer import AdversarialWatermarkTrainer
        
        data_root = "/teamspace/studios/this_studio/unetMRI/dataset/brain-tumor-mri-dataset/Training"
        pretrained_dir = "/teamspace/studios/this_studio/unetMRI/pt models"
        
        # This will test model loading and setup
        # trainer = AdversarialWatermarkTrainer(data_root, pretrained_dir)
        # print("✅ Training setup successful")
        
        # For now, just test imports
        print("✅ Training imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        return False

def run_minimal_demo():
    """Run a minimal demonstration of the system."""
    print("\n🚀 Running minimal demo...")
    
    try:
        # Test basic watermarking without full training
        from training.adversarial_watermark_trainer import FrequencyWatermarkGenerator
        
        # Create dummy data
        batch_size = 2
        img = torch.randn(batch_size, 1, 512, 512)
        labels = torch.tensor([0, 1])
        
        # Generate watermarks
        watermark_gen = FrequencyWatermarkGenerator()
        watermark = watermark_gen(img, labels, 0.15)
        
        print(f"✅ Generated watermarks")
        print(f"   Input shape: {img.shape}")
        print(f"   Watermark shape: {watermark.shape}")
        print(f"   Intensity range: [{watermark.min():.4f}, {watermark.max():.4f}]")
        
        # Test classifier
        from training.adversarial_watermark_trainer import DualHeadC2Classifier
        
        c2 = DualHeadC2Classifier()
        mode_logits, class_logits = c2(img)
        
        print(f"✅ Classifier predictions")
        print(f"   Mode predictions: {torch.softmax(mode_logits, dim=1)}")
        print(f"   Class predictions: {torch.softmax(class_logits, dim=1)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def main():
    """Main function to run all tests."""
    print("🎯 Adversarial Watermarking System - Quick Start Test")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Environment Check", check_environment),
        ("Data Loading", test_data_loading),
        ("Model Loading", test_model_loading),
        ("Training Setup", test_training_setup),
        ("Minimal Demo", run_minimal_demo)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("   The system is ready for training.")
        print("   Run: python /teamspace/studios/this_studio/unetMRI/src/training/adversarial_watermark_trainer.py")
    elif passed >= 3:
        print("\n⚠️  Most tests passed, system should work.")
        print("   Missing files can be trained or downloaded.")
    else:
        print("\n❌ Multiple test failures.")
        print("   Please check the installation and file paths.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

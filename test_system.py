#!/usr/bin/env python3
"""
Quick test script to verify the advanced model implementations work correctly.
This script performs basic functionality tests without running full training.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / 'scripts'))

def test_imports():
    """Test that all modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from advanced_model_improvements import (
            SpectralAugmentation, AdvancedFeatureExtractor, 
            TransformerAudioClassifier, ContrastiveLoss
        )
        print("✅ Advanced model improvements imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import advanced_model_improvements: {e}")
        return False
    
    try:
        from transformers import Wav2Vec2FeatureExtractor
        print("✅ Transformers library imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import transformers: {e}")
        return False
    
    try:
        import librosa
        print("✅ Librosa imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import librosa: {e}")
        return False
    
    return True

def test_feature_extraction():
    """Test feature extraction components."""
    print("\n🎵 Testing feature extraction...")
    
    try:
        from advanced_model_improvements import AdvancedFeatureExtractor
        
        # Create test audio
        sr = 16000
        duration = 2.0  # 2 seconds
        test_audio = np.random.randn(int(sr * duration)) * 0.1
        
        # Test feature extractor
        feature_extractor = AdvancedFeatureExtractor(sr=sr)
        features = feature_extractor.extract_spectral_features(test_audio)
        
        print(f"✅ Extracted features: {list(features.keys())}")
        print(f"   MFCC shape: {features['mfcc'].shape}")
        print(f"   Mel-spec shape: {features['mel_spec'].shape}")
        
        # Test feature statistics
        stats = feature_extractor.get_feature_statistics(features)
        print(f"✅ Feature statistics computed: {len(stats)} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def test_model_architecture():
    """Test model architecture components."""
    print("\n🧠 Testing model architecture...")
    
    try:
        from advanced_model_improvements import TransformerAudioClassifier
        
        # Create test model
        model = TransformerAudioClassifier(
            input_dim=768, 
            num_labels=9, 
            num_heads=8, 
            num_layers=4
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 100
        test_input = torch.randn(batch_size, seq_len, 768)
        test_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            outputs = model(test_input, test_mask)
        
        print(f"✅ Model forward pass successful")
        print(f"   Logits shape: {outputs['logits'].shape}")
        print(f"   Features shape: {outputs['features'].shape}")
        
        if 'uncertainty' in outputs:
            print(f"   Uncertainty shape: {outputs['uncertainty'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model architecture test failed: {e}")
        return False

def test_augmentation():
    """Test augmentation components."""
    print("\n🎨 Testing augmentation...")
    
    try:
        from advanced_model_improvements import SpectralAugmentation
        
        # Create test spectrogram
        time_steps = 128
        freq_bins = 80
        test_spec = torch.randn(1, time_steps, freq_bins)
        
        # Test spectral augmentation
        augmentor = SpectralAugmentation()
        augmented_spec = augmentor.spec_augment(test_spec)
        
        print(f"✅ Spectral augmentation successful")
        print(f"   Input shape: {test_spec.shape}")
        print(f"   Output shape: {augmented_spec.shape}")
        
        # Test MixUp
        test_spec2 = torch.randn(1, time_steps, freq_bins)
        mixed_spec, lam = augmentor.mixup_spectrogram(test_spec, test_spec2)
        
        print(f"✅ MixUp augmentation successful")
        print(f"   Lambda: {lam:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Augmentation test failed: {e}")
        return False

def test_loss_functions():
    """Test advanced loss functions."""
    print("\n📉 Testing loss functions...")
    
    try:
        from advanced_model_improvements import ContrastiveLoss
        
        # Test contrastive loss
        batch_size = 4
        feature_dim = 128
        num_classes = 9
        
        features = torch.randn(batch_size, feature_dim)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        contrastive_loss = ContrastiveLoss(temperature=0.1)
        loss = contrastive_loss(features, labels)
        
        print(f"✅ Contrastive loss computation successful")
        print(f"   Loss value: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n⚙️ Testing configuration...")
    
    config_path = Path(__file__).parent / 'config' / 'advanced_training_config.json'
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Configuration loaded successfully")
        print(f"   Sections: {list(config.keys())}")
        
        # Check key sections
        required_sections = [
            'model_config', 'training_config', 'advanced_features', 
            'preprocessing_config', 'evaluation_config'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"⚠️ Missing configuration sections: {missing_sections}")
        else:
            print("✅ All required configuration sections present")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("\n📊 Testing data loading...")
    
    csv_path = Path(__file__).parent / 'csv' / 'main.csv'
    
    if not csv_path.exists():
        print(f"❌ Dataset file not found: {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Dataset loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Check required columns
        required_columns = ['file_path', 'start_time', 'end_time', 'label']
        missing_columns = []
        for col in required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        # Check label distribution
        label_counts = df['label'].value_counts()
        print(f"✅ Label distribution:")
        for label, count in label_counts.items():
            print(f"   {label}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Advanced Thai Profanity Detection Model - System Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_configuration,
        test_data_loading,
        test_feature_extraction,
        test_model_architecture,
        test_augmentation,
        test_loss_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for i, test_func in enumerate(tests, 1):
        print(f"\n[{i}/{total}] Running {test_func.__name__}...")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"📋 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready for advanced training.")
        print("\nNext steps:")
        print("1. Run basic training: python scripts/ultimate_model_training.py")
        print("2. Run ablation study: python run_experiments.py --action ablation")
        print("3. Evaluate models: python scripts/comprehensive_evaluation.py")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        print("\nCommon fixes:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check file paths and directory structure")
        print("3. Verify dataset format and columns")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

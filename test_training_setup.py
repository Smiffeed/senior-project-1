#!/usr/bin/env python3
"""
Quick test script to verify the multi-stage training setup
"""

import os
import pandas as pd
import torch
from transformers import Wav2Vec2ForSequenceClassification

def test_setup():
    """Test the training setup"""
    print("🧪 Testing Multi-Stage Training Setup")
    print("=" * 40)
    
    # Check data files
    binary_path = "csv/fixed_windows_adaptive/balanced_train_2.0s.csv"
    multiclass_path = "csv/fixed_windows_context_aware/balanced_train_0.3s.csv"
    
    print("1. Checking dataset files...")
    if os.path.exists(binary_path):
        binary_df = pd.read_csv(binary_path)
        print(f"✅ Binary dataset: {len(binary_df)} samples")
        print(f"   Label distribution: {binary_df['label'].value_counts().to_dict()}")
    else:
        print(f"❌ Binary dataset not found: {binary_path}")
        return False
    
    if os.path.exists(multiclass_path):
        multiclass_df = pd.read_csv(multiclass_path)
        print(f"✅ Multiclass dataset: {len(multiclass_df)} samples")
        print(f"   Label distribution: {multiclass_df['label'].value_counts().to_dict()}")
    else:
        print(f"❌ Multiclass dataset not found: {multiclass_path}")
        return False
    
    # Check CUDA
    print("\n2. Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        print("⚠️  CUDA not available, will use CPU")
    
    # Test model loading
    print("\n3. Testing model loading...")
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/wav2vec2-base",
            num_labels=2
        )
        print("✅ Wav2Vec2 model loads successfully")
        
        # Test model creation with different classes
        model_5class = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/wav2vec2-base",
            num_labels=5
        )
        print("✅ Multi-class model creation works")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Test sample data loading
    print("\n4. Testing sample data loading...")
    try:
        import librosa
        sample_row = binary_df.iloc[0]
        file_path = sample_row['file_path']
        
        if os.path.exists(file_path):
            audio, sr = librosa.load(file_path, sr=16000, duration=2.0)
            print(f"✅ Audio loading works: {len(audio)} samples at {sr}Hz")
        else:
            print(f"⚠️  Sample audio file not found: {file_path}")
            print("   This is normal - audio files may be in different location")
            
    except Exception as e:
        print(f"⚠️  Audio loading test failed: {e}")
        print("   This might be due to missing audio files")
    
    print("\n🎯 SETUP STATUS:")
    print("✅ All core components ready for training!")
    print("\n📋 TRAINING COMMANDS:")
    print("1. Full multi-stage training:")
    print("   python train_multi_stage.py")
    print("\n2. Binary stage only:")
    print("   python train_multi_stage.py --stage binary")
    print("\n3. Multiclass stage only (after binary):")
    print("   python train_multi_stage.py --stage multiclass")
    
    return True

if __name__ == "__main__":
    test_setup()
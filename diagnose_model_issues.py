#!/usr/bin/env python3
"""
Diagnostic script to identify issues with multiclass_classifier_full model
"""

import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from pathlib import Path
import json

def test_model_predictions(model_path, test_audio_file=None):
    """Test model predictions on a small audio sample"""
    
    print(f"\n=== Testing Model: {model_path} ===")
    
    # Load model
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   - Architecture: {model.config.architectures}")
        print(f"   - Hidden size: {model.config.hidden_size}")
        print(f"   - Codevector dim: {model.config.codevector_dim}")
        print(f"   - Attention dropout: {model.config.attention_dropout}")
        print(f"   - Conv bias: {model.config.conv_bias}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load feature extractor
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        print(f"✅ Feature extractor loaded from model")
        extractor_config = feature_extractor.to_dict()
        print(f"   - Feature size: {extractor_config.get('feature_size', 'unknown')}")
        print(f"   - Sampling rate: {extractor_config.get('sampling_rate', 'unknown')}")
        print(f"   - Padding: {extractor_config.get('padding_value', 'unknown')}")
    except Exception as e:
        print(f"❌ Failed to load feature extractor from model: {e}")
        print("   Using base feature extractor...")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        extractor_config = feature_extractor.to_dict()
        print(f"   - Feature size: {extractor_config.get('feature_size', 'unknown')}")
        print(f"   - Sampling rate: {extractor_config.get('sampling_rate', 'unknown')}")
        print(f"   - Padding: {extractor_config.get('padding_value', 'unknown')}")
    
    # Test prediction on dummy audio
    print("\n🧪 Testing prediction on dummy audio...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create dummy audio (1 second of sine wave)
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    dummy_audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Extract features
    inputs = feature_extractor(
        dummy_audio, 
        sampling_rate=16000, 
        return_tensors="pt", 
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    print(f"   - Input shape: {inputs['input_values'].shape}")
    print(f"   - Logits shape: {logits.shape}")
    print(f"   - Predicted class: {predicted_class}")
    print(f"   - Class probabilities: {probabilities.cpu().numpy().flatten()}")
    
    # Label mapping
    label_map = {0: 'none', 1: 'เย็ด', 2: 'กู', 3: 'มึง', 4: 'เหี้ย'}
    print(f"   - Predicted label: {label_map.get(predicted_class, f'unknown_{predicted_class}')}")
    
    return {
        'predicted_class': predicted_class,
        'probabilities': probabilities.cpu().numpy().flatten(),
        'logits': logits.cpu().numpy().flatten()
    }

def compare_model_configs(model1_path, model2_path):
    """Compare configurations between two models"""
    
    print(f"\n=== Comparing Model Configurations ===")
    
    # Load configs
    with open(f"{model1_path}/config.json", 'r') as f:
        config1 = json.load(f)
    
    with open(f"{model2_path}/config.json", 'r') as f:
        config2 = json.load(f)
    
    # Important parameters to compare
    important_params = [
        'hidden_size', 'codevector_dim', 'attention_dropout', 'conv_bias',
        'num_conv_pos_embedding_groups', 'num_conv_pos_embeddings',
        'classifier_proj_size', 'tdnn_dim', 'tdnn_kernel', 'tdnn_dilation'
    ]
    
    print(f"\n{'Parameter':<30} {'4_classes_max_steps':<20} {'multiclass_classifier_full':<25} {'Match':<10}")
    print("-" * 85)
    
    for param in important_params:
        val1 = config1.get(param, 'N/A')
        val2 = config2.get(param, 'N/A')
        match = "✅" if val1 == val2 else "❌"
        print(f"{param:<30} {str(val1):<20} {str(val2):<25} {match:<10}")

def main():
    """Main diagnostic function"""
    
    print("🔍 DIAGNOSING MODEL COMPATIBILITY ISSUES")
    print("=" * 50)
    
    # Test both models
    model1_path = "./models/4_classes_max_steps"
    model2_path = "./models/multiclass_classifier_full"
    
    # Test model 1 (known working)
    result1 = test_model_predictions(model1_path)
    
    # Test model 2 (problematic)  
    result2 = test_model_predictions(model2_path)
    
    # Compare configurations
    compare_model_configs(model1_path, model2_path)
    
    # Compare predictions
    if result1 and result2:
        print(f"\n=== Prediction Comparison ===")
        print(f"4_classes_max_steps prediction: {result1['predicted_class']}")
        print(f"multiclass_classifier_full prediction: {result2['predicted_class']}")
        
        print(f"\nLogits comparison:")
        print(f"4_classes_max_steps: {result1['logits']}")
        print(f"multiclass_classifier_full: {result2['logits']}")
        
        # Check if multiclass_classifier_full is stuck on one class
        probs2 = result2['probabilities']
        max_prob = max(probs2)
        if max_prob > 0.9:
            stuck_class = np.argmax(probs2)
            label_map = {0: 'none', 1: 'เย็ด', 2: 'กู', 3: 'มึง', 4: 'เหี้ย'}
            print(f"\n⚠️  WARNING: multiclass_classifier_full appears stuck on class {stuck_class} ({label_map[stuck_class]}) with {max_prob:.3f} probability")
    
    print(f"\n=== RECOMMENDATIONS ===")
    print("1. Use 4_classes_max_steps model - it has proper preprocessor_config.json")
    print("2. If you must use multiclass_classifier_full, retrain with consistent preprocessing")
    print("3. Check if models were trained with same audio preprocessing pipeline")
    print("4. Consider using the same feature extractor for both training and evaluation")

if __name__ == "__main__":
    main()
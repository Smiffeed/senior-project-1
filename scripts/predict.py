import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import numpy as np

def load_model_and_predict_with_timestamps(model_path, audio_file_path, window_size=0.5, overlap=0.25, confidence_threshold=0.7):
    # Label mapping (same as training)
    label_map = {
        0: 'none',
        1: 'เย็ด',
        2: 'กู', 
        3: 'มึง',
        4: 'เหี้ย',
        5: 'ควย',
        6: 'สวะ',
        7: 'หี',
        8: 'แตด'
    }
    
    # Load the trained model AND feature extractor (CRITICAL!)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model.eval()
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    def preprocess_audio_segment(waveform, sample_rate, target_sr=16000):
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
        
        # CRITICAL: Use same normalization as test_wav2vec2_model.py
        audio = waveform.squeeze().numpy()
        audio = (audio - audio.mean()) / (audio.std() + 1e-7)
        
        return audio
    
    # Load the full audio file
    full_waveform, sample_rate = torchaudio.load(audio_file_path)
    total_duration = full_waveform.shape[1] / sample_rate
    
    print(f"Audio file: {audio_file_path}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Window size: {window_size}s, Overlap: {overlap}s")
    print(f"Confidence threshold: {confidence_threshold}")
    print("=" * 60)
    
    results = []
    current_time = 0.0
    step_size = window_size - overlap
    window_count = 0
    
    while current_time < total_duration:
        window_count += 1
        
        # Calculate window boundaries
        start_time = current_time
        end_time = min(current_time + window_size, total_duration)
        
        # Extract audio segment
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment = full_waveform[:, start_sample:end_sample]
        
        # Skip if segment is too short
        min_samples = int(sample_rate * 0.1)  # Minimum 0.1 seconds
        if segment.shape[1] < min_samples:
            break
        
        # Preprocess segment (returns numpy array)
        processed_segment = preprocess_audio_segment(segment, sample_rate)
        
        # CRITICAL: Use feature extractor like test_wav2vec2_model.py
        inputs = feature_extractor(
            processed_segment, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        input_values = inputs.input_values.to(device)
        
        # CRITICAL: Handle attention mask properly
        if 'attention_mask' not in inputs:
            attention_mask = torch.ones_like(input_values)
        else:
            attention_mask = inputs.attention_mask.to(device)
        
        # Make prediction with attention mask
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        # Get label name
        predicted_label = label_map[predicted_class]
        
        # Store result
        result = {
            'window': window_count,
            'start_time': start_time,
            'end_time': end_time,
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'all_probabilities': predictions[0].cpu().numpy()
        }
        results.append(result)
        
        # Print result with window number
        status_icon = "🚨" if (predicted_label != 'none' and confidence > confidence_threshold) else "✅"
        print(f"Window {window_count:3d} | {start_time:6.2f}s-{end_time:6.2f}s | "
              f"{predicted_label:8s} | {confidence:6.4f} {status_icon}")
        
        # Debug: Print all probabilities like test_wav2vec2_model.py
        print(f"    All probabilities:")
        for i, (label, prob) in enumerate(zip(label_map.values(), predictions[0].cpu().numpy())):
            print(f"    - {label}: {prob:.3f}")
        
        # If profanity detected, show detailed breakdown
        if predicted_label != 'none' and confidence > confidence_threshold:
            print(f"    🚨 PROFANITY: '{predicted_label}' detected!")
        
        current_time += step_size
    
    return results

# Add this after preprocessing to filter extreme lengths
def filter_extreme_lengths(processed, min_length=1600, max_length=160000):  # 0.1s to 10s
    filtered = []
    for item in processed:
        length = len(item['input_values'])
        if min_length <= length <= max_length:
            filtered.append(item)
        else:
            print(f"Filtered out sample with length {length}")
    return filtered

def merge_consecutive_detections(results, confidence_threshold=0.7, max_gap=0.5):
    """Merge consecutive profanity detections of the same type"""
    merged_detections = []
    
    profanity_results = [r for r in results 
                        if r['predicted_label'] != 'none' and r['confidence'] > confidence_threshold]
    
    if not profanity_results:
        return merged_detections
    
    current_detection = profanity_results[0].copy()
    
    for i in range(1, len(profanity_results)):
        prev = current_detection
        curr = profanity_results[i]
        
        # Check if same profanity type and close in time
        time_gap = curr['start_time'] - prev['end_time']
        same_type = curr['predicted_label'] == prev['predicted_label']
        
        if same_type and time_gap <= max_gap:
            # Merge detections
            current_detection['end_time'] = curr['end_time']
            current_detection['confidence'] = max(current_detection['confidence'], curr['confidence'])
        else:
            # Save current detection and start new one
            merged_detections.append(current_detection)
            current_detection = curr.copy()
    
    # Add the last detection
    merged_detections.append(current_detection)
    
    return merged_detections

def print_summary(results, confidence_threshold=0.7):
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    
    # Count profanity detections
    profanity_count = sum(1 for r in results if r['predicted_label'] != 'none' and r['confidence'] > confidence_threshold)
    total_segments = len(results)
    
    print(f"Total windows analyzed: {total_segments}")
    print(f"Profanity windows detected: {profanity_count}")
    print(f"Clean windows: {total_segments - profanity_count}")
    
    # Merge consecutive detections
    merged_detections = merge_consecutive_detections(results, confidence_threshold)
    
    if merged_detections:
        print(f"\nMerged profanity segments: {len(merged_detections)}")
        print("\nDetailed breakdown:")
        
        for i, detection in enumerate(merged_detections, 1):
            duration = detection['end_time'] - detection['start_time']
            print(f"  {i}. '{detection['predicted_label']}' | "
                  f"{detection['start_time']:6.2f}s-{detection['end_time']:6.2f}s | "
                  f"Duration: {duration:.2f}s | "
                  f"Confidence: {detection['confidence']:.4f}")
    else:
        print("\n✅ No profanity detected!")

def compare_detection_methods(results_coarse, results_fine):
    """Compare detection results between different window sizes"""
    print("\n" + "=" * 60)
    print("DETECTION METHOD COMPARISON")
    print("=" * 60)
    
    coarse_detections = len([r for r in results_coarse if r['predicted_label'] != 'none' and r['confidence'] > 0.7])
    fine_detections = len([r for r in results_fine if r['predicted_label'] != 'none' and r['confidence'] > 0.7])
    
    print(f"Coarse method (larger windows): {coarse_detections} detections")
    print(f"Fine method (smaller windows): {fine_detections} detections")
    print(f"Detection difference: {abs(fine_detections - coarse_detections)}")
    
    if fine_detections > coarse_detections:
        print("✨ Fine method detected more instances (higher sensitivity)")
    elif coarse_detections > fine_detections:
        print("⚖️ Coarse method may have more false positives")
    else:
        print("🎯 Both methods found similar results")

def analyze_model_confidence(results):
    """Analyze and plot the model's confidence distribution"""
    import matplotlib.pyplot as plt
    
    # Extract confidence scores
    confidence_scores = [r['confidence'] for r in results]
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(confidence_scores, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.title("Model Confidence Score Distribution")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def comprehensive_test(model_path, audio_file, window_size=1.0, overlap=0.5, confidence_threshold=0.7):
    """Run a comprehensive test with various window sizes and thresholds"""
    all_results = {}
    
    # Test with coarse and fine window sizes
    for ws in [0.5, 1.0, 2.0]:
        for ov in [0.25, 0.5]:
            key = f"ws{ws}_ov{ov}"
            print(f"\nTesting Window Size: {ws}s, Overlap: {ov}s")
            results = load_model_and_predict_with_timestamps(
                model_path, 
                audio_file, 
                window_size=ws, 
                overlap=ov, 
                confidence_threshold=confidence_threshold
            )
            all_results[key] = results
    
    # Compare detection methods
    compare_detection_methods(all_results['ws0.5_ov0.25'], all_results['ws1.0_ov0.5'])
    compare_detection_methods(all_results['ws1.0_ov0.5'], all_results['ws2.0_ov0.5'])

def batch_predict(model_path, audio_file, batch_size=8, window_size=1.0, overlap=0.5):
    """Process multiple windows in batches for better efficiency"""
    # Load model once
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    label_map = {0: 'none', 1: 'เย็ด', 2: 'กู', 3: 'มึง', 4: 'เหี้ย', 5: 'ควย', 6: 'สวะ', 7: 'หี', 8: 'แตด'}
    
    # Load audio and prepare segments
    full_waveform, sample_rate = torchaudio.load(audio_file)
    total_duration = full_waveform.shape[1] / sample_rate
    
    segments_data = []
    current_time = 0.0
    step_size = window_size - overlap
    
    while current_time < total_duration:
        start_time = current_time
        end_time = min(current_time + window_size, total_duration)
        
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment = full_waveform[:, start_sample:end_sample]
        
        if segment.shape[1] < int(sample_rate * 0.1):
            break
        
        # Preprocess
        if segment.shape[0] > 1:
            segment = segment.mean(dim=0, keepdim=True)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            segment = resampler(segment)
        
        audio = segment.squeeze().numpy()
        audio = (audio - audio.mean()) / (audio.std() + 1e-7)
        
        segments_data.append({
            'audio': audio,
            'start_time': start_time,
            'end_time': end_time,
            'window': len(segments_data) + 1
        })
        
        current_time += step_size
    
    # Process in batches
    results = []
    for i in range(0, len(segments_data), batch_size):
        batch = segments_data[i:i+batch_size]
        batch_audio = [seg['audio'] for seg in batch]
        
        inputs = feature_extractor(
            batch_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if 'attention_mask' in inputs else torch.ones_like(input_values)
        
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.nn.functional.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(logits, dim=-1)
            confidences = torch.max(predictions, dim=-1)[0]
        
        for j, seg_data in enumerate(batch):
            predicted_class = predicted_classes[j].item()
            confidence = confidences[j].item()
            predicted_label = label_map[predicted_class]
            
            results.append({
                'window': seg_data['window'],
                'start_time': seg_data['start_time'],
                'end_time': seg_data['end_time'],
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'all_probabilities': predictions[j].cpu().numpy()
            })
    
    return results

# Example usage
if __name__ == '__main__':
    model_path = "./output/model"
    audio_file = "test1.wav"
    
    # 1. Quick single test
    print("🎯 QUICK TEST")
    print("=" * 60)
    results = load_model_and_predict_with_timestamps(
        model_path, 
        audio_file, 
        window_size=1.0,     # 1 second windows
        overlap=0.5,         # 50% overlap
        confidence_threshold=0.4  # Lower threshold
    )
    
    print_summary(results, confidence_threshold=0.4)
    
    # 2. Confidence analysis
    analyze_model_confidence(results)
    
    # 3. Comprehensive testing
    print("\n" + "=" * 80)
    comprehensive_results = comprehensive_test(model_path, audio_file)
    
    # 4. Test with different thresholds on best config
    print("\n🎚️ THRESHOLD SENSITIVITY TEST")
    print("=" * 60)
    best_results = load_model_and_predict_with_timestamps(
        model_path, 
        audio_file, 
        window_size=1.0,
        overlap=0.5,
        confidence_threshold=0.1  # Very low to see all predictions
    )
    
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print("Threshold | Detections | Merged Segments")
    print("-" * 40)
    for threshold in thresholds:
        detections = [r for r in best_results if r['predicted_label'] != 'none' and r['confidence'] > threshold]
        merged = merge_consecutive_detections(best_results, threshold)
        print(f"   {threshold:.1f}    |     {len(detections):2d}     |       {len(merged):2d}")
    
    print("\n✅ Testing complete!")
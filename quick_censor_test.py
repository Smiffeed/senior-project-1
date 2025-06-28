"""
Production-Ready Audio Censoring Test

This script implements full preprocessing pipeline for optimal profanity detection.
Includes proper audio preprocessing to match training pipeline for maximum accuracy.
"""

import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf

# Add scripts to path
sys.path.append('./scripts')

class ProductionAudioPreprocessor:
    """Production-ready audio preprocessing pipeline for optimal detection."""
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def preprocess_audio_segment(self, audio_segment):
        """Apply full preprocessing pipeline to audio segment."""
        try:
            # 1. Pre-emphasis filter (improves high-frequency detection)
            audio_segment = self._apply_pre_emphasis(audio_segment)
            
            # 2. Simple noise reduction
            audio_segment = self._apply_simple_noise_reduction(audio_segment)
            
            # 3. RMS normalization
            audio_segment = self._apply_rms_normalization(audio_segment)
            
            # 4. Apply Hamming window for spectral clarity
            audio_segment = self._apply_hamming_window(audio_segment)
            
            # 5. Ensure proper range and no NaN/Inf values
            audio_segment = self._validate_audio(audio_segment)
            
            return audio_segment
            
        except Exception as e:
            print(f"⚠️ Warning: Preprocessing error: {e}")
            # Return original segment if preprocessing fails
            return audio_segment
    
    def _apply_pre_emphasis(self, audio, coeff=0.97):
        """Apply pre-emphasis filter to enhance high frequencies."""
        if len(audio) < 2:
            return audio
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])
    
    def _apply_simple_noise_reduction(self, audio, threshold=0.005):
        """Simple threshold-based noise reduction."""
        # Suppress low-amplitude noise
        audio = np.where(np.abs(audio) < threshold, 0, audio)
        return audio
    
    def _apply_rms_normalization(self, audio):
        """Apply RMS normalization for consistent volume."""
        rms = np.sqrt(np.mean(audio**2))
        if rms > 1e-8:  # Avoid division by zero
            audio = audio / (rms + 1e-8)
        return audio
    
    def _apply_hamming_window(self, audio):
        """Apply Hamming window for better spectral characteristics."""
        if len(audio) > 1:
            window = np.hamming(len(audio))
            audio = audio * window
        return audio
    
    def _validate_audio(self, audio):
        """Ensure audio is in valid range and contains no invalid values."""
        # Remove NaN and Inf values
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip to reasonable range
        audio = np.clip(audio, -10.0, 10.0)
        
        return audio

def test_production_audio_censoring():
    """Test audio censoring with full production preprocessing pipeline."""
    
    # Choose test files (you can modify this list)
    test_files = [
        './test.wav',
        './eval/ถามจริง กูจะรั่ว.wav',
        './eval/พูดอะไรเนี่ย เย็ดแม่ ฟังไม่รู้เรื่อง.wav',
        './eval/มึง กูหิวข้าววะ อยากกินตีนอะ.wav'
    ]
    
    # Find first available test file
    input_file = None
    for file in test_files:
        if os.path.exists(file):
            input_file = file
            break
    
    if not input_file:
        print("❌ No test files found!")
        print("Available files to test:")
        if os.path.exists('./eval'):
            for f in os.listdir('./eval')[:5]:
                if f.endswith('.wav'):
                    print(f"  ./eval/{f}")
        return
    
    print(f"🎵 Testing with: {input_file}")
    print("🔧 Using PRODUCTION preprocessing pipeline")
    
    try:
        # Import required modules
        from simplified_ultimate_training import EnhancedAudioClassifier
        from transformers import Wav2Vec2FeatureExtractor
        
        # Constants
        LABEL_MAP = {'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4, 'ควย': 5, 'สวะ': 6, 'หี': 7, 'แตด': 8}
        CLASS_NAMES = list(LABEL_MAP.keys())
        
        # Initialize production components
        model_dir = './models/simplified_advanced_audio_train'
        model_name = "airesearch/wav2vec2-large-xlsr-53-th"
        
        print("🤖 Loading production components...")
        
        # Load feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, return_attention_mask=True, do_normalize=True
        )
        
        # Load preprocessor
        preprocessor = ProductionAudioPreprocessor()
        
        # Load trained model
        model_path = os.path.join(model_dir, 'fold_1', 'best_model.pt')
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        
        model = EnhancedAudioClassifier(model_name, len(LABEL_MAP))
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("✅ Production components loaded successfully!")
        
        # Load and preprocess audio
        print("🎵 Loading audio...")
        audio, sr = librosa.load(input_file, sr=16000)
        audio_duration = len(audio) / sr
        print(f"✅ Audio loaded: {audio_duration:.2f} seconds")
        
        # Production parameters
        window_size = 0.25  # seconds
        hop_length = 0.125  # seconds
        confidence_threshold = 0.7  # Production threshold
        
        print("🔍 Detecting profanities with production preprocessing...")
        detections = []
        censored_audio = audio.copy()
        
        # Sliding window detection with full preprocessing
        audio_length = len(audio) / sr
        window_starts = np.arange(0, audio_length - window_size, hop_length)
        
        detection_count = 0
        total_windows = len(window_starts)
        
        for i, window_start in enumerate(window_starts):
            if i % 5 == 0:  # Progress indicator
                print(f"   Processing window {i+1}/{total_windows} ({(i/total_windows)*100:.1f}%)")
            
            window_end = window_start + window_size
            
            # Extract segment
            start_sample = int(window_start * sr)
            end_sample = int(window_end * sr)
            audio_segment = audio[start_sample:end_sample]
            
            # Pad if needed
            if len(audio_segment) < int(window_size * sr):
                audio_segment = np.pad(audio_segment, 
                                     (0, int(window_size * sr) - len(audio_segment)), 
                                     'constant')
            
            try:
                # APPLY FULL PRODUCTION PREPROCESSING
                print(f"   🔧 Before preprocessing: range [{audio_segment.min():.3f}, {audio_segment.max():.3f}]") if i == 0 else None
                audio_segment = preprocessor.preprocess_audio_segment(audio_segment)
                print(f"   ✅ After preprocessing: range [{audio_segment.min():.3f}, {audio_segment.max():.3f}]") if i == 0 else None
                
                # Feature extraction
                inputs = feature_extractor(
                    audio_segment, sampling_rate=sr, return_tensors="pt", padding=True
                )
                
                # Model prediction
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs['logits'], dim=-1)
                    prediction = torch.argmax(outputs['logits'], dim=-1).item()
                    confidence = probs.max().item()
                
                # Check if profanity detected with sufficient confidence
                if prediction != 0 and confidence >= confidence_threshold:
                    detection_count += 1
                    class_name = CLASS_NAMES[prediction]
                    
                    detections.append({
                        'start_time': window_start,
                        'end_time': window_end,
                        'start_sample': start_sample,
                        'end_sample': end_sample,
                        'class': class_name,
                        'confidence': confidence,
                        'all_probs': probs.squeeze().cpu().numpy()
                    })
                    
                    print(f"🚨 DETECTION #{detection_count}: {class_name} at {window_start:.2f}s-{window_end:.2f}s (confidence: {confidence:.3f})")
                    
                    # Apply censoring (beep method)
                    duration = (end_sample - start_sample) / sr
                    t = np.linspace(0, duration, end_sample - start_sample)
                    beep = 0.3 * np.sin(2 * np.pi * 1000 * t)  # 1000 Hz beep
                    censored_audio[start_sample:end_sample] = beep
                    
            except Exception as e:
                print(f"⚠️ Warning: Error processing window at {window_start:.2f}s: {e}")
                continue
        
        # Merge nearby detections for cleaner results
        if detections:
            print(f"\n🔧 Merging nearby detections...")
            merged_detections = merge_nearby_detections(detections, merge_threshold=0.1)
            print(f"   Merged {len(detections)} raw detections into {len(merged_detections)} final detections")
            detections = merged_detections
        
        # Save results
        print(f"\n📊 PRODUCTION RESULTS:")
        print(f"   Total detections: {len(detections)}")
        print(f"   Confidence threshold: {confidence_threshold}")
        print(f"   Preprocessing: FULL PIPELINE ✅")
        
        if detections:
            print(f"\n🚨 DETECTED PROFANITIES:")
            total_censored_duration = 0
            for i, detection in enumerate(detections, 1):
                duration = detection['end_time'] - detection['start_time']
                total_censored_duration += duration
                print(f"   {i}. {detection['class']} at {detection['start_time']:.2f}s-{detection['end_time']:.2f}s")
                print(f"      Confidence: {detection['confidence']:.3f}, Duration: {duration:.2f}s")
            
            print(f"\n📈 CENSORING SUMMARY:")
            print(f"   Total censored duration: {total_censored_duration:.2f}s")
            print(f"   Percentage censored: {(total_censored_duration/audio_duration)*100:.1f}%")
            
            # Save censored audio
            output_file = './production_censored_output.wav'
            sf.write(output_file, censored_audio, sr)
            print(f"\n✅ Censored audio saved to: {output_file}")
            
            # Save original for comparison
            comparison_file = './production_original_copy.wav'
            sf.write(comparison_file, audio, sr)
            print(f"📄 Original copy saved to: {comparison_file}")
            
            # Save detailed report
            save_production_report(input_file, output_file, detections, audio_duration, confidence_threshold)
            
            return output_file
            
        else:
            print("✨ No profanities detected - audio is clean!")
            print("💡 Try lowering confidence_threshold if you expect profanities")
            return input_file
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the project root directory")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def merge_nearby_detections(detections, merge_threshold=0.1):
    """Merge detections that are close together for cleaner censoring."""
    if not detections:
        return []
    
    # Sort by start time
    detections = sorted(detections, key=lambda x: x['start_time'])
    merged = [detections[0]]
    
    for current in detections[1:]:
        last_merged = merged[-1]
        
        # Check if current detection is close to the last merged one
        if current['start_time'] <= last_merged['end_time'] + merge_threshold:
            # Merge: extend the end time and take the highest confidence class
            if current['confidence'] > last_merged['confidence']:
                last_merged['class'] = current['class']
                last_merged['confidence'] = current['confidence']
            
            last_merged['end_time'] = max(last_merged['end_time'], current['end_time'])
            last_merged['end_sample'] = max(last_merged['end_sample'], current['end_sample'])
        else:
            # No overlap, add as new detection
            merged.append(current)
    
    return merged

def save_production_report(input_file, output_file, detections, duration, threshold):
    """Save detailed production report."""
    from datetime import datetime
    import json
    
    report = {
        'production_report': {
            'timestamp': datetime.now().isoformat(),
            'input_file': input_file,
            'output_file': output_file,
            'preprocessing': 'FULL_PRODUCTION_PIPELINE',
            'total_duration': duration,
            'confidence_threshold': threshold,
            'total_detections': len(detections),
            'total_censored_duration': sum(d['end_time'] - d['start_time'] for d in detections),
            'detections': []
        }
    }
    
    for i, detection in enumerate(detections, 1):
        det_info = {
            'detection_id': i,
            'class': detection['class'],
            'start_time': detection['start_time'],
            'end_time': detection['end_time'],
            'duration': detection['end_time'] - detection['start_time'],
            'confidence': detection['confidence'],
            'all_probabilities': detection['all_probs'].tolist() if 'all_probs' in detection else []
        }
        report['production_report']['detections'].append(det_info)
    
    report_file = './production_censoring_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Detailed report saved to: {report_file}")

def test_alternative_files():
    """Test with multiple files using production preprocessing."""
    
    print("🔍 Testing multiple files with PRODUCTION preprocessing...")
    
    # Get all wav files in eval directory
    eval_dir = './eval'
    if not os.path.exists(eval_dir):
        print("❌ Eval directory not found")
        return
    
    wav_files = [f for f in os.listdir(eval_dir) if f.endswith('.wav')]
    print(f"📁 Found {len(wav_files)} audio files")
    
    # Test first 3 files
    for i, filename in enumerate(wav_files[:3], 1):
        filepath = os.path.join(eval_dir, filename)
        print(f"\n📁 Test {i}/{min(3, len(wav_files))}: {filename}")
        
        try:
            # Test with production preprocessing
            result = test_single_file_production(filepath, f'./multi_test_output_{i}.wav')
            
            if result and result.get('detections', 0) > 0:
                print(f"✅ Found {result['detections']} detections!")
                print(f"🎵 Output saved to: {result['output_file']}")
                return filepath
            else:
                print(f"ℹ️ No detections found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    print("ℹ️ No profanities detected in test files - try lowering confidence threshold")

def test_single_file_production(input_file, output_file):
    """Test single file with production preprocessing and actually save censored audio."""
    try:
        from simplified_ultimate_training import EnhancedAudioClassifier
        from transformers import Wav2Vec2FeatureExtractor
        import librosa
        import soundfile as sf
        import torch
        import numpy as np
        
        print(f"🔄 Processing: {input_file}")
        print(f"💾 Will save to: {output_file}")
        
        # Constants
        LABEL_MAP = {'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4, 'ควย': 5, 'สวะ': 6, 'หี': 7, 'แตด': 8}
        CLASS_NAMES = list(LABEL_MAP.keys())
        
        # Load components
        preprocessor = ProductionAudioPreprocessor()
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "airesearch/wav2vec2-large-xlsr-53-th", return_attention_mask=True, do_normalize=True
        )
        
        # Load model
        model_path = './models/simplified_advanced_audio_train/fold_1/best_model.pt'
        model = EnhancedAudioClassifier("airesearch/wav2vec2-large-xlsr-53-th", len(LABEL_MAP))
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Load audio
        audio, sr = librosa.load(input_file, sr=16000)
        censored_audio = audio.copy()  # Create copy for censoring
        
        # Process with sliding windows
        detections_list = []
        detection_count = 0
        window_size = 0.25
        hop_length = 0.125
        confidence_threshold = 0.7
        
        audio_length = len(audio) / sr
        window_starts = np.arange(0, audio_length - window_size, hop_length)
        
        print(f"🔍 Analyzing {len(window_starts)} windows...")
        
        for i, window_start in enumerate(window_starts):
            if i % 10 == 0:  # Progress update
                print(f"   Progress: {i+1}/{len(window_starts)} windows ({((i+1)/len(window_starts))*100:.1f}%)")
            
            window_end = window_start + window_size
            start_sample = int(window_start * sr)
            end_sample = int(window_end * sr)
            audio_segment = audio[start_sample:end_sample]
            
            if len(audio_segment) < int(window_size * sr):
                audio_segment = np.pad(audio_segment, 
                                     (0, int(window_size * sr) - len(audio_segment)), 
                                     'constant')
            
            # Apply production preprocessing
            audio_segment = preprocessor.preprocess_audio_segment(audio_segment)
            
            # Model prediction
            inputs = feature_extractor(audio_segment, sampling_rate=sr, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs['logits'], dim=-1)
                prediction = torch.argmax(outputs['logits'], dim=-1).item()
                confidence = probs.max().item()
            
            if prediction != 0 and confidence >= confidence_threshold:
                detection_count += 1
                class_name = CLASS_NAMES[prediction]
                
                # Store detection info
                detections_list.append({
                    'start_time': window_start,
                    'end_time': window_end,
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'class': class_name,
                    'confidence': confidence
                })
                
                print(f"🚨 DETECTION #{detection_count}: {class_name} at {window_start:.2f}s-{window_end:.2f}s (confidence: {confidence:.3f})")
                
                # Apply censoring with beep sound
                duration = (end_sample - start_sample) / sr
                t = np.linspace(0, duration, end_sample - start_sample)
                beep = 0.3 * np.sin(2 * np.pi * 1000 * t)  # 1000 Hz beep
                censored_audio[start_sample:end_sample] = beep
        
        # Merge nearby detections for cleaner results
        if detections_list:
            merged_detections = merge_nearby_detections(detections_list, merge_threshold=0.1)
            print(f"🔧 Merged {len(detections_list)} raw detections into {len(merged_detections)} final detections")
        else:
            merged_detections = []
        
        # Always save the audio file (even if no detections)
        print(f"💾 Saving audio to: {output_file}")
        sf.write(output_file, censored_audio, sr)
        
        # Verify file was saved
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ File saved successfully! Size: {file_size:,} bytes")
        else:
            print(f"❌ Warning: File was not saved to {output_file}")
        
        return {
            'detections': len(merged_detections),
            'raw_detections': len(detections_list),
            'output_file': output_file,
            'input_file': input_file,
            'file_saved': os.path.exists(output_file),
            'detections_list': merged_detections
        }
        
    except Exception as e:
        print(f"❌ Error in single file test: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_window_sizes(input_file, window_sizes=[0.5, 0.25], confidence_threshold=0.5):
    """
    Compare detection accuracy between different window sizes.
    
    Args:
        input_file: Path to audio file to test
        window_sizes: List of window sizes to compare (in seconds)
        confidence_threshold: Minimum confidence for detection
    
    Returns:
        Dictionary with comparison results
    """
    print(f"\n🔬 WINDOW SIZE COMPARISON ANALYSIS")
    print(f"📁 File: {input_file}")
    print(f"🎯 Confidence threshold: {confidence_threshold}")
    print("=" * 60)
    
    try:
        # Try to import from scripts directory first
        sys.path.append('./scripts')
        try:
            from simplified_ultimate_training import EnhancedAudioClassifier
        except ImportError:
            # Fallback: try current directory
            sys.path.append('.')
            from simplified_ultimate_training import EnhancedAudioClassifier
        
        from transformers import Wav2Vec2FeatureExtractor
        
        # Constants
        LABEL_MAP = {'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4, 'ควย': 5, 'สวะ': 6, 'หี': 7, 'แตด': 8}
        CLASS_NAMES = list(LABEL_MAP.keys())
        
        # Load components once
        preprocessor = ProductionAudioPreprocessor()
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "airesearch/wav2vec2-large-xlsr-53-th", return_attention_mask=True, do_normalize=True
        )
        
        model_path = './models/simplified_advanced_audio_train/fold_1/best_model.pt'
        model = EnhancedAudioClassifier("airesearch/wav2vec2-large-xlsr-53-th", len(LABEL_MAP))
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Load audio once
        audio, sr = librosa.load(input_file, sr=16000)
        audio_duration = len(audio) / sr
        
        results = {}
        
        for window_size in window_sizes:
            print(f"\n🔍 Testing window size: {window_size}s")
            
            hop_length = window_size / 2  # 50% overlap
            detections = []
            total_windows = 0
            processing_time = 0
            
            audio_length = len(audio) / sr
            window_starts = np.arange(0, audio_length - window_size, hop_length)
            total_windows = len(window_starts)
            
            import time
            start_time = time.time()
            
            for window_start in window_starts:
                window_end = window_start + window_size
                start_sample = int(window_start * sr)
                end_sample = int(window_end * sr)
                audio_segment = audio[start_sample:end_sample]
                
                # Pad if needed
                if len(audio_segment) < int(window_size * sr):
                    audio_segment = np.pad(audio_segment, 
                                         (0, int(window_size * sr) - len(audio_segment)), 
                                         'constant')
                
                try:
                    # Apply production preprocessing
                    audio_segment = preprocessor.preprocess_audio_segment(audio_segment)
                    
                    # Feature extraction and prediction
                    inputs = feature_extractor(audio_segment, sampling_rate=sr, return_tensors="pt", padding=True)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs['logits'], dim=-1)
                        prediction = torch.argmax(outputs['logits'], dim=-1).item()
                        confidence = probs.max().item()
                    
                    if prediction != 0 and confidence >= confidence_threshold:
                        class_name = CLASS_NAMES[prediction]
                        detections.append({
                            'start_time': window_start,
                            'end_time': window_end,
                            'start_sample': start_sample,
                            'end_sample': end_sample,
                            'class': class_name,
                            'confidence': confidence,
                            'window_size': window_size
                        })
                
                except Exception as e:
                    continue
            
            processing_time = time.time() - start_time
            
            # Merge nearby detections
            merged_detections = merge_nearby_detections(detections, merge_threshold=0.1)
            
            # Calculate metrics
            total_detected_duration = sum(d['end_time'] - d['start_time'] for d in merged_detections)
            avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0
            max_confidence = max([d['confidence'] for d in detections]) if detections else 0
            
            results[window_size] = {
                'window_size': window_size,
                'hop_length': hop_length,
                'total_windows': total_windows,
                'raw_detections': len(detections),
                'merged_detections': len(merged_detections),
                'total_detected_duration': total_detected_duration,
                'percentage_detected': (total_detected_duration / audio_duration) * 100,
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'processing_time': processing_time,
                'detections_per_second': len(detections) / audio_duration if audio_duration > 0 else 0,
                'detailed_detections': merged_detections
            }
            
            print(f"   📊 Raw detections: {len(detections)}")
            print(f"   🔗 Merged detections: {len(merged_detections)}")
            print(f"   📈 Avg confidence: {avg_confidence:.3f}")
            print(f"   🎯 Max confidence: {max_confidence:.3f}")
            print(f"   ⏱️ Processing time: {processing_time:.2f}s")
            print(f"   🎵 Detected duration: {total_detected_duration:.2f}s ({(total_detected_duration/audio_duration)*100:.1f}%)")
        
        # Analysis and recommendations
        print(f"\n" + "=" * 60)
        print("📊 WINDOW SIZE COMPARISON RESULTS")
        print("=" * 60)
        
        for window_size, result in results.items():
            print(f"\n🔹 Window Size: {window_size}s")
            print(f"   Total windows processed: {result['total_windows']}")
            print(f"   Detections found: {result['merged_detections']}")
            print(f"   Average confidence: {result['avg_confidence']:.3f}")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   Detection density: {result['detections_per_second']:.2f} detections/second")
        
        # Recommendations
        print(f"\n💡 ANALYSIS & RECOMMENDATIONS:")
        
        if len(window_sizes) >= 2:
            smaller_ws = min(window_sizes)
            larger_ws = max(window_sizes)
            
            smaller_result = results[smaller_ws]
            larger_result = results[larger_ws]
            
            print(f"\n🔍 Smaller windows ({smaller_ws}s) vs Larger windows ({larger_ws}s):")
            
            # Detection count comparison
            if smaller_result['merged_detections'] > larger_result['merged_detections']:
                print(f"   ✅ Smaller windows found MORE detections ({smaller_result['merged_detections']} vs {larger_result['merged_detections']})")
                print(f"      → Better for detecting SHORT profanities")
            elif smaller_result['merged_detections'] < larger_result['merged_detections']:
                print(f"   ✅ Larger windows found MORE detections ({larger_result['merged_detections']} vs {smaller_result['merged_detections']})")
                print(f"      → Better for OVERALL context understanding")
            else:
                print(f"   ⚖️ Same number of detections ({smaller_result['merged_detections']})")
            
            # Confidence comparison
            if smaller_result['avg_confidence'] > larger_result['avg_confidence']:
                print(f"   🎯 Smaller windows have HIGHER confidence ({smaller_result['avg_confidence']:.3f} vs {larger_result['avg_confidence']:.3f})")
                print(f"      → More precise detection")
            else:
                print(f"   🎯 Larger windows have HIGHER confidence ({larger_result['avg_confidence']:.3f} vs {smaller_result['avg_confidence']:.3f})")
                print(f"      → More stable detection")
            
            # Processing time comparison
            efficiency_ratio = smaller_result['processing_time'] / larger_result['processing_time']
            print(f"   ⏱️ Processing time ratio: {efficiency_ratio:.2f}x")
            if efficiency_ratio > 1.5:
                print(f"      → Smaller windows are significantly SLOWER")
            elif efficiency_ratio > 1.1:
                print(f"      → Smaller windows are slightly slower")
            else:
                print(f"      → Similar processing times")
        
        print(f"\n🎯 FINAL RECOMMENDATION:")
        
        # Determine best window size based on multiple factors
        best_window = None
        best_score = -1
        
        for window_size, result in results.items():
            # Composite score: detections + confidence - processing_penalty
            score = (result['merged_detections'] * 2 + 
                    result['avg_confidence'] * 10 + 
                    result['max_confidence'] * 5 -
                    (result['processing_time'] / audio_duration))  # Processing penalty
            
            if score > best_score:
                best_score = score
                best_window = window_size
        
        if best_window:
            print(f"   🏆 Recommended window size: {best_window}s")
            best_result = results[best_window]
            
            if best_window <= 0.25:
                print(f"   📝 Reason: Better for detecting SHORT, quick profanities")
                print(f"   📝 Trade-off: More processing overhead, potential over-segmentation")
            elif best_window >= 0.5:
                print(f"   📝 Reason: Better for CONTEXTUAL understanding and efficiency")
                print(f"   📝 Trade-off: Might miss very short profanities")
            
            print(f"   📊 Expected performance: {best_result['merged_detections']} detections, {best_result['avg_confidence']:.3f} avg confidence")
        
        print(f"\n💭 GENERAL GUIDELINES:")
        print(f"   • Use 0.25s windows for: Short words, rapid speech, high precision needs")
        print(f"   • Use 0.5s windows for: Normal speech, efficiency, contextual understanding")
        print(f"   • Use 1.0s windows for: Long phrases, computational efficiency")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in window size comparison: {e}")
        import traceback
        traceback.print_exc()
        return {}

def main():
    """Main function with interactive menu for testing."""
    print("=" * 70)
    print("🎵 PRODUCTION AUDIO CENSORING TEST")
    print("🔧 Full Preprocessing Pipeline for Maximum Accuracy")
    print("=" * 70)
    
    while True:
        print("\n📋 Choose your test:")
        print("1. Single file test (with custom file path)")
        print("2. Multiple file test (scan eval directory)")
        print("3. Quick test (use predefined test file)")
        print("4. 🔬 Compare window sizes (0.5s vs 0.25s) - ADVANCED")
        print("5. Exit")
        
        choice = input("\n👉 Enter choice (1-5): ").strip()
        
        if choice == "1":
            # Custom file path
            custom_file = input("📁 Enter audio file path: ").strip().strip('"')
            if os.path.exists(custom_file):
                print(f"🎵 Testing custom file: {custom_file}")
                try:
                    # Try to import from scripts directory first
                    sys.path.append('./scripts')
                    try:
                        from simplified_ultimate_training import EnhancedAudioClassifier
                    except ImportError:
                        # Fallback: try current directory
                        sys.path.append('.')
                        from simplified_ultimate_training import EnhancedAudioClassifier
                    from transformers import Wav2Vec2FeatureExtractor
                    
                    result = test_single_file_production(custom_file, './custom_censored_output.wav')
                    if result:
                        if result.get('file_saved', False):
                            print(f"✅ File processing complete!")
                            print(f"📊 Found {result.get('detections', 0)} merged detections ({result.get('raw_detections', 0)} raw)")
                            print(f"💾 Censored file saved as: {result['output_file']}")
                            
                            # Verify file exists and show details
                            if os.path.exists(result['output_file']):
                                file_size = os.path.getsize(result['output_file'])
                                print(f"📁 File size: {file_size:,} bytes")
                                print(f"📍 Full path: {os.path.abspath(result['output_file'])}")
                            else:
                                print(f"⚠️ Warning: Output file not found at {result['output_file']}")
                        else:
                            print(f"❌ File processing failed - output not saved")
                        
                        if result.get('detections', 0) == 0:
                            print("ℹ️ No profanities detected in your file")
                    else:
                        print("❌ Processing failed - no result returned")
                except Exception as e:
                    print(f"❌ Error testing custom file: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"❌ File not found: {custom_file}")
        
        elif choice == "2":
            test_alternative_files()
        
        elif choice == "3":
            # Quick test with a known file
            test_production_audio_censoring()
        
        elif choice == "4":
            # Window size comparison
            print("\n🔬 WINDOW SIZE COMPARISON MODE")
            print("This will compare 0.5s vs 0.25s window sizes for detection accuracy")
            
            custom_file = input("📁 Enter audio file path for comparison: ").strip().strip('"')
            if os.path.exists(custom_file):
                # Allow custom confidence threshold
                conf_input = input("🎯 Enter confidence threshold (0.1-0.9, default 0.5): ").strip()
                try:
                    confidence_threshold = float(conf_input) if conf_input else 0.5
                    confidence_threshold = max(0.1, min(0.9, confidence_threshold))
                except ValueError:
                    confidence_threshold = 0.5
                
                print(f"🚀 Starting window size comparison with confidence threshold: {confidence_threshold}")
                compare_window_sizes(custom_file, window_sizes=[0.5, 0.25], confidence_threshold=confidence_threshold)
            else:
                print(f"❌ File not found: {custom_file}")
        
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")
    
    print("\n" + "=" * 70)
    print("🎉 PRODUCTION TEST COMPLETE")
    print("💡 Check the output files to hear the results!")
    print("📋 Review the JSON report for detailed detection info")
    print("=" * 70)

if __name__ == "__main__":
    main()

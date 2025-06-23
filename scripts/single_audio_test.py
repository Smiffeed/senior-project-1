import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import os
import argparse

class SingleAudioTester:
    def __init__(self, model_path="./output/model"):
        """Initialize the single audio tester"""
        self.model_path = model_path
        self.label_map = {
            0: 'none', 1: 'เย็ด', 2: 'กู', 3: 'มึง', 4: 'เหี้ย',
            5: 'ควย', 6: 'สวะ', 7: 'หี', 8: 'แตด'
        }
        
        # Load model and feature extractor
        print(f"Loading model from {model_path}...")
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model.eval()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"Model loaded on {self.device}")    def apply_spectral_gating(self, audio, sr, alpha=2.0, beta=0.15):
        """Apply spectral gating to reduce background noise - SAME AS TRAINING"""
        try:
            if len(audio) < 512:  # Too short for STFT
                return audio
                
            stft = torch.stft(torch.from_numpy(audio), n_fft=512, hop_length=256, return_complex=True)
            magnitude = torch.abs(stft)
            
            # Estimate noise floor
            noise_floor = torch.quantile(magnitude, 0.1, dim=-1, keepdim=True)
            
            # Create gate
            gate = torch.where(magnitude > alpha * noise_floor, 
                              torch.ones_like(magnitude), 
                              beta * torch.ones_like(magnitude))
            
            # Apply gate
            gated_stft = stft * gate
            
            # Reconstruct audio
            audio_gated = torch.istft(gated_stft, n_fft=512, hop_length=256)
            return audio_gated.numpy()
        except Exception as e:
            print(f"Spectral gating failed: {e}, using original audio")
            return audio

    def preprocess_audio(self, waveform, sample_rate, target_sr=16000):
        """Preprocess audio exactly like training"""
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
        
        # Convert to numpy
        audio = waveform.squeeze().numpy()
        
        # Ensure minimum length for Wav2Vec2 processing (at least 0.3 seconds)
        min_samples = int(0.3 * target_sr)
        if len(audio) < min_samples:
            # Repeat the audio or pad with zeros
            if len(audio) > 0:
                # Repeat the audio to reach minimum length
                repeats_needed = (min_samples // len(audio)) + 1
                audio = np.tile(audio, repeats_needed)[:min_samples]
            else:
                # Create silence if audio is empty
                audio = np.zeros(min_samples)
        
        # Apply spectral gating for noise reduction - SAME AS TRAINING
        audio = self.apply_spectral_gating(audio, target_sr)
        
        return audio

    def predict_single_audio(self, audio_file_path, window_size=0.5, overlap=0.25, confidence_threshold=0.4):
        """Predict profanity in a single audio file"""
        print(f"\n🎵 Testing Audio File: {audio_file_path}")
        print("=" * 80)
        
        if not os.path.exists(audio_file_path):
            print(f"❌ Error: Audio file '{audio_file_path}' not found!")
            return None
        
        # Load audio
        try:
            waveform, sample_rate = torchaudio.load(audio_file_path)
            total_duration = waveform.shape[1] / sample_rate
            print(f"📊 Audio Info:")
            print(f"   Duration: {total_duration:.2f} seconds")
            print(f"   Sample Rate: {sample_rate} Hz")
            print(f"   Channels: {waveform.shape[0]}")
            print(f"   Total Samples: {waveform.shape[1]:,}")
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return None
        
        print(f"\n🔍 Analysis Settings:")
        print(f"   Window Size: {window_size}s")
        print(f"   Overlap: {overlap}s")
        print(f"   Confidence Threshold: {confidence_threshold}")
        print(f"   Step Size: {window_size - overlap}s")
        
        # Calculate number of windows
        step_size = window_size - overlap
        num_windows = int((total_duration - overlap) / step_size) + 1
        print(f"   Expected Windows: ~{num_windows}")
        
        print(f"\n📝 Detailed Results:")
        print("-" * 80)
        print("Window | Time Range    | Prediction | Confidence | Status")
        print("-" * 80)
        
        results = []
        current_time = 0.0
        window_count = 0
        detections = 0
        
        while current_time < total_duration:
            start_time = current_time
            end_time = min(current_time + window_size, total_duration)
            
            # Extract segment
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment = waveform[:, start_sample:end_sample]
            
            # Skip if too short
            if segment.shape[1] < int(sample_rate * 0.1):
                break
            
            window_count += 1  # FIXED: was += 0.5, now += 1
            
            try:
                # Preprocess
                processed_audio = self.preprocess_audio(segment, sample_rate)
                
                # Feature extraction
                inputs = self.feature_extractor(
                    processed_audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
                
                input_values = inputs.input_values.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device) if 'attention_mask' in inputs else torch.ones_like(input_values)
                
                # Prediction
                with torch.no_grad():
                    outputs = self.model(input_values, attention_mask=attention_mask)
                    logits = outputs.logits
                    predictions = torch.nn.functional.softmax(logits, dim=-1)
                    predicted_class = torch.argmax(logits, dim=-1).item()
                    confidence = torch.max(predictions, dim=-1)[0].item()
                    all_probs = predictions[0].cpu().numpy()
                
                predicted_label = self.label_map[predicted_class]
                
                # Store result
                result = {
                    'window': window_count,
                    'start_time': start_time,
                    'end_time': end_time,
                    'predicted_class': predicted_class,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'all_probabilities': all_probs
                }
                results.append(result)
                
                # Determine status
                is_detection = predicted_label != 'none' and confidence > confidence_threshold
                if is_detection:
                    detections += 1
                    status_icon = "🚨 DETECT"
                else:
                    status_icon = "✅ CLEAN"
            
                # Print result
                print(f"{window_count:6d} | {start_time:5.2f}s-{end_time:5.2f}s | {predicted_label:10s} | "
                      f"{confidence:10.4f} | {status_icon}")
                
                # Show top 3 predictions if it's a detection
                if is_detection:
                    top_3_indices = np.argsort(all_probs)[-3:][::-1]
                    print(f"       | Top predictions:")
                    for i, idx in enumerate(top_3_indices):
                        print(f"       |   {i+1}. {self.label_map[idx]}: {all_probs[idx]:.4f}")
                
            except Exception as e:
                print(f"{window_count:6d} | {start_time:5.2f}s-{end_time:5.2f}s | ERROR      | {str(e)[:20]:10s} | ❌ ERROR")
        
            current_time += step_size
        
        # Summary
        print("-" * 80)
        print(f"\n📊 SUMMARY:")
        print(f"   Total Windows Processed: {window_count}")
        print(f"   Profanity Detections: {detections}")
        print(f"   Clean Windows: {window_count - detections}")
        print(f"   Detection Rate: {(detections/window_count)*100:.1f}%" if window_count > 0 else "   Detection Rate: N/A")
        
        if detections > 0:
            print(f"\n🚨 DETECTED PROFANITY:")
            detection_results = [r for r in results if r['predicted_label'] != 'none' and r['confidence'] > confidence_threshold]
            for det in detection_results:
                print(f"   {det['start_time']:.2f}s-{det['end_time']:.2f}s: {det['predicted_label']} (confidence: {det['confidence']:.4f})")
        else:
            print(f"\n✅ No profanity detected above threshold {confidence_threshold}")
        
        return results

    def quick_test(self, audio_file_path, confidence_threshold=0.4):
        """Quick test with 0.5s windows"""
        print(f"🚀 QUICK TEST (0.5s windows)")
        return self.predict_single_audio(
            audio_file_path, 
            window_size=0.5,  # Changed to 0.5 seconds
            overlap=0.25,     # 0.25s overlap (50% overlap)
            confidence_threshold=confidence_threshold
        )

    def sensitive_test(self, audio_file_path):
        """More sensitive test with smaller windows"""
        print(f"🔍 SENSITIVE TEST")
        return self.predict_single_audio(
            audio_file_path, 
            window_size=0.3,  # Even smaller windows for sensitivity
            overlap=0.15,     # 0.15s overlap
            confidence_threshold=0.3
        )

    def conservative_test(self, audio_file_path):
        """Conservative test with 0.5s windows but higher threshold"""
        print(f"🛡️ CONSERVATIVE TEST")
        return self.predict_single_audio(
            audio_file_path, 
            window_size=0.25,  # Still 0.5s windows
            overlap=0.125,     # 0.25s overlap
            confidence_threshold=0.6  # Higher threshold
        )

    def multi_threshold_test(self, audio_file_path):
        """Test with multiple thresholds to find optimal setting"""
        print(f"\n🎚️ MULTI-THRESHOLD ANALYSIS")
        print("=" * 80)
        
        # Get predictions with very low threshold
        base_results = self.predict_single_audio(audio_file_path, confidence_threshold=0.1)
        
        if not base_results:
            return None
        
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        print(f"\nThreshold Analysis:")
        print("-" * 50)
        print("Threshold | Detections | Time Segments")
        print("-" * 50)
        
        for threshold in thresholds:
            detections = [r for r in base_results if r['predicted_label'] != 'none' and r['confidence'] > threshold]
            
            time_segments = []
            for det in detections:
                time_segments.append(f"{det['start_time']:.1f}s-{det['end_time']:.1f}s")
            
            segments_str = ", ".join(time_segments[:3])  # Show first 3
            if len(time_segments) > 3:
                segments_str += f" (+{len(time_segments)-3} more)"
            
            print(f"   {threshold:.1f}    |     {len(detections):2d}     | {segments_str}")
        
        return base_results

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Test Thai Profanity Detection Model on Single Audio File')
    parser.add_argument('audio_file', help='Path to audio file to test')
    parser.add_argument('--model_path', default='./output/rework_v1', help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.4, help='Confidence threshold (default: 0.4)')
    parser.add_argument('--window_size', type=float, default=0.5, help='Window size in seconds (default: 0.5)')  # Changed default
    parser.add_argument('--overlap', type=float, default=0.25, help='Overlap in seconds (default: 0.25)')  # Changed default
    parser.add_argument('--test_type', choices=['quick', 'sensitive', 'conservative', 'multi'], 
                       default='quick', help='Type of test to run')
    
    args = parser.parse_args()
    
    # Initialize tester
    try:
        tester = SingleAudioTester(args.model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run selected test
    if args.test_type == 'quick':
        results = tester.quick_test(args.audio_file, args.threshold)
    elif args.test_type == 'sensitive':
        results = tester.sensitive_test(args.audio_file)
    elif args.test_type == 'conservative':
        results = tester.conservative_test(args.audio_file)
    elif args.test_type == 'multi':
        results = tester.multi_threshold_test(args.audio_file)
    else:
        # Custom test
        results = tester.predict_single_audio(
            args.audio_file, 
            window_size=args.window_size,
            overlap=args.overlap,
            confidence_threshold=args.threshold
        )
    
    print(f"\n✅ Testing completed!")

def simple_test(audio_file="test1.wav", model_path="./output/improved_model_fixed"):
    """Simple function for direct testing in Python with 0.5s windows"""
    tester = SingleAudioTester(model_path)
    return tester.quick_test(audio_file)

if __name__ == '__main__':
    # If no command line arguments, run simple test
    import sys
    if len(sys.argv) == 1:
        print("No arguments provided. Running simple test on 'test1.wav'")
        print("For full options, run: python single_audio_test.py --help")
        simple_test()
    else:
        main()
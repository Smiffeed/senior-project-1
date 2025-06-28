import torch
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm
import json
from datetime import datetime

# Import our models and utilities
from simplified_ultimate_training import EnhancedAudioClassifier, SimpleAudioPreprocessor
from transformers import Wav2Vec2FeatureExtractor

# Constants
LABEL_MAP = {
    'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4,
    'ควย': 5, 'สวะ': 6, 'หี': 7, 'แตด': 8
}
NUM_LABELS = len(LABEL_MAP)
CLASS_NAMES = list(LABEL_MAP.keys())

class AudioCensor:
    """Real-time audio censoring system based on profanity detection."""
    
    def __init__(self, model_dir, model_name="airesearch/wav2vec2-large-xlsr-53-th"):
        self.model_dir = model_dir
        self.model_name = model_name
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, return_attention_mask=True, do_normalize=True
        )
        self.preprocessor = SimpleAudioPreprocessor()
        self.model = None
        self.window_size = 0.5  # seconds
        self.hop_length = 0.25  # seconds
        self.confidence_threshold = 0.7  # minimum confidence for censoring
        
    def load_model(self, fold_num=1, stage='best'):
        """Load a trained model for censoring."""
        if stage == 'best':
            model_path = os.path.join(self.model_dir, f'fold_{fold_num}', 'best_model.pt')
        else:
            model_path = os.path.join(self.model_dir, f'fold_{fold_num}', stage, 'model.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create model using the simplified architecture
        self.model = EnhancedAudioClassifier(self.model_name, NUM_LABELS)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"Model loaded from {model_path}")
        
    def detect_profanities(self, audio, sr=16000):
        """Detect profanities in audio using sliding windows."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        audio_length = len(audio) / sr
        detections = []
        
        # Scan with sliding windows
        for window_start in np.arange(0, audio_length - self.window_size, self.hop_length):
            window_end = window_start + self.window_size
            
            # Extract audio segment
            start_sample = int(window_start * sr)
            end_sample = int(window_end * sr)
            audio_segment = audio[start_sample:end_sample]
            
            # Ensure correct length
            if len(audio_segment) < int(self.window_size * sr):
                audio_segment = np.pad(audio_segment, 
                                     (0, int(self.window_size * sr) - len(audio_segment)), 
                                     'constant')
            
            # Preprocess - skip for now since method expects file path
            # audio_segment = self.preprocessor.preprocess_audio_simple(audio_segment)
            
            # Get model prediction
            inputs = self.feature_extractor(
                audio_segment, 
                sampling_rate=sr, 
                return_tensors="pt", 
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs['logits'], dim=-1)
                prediction = torch.argmax(outputs['logits'], dim=-1).item()
                confidence = probs.max().item()
            
            # Store detection if profanity detected with sufficient confidence
            if prediction != 0 and confidence >= self.confidence_threshold:
                detection = {
                    'start_time': window_start,
                    'end_time': window_end,
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'predicted_class': CLASS_NAMES[prediction],
                    'predicted_id': prediction,
                    'confidence': confidence,
                    'all_probabilities': probs.squeeze().cpu().numpy()
                }
                detections.append(detection)
        
        return detections
    
    def merge_overlapping_detections(self, detections, merge_threshold=0.1):
        """Merge overlapping or nearby detections to avoid fragmented censoring."""
        if not detections:
            return []
        
        # Sort by start time
        detections = sorted(detections, key=lambda x: x['start_time'])
        merged = [detections[0]]
        
        for current in detections[1:]:
            last_merged = merged[-1]
            
            # Check if current detection overlaps or is very close to the last merged one
            if current['start_time'] <= last_merged['end_time'] + merge_threshold:
                # Merge: extend the end time and take the highest confidence class
                if current['confidence'] > last_merged['confidence']:
                    last_merged['predicted_class'] = current['predicted_class']
                    last_merged['predicted_id'] = current['predicted_id']
                    last_merged['confidence'] = current['confidence']
                
                last_merged['end_time'] = max(last_merged['end_time'], current['end_time'])
                last_merged['end_sample'] = max(last_merged['end_sample'], current['end_sample'])
            else:
                # No overlap, add as new detection
                merged.append(current)
        
        return merged
    
    def censor_audio(self, audio, detections, sr=16000, censor_method='silence'):
        """Censor detected profanity segments in audio."""
        censored_audio = audio.copy()
        
        for detection in detections:
            start_sample = detection['start_sample']
            end_sample = detection['end_sample']
            
            # Ensure we don't go out of bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if censor_method == 'silence':
                # Replace with silence
                censored_audio[start_sample:end_sample] = 0
                
            elif censor_method == 'beep':
                # Replace with beep tone (1000 Hz)
                duration = (end_sample - start_sample) / sr
                t = np.linspace(0, duration, end_sample - start_sample)
                beep = 0.3 * np.sin(2 * np.pi * 1000 * t)  # 1000 Hz beep at 30% volume
                censored_audio[start_sample:end_sample] = beep
                
            elif censor_method == 'noise':
                # Replace with white noise at lower volume
                noise_length = end_sample - start_sample
                noise = 0.1 * np.random.normal(0, 1, noise_length)  # Low volume white noise
                censored_audio[start_sample:end_sample] = noise
                
            elif censor_method == 'reverse':
                # Replace with reversed audio segment
                segment = audio[start_sample:end_sample]
                censored_audio[start_sample:end_sample] = segment[::-1]
        
        return censored_audio
    
    def process_audio_file(self, input_file, output_file=None, censor_method='silence', 
                          save_report=True, merge_detections=True):
        """Process a single audio file for censoring."""
        print(f"Processing audio file: {input_file}")
        
        # Load audio file
        try:
            audio, sr = librosa.load(input_file, sr=16000)
            print(f"Loaded audio: {len(audio)/sr:.2f} seconds")
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None
        
        # Detect profanities
        print("Detecting profanities...")
        detections = self.detect_profanities(audio, sr)
        print(f"Found {len(detections)} potential profanity segments")
        
        # Merge overlapping detections if requested
        if merge_detections and detections:
            original_count = len(detections)
            detections = self.merge_overlapping_detections(detections)
            print(f"Merged to {len(detections)} segments (was {original_count})")
        
        # Censor audio
        if detections:
            print(f"Censoring audio using method: {censor_method}")
            censored_audio = self.censor_audio(audio, detections, sr, censor_method)
        else:
            print("No profanities detected, audio unchanged")
            censored_audio = audio
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_censored_{censor_method}.wav"
        
        # Save censored audio
        try:
            sf.write(output_file, censored_audio, sr)
            print(f"Censored audio saved to: {output_file}")
        except Exception as e:
            print(f"Error saving audio file: {e}")
            return None
        
        # Generate and save report
        report = {
            'input_file': input_file,
            'output_file': output_file,
            'processing_time': datetime.now().isoformat(),
            'original_duration': len(audio) / sr,
            'censor_method': censor_method,
            'confidence_threshold': self.confidence_threshold,
            'total_detections': len(detections),
            'total_censored_duration': sum(d['end_time'] - d['start_time'] for d in detections),
            'detections': detections
        }
        
        if save_report:
            report_file = os.path.splitext(output_file)[0] + '_report.json'
            try:
                with open(report_file, 'w', encoding='utf-8') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    json_detections = []
                    for d in detections:
                        json_d = d.copy()
                        if 'all_probabilities' in json_d:
                            json_d['all_probabilities'] = json_d['all_probabilities'].tolist()
                        json_detections.append(json_d)
                    
                    report['detections'] = json_detections
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"Processing report saved to: {report_file}")
            except Exception as e:
                print(f"Error saving report: {e}")
        
        return report
    
    def process_directory(self, input_dir, output_dir=None, censor_method='silence', 
                         audio_extensions=None):
        """Process all audio files in a directory."""
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        
        # Find all audio files
        audio_files = []
        for ext in audio_extensions:
            pattern = os.path.join(input_dir, f"**/*{ext}")
            audio_files.extend([f for f in os.listdir(input_dir) 
                              if f.lower().endswith(ext.lower())])
        
        audio_files = [os.path.join(input_dir, f) for f in audio_files]
        print(f"Found {len(audio_files)} audio files to process")
        
        if not audio_files:
            print("No audio files found!")
            return []
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(input_dir, 'censored')
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each file
        results = []
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            try:
                filename = os.path.basename(audio_file)
                name, ext = os.path.splitext(filename)
                output_file = os.path.join(output_dir, f"{name}_censored_{censor_method}.wav")
                
                result = self.process_audio_file(
                    audio_file, output_file, censor_method, 
                    save_report=True, merge_detections=True
                )
                if result:
                    results.append(result)
                    
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        # Generate summary report
        summary_file = os.path.join(output_dir, 'censoring_summary.json')
        summary = {
            'processing_time': datetime.now().isoformat(),
            'input_directory': input_dir,
            'output_directory': output_dir,
            'censor_method': censor_method,
            'total_files_processed': len(results),
            'total_files_found': len(audio_files),
            'total_detections': sum(r['total_detections'] for r in results),
            'total_censored_duration': sum(r['total_censored_duration'] for r in results),
            'files': results
        }
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"Summary report saved to: {summary_file}")
        except Exception as e:
            print(f"Error saving summary: {e}")
        
        return results

def main():
    """Main function for audio censoring."""
    # Configuration
    MODEL_DIR = './models/simplified_advanced_audio_train'
    
    # Initialize censoring system
    censor = AudioCensor(MODEL_DIR)
    
    try:
        # Load the trained model
        censor.load_model(fold_num=1, stage='best')
        
        # Example: Process a single file
        # Uncomment and modify the path below to test with your audio file
        # input_file = './eval/example_audio.wav'
        # result = censor.process_audio_file(input_file, censor_method='silence')
        
        # Example: Process all files in the eval directory
        input_directory = './eval'
        if os.path.exists(input_directory):
            print(f"Processing all audio files in: {input_directory}")
            
            # Test different censoring methods
            for method in ['silence', 'beep', 'noise']:
                print(f"\n--- Processing with {method} censoring ---")
                results = censor.process_directory(
                    input_directory, 
                    output_dir=f'./eval/censored_{method}',
                    censor_method=method
                )
                
                if results:
                    total_detections = sum(r['total_detections'] for r in results)
                    total_duration = sum(r['total_censored_duration'] for r in results)
                    print(f"Completed {method} censoring:")
                    print(f"  Files processed: {len(results)}")
                    print(f"  Total detections: {total_detections}")
                    print(f"  Total censored duration: {total_duration:.2f} seconds")
        else:
            print(f"Directory not found: {input_directory}")
            print("Please specify a valid input directory or audio file")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nUsage examples:")
        print("1. Process single file:")
        print("   result = censor.process_audio_file('input.wav', 'output_censored.wav')")
        print("2. Process directory:")
        print("   results = censor.process_directory('./input_dir', './output_dir')")

if __name__ == "__main__":
    main()

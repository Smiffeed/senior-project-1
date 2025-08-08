import torch
import numpy as np
import librosa
import soundfile as sf
import os
import argparse
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

class SimpleCensor:
    """Simple audio censoring for single files."""
    
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model_name = "airesearch/wav2vec2-large-xlsr-53-th"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name, return_attention_mask=True, do_normalize=True
        )
        self.preprocessor = SimpleAudioPreprocessor()
        self.model = self._load_model()
        
    def _load_model(self, fold_num=1):
        """Load the trained model."""
        model_path = os.path.join(self.model_dir, f'fold_{fold_num}', 'best_model.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = EnhancedAudioClassifier(self.model_name, NUM_LABELS)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"Model loaded from {model_path}")
        return model
        
    def detect_and_censor(self, input_file, output_file=None, method='silence', threshold=0.7):
        """Detect profanities and censor them in one go."""
        print(f"Processing: {input_file}")
        
        # Load audio
        audio, sr = librosa.load(input_file, sr=16000)
        print(f"Audio duration: {len(audio)/sr:.2f} seconds")
        
        # Sliding window parameters
        window_size = 0.5  # seconds
        hop_length = 0.25  # seconds
        
        audio_length = len(audio) / sr
        censored_audio = audio.copy()
        detections = []
        
        # Process with sliding windows
        window_starts = np.arange(0, audio_length - window_size, hop_length)
        
        for window_start in window_starts:
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
            
            # Preprocess and predict
            audio_segment = self.preprocessor.preprocess_audio_simple(audio_segment)
            inputs = self.feature_extractor(
                audio_segment, sampling_rate=sr, return_tensors="pt", padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs['logits'], dim=-1)
                prediction = torch.argmax(outputs['logits'], dim=-1).item()
                confidence = probs.max().item()
            
            # Censor if profanity detected with sufficient confidence
            if prediction != 0 and confidence >= threshold:
                # Record detection
                detections.append({
                    'time': f"{window_start:.2f}s - {window_end:.2f}s",
                    'class': CLASS_NAMES[prediction],
                    'confidence': f"{confidence:.3f}"
                })
                
                # Apply censoring
                if method == 'silence':
                    censored_audio[start_sample:end_sample] = 0
                elif method == 'beep':
                    duration = (end_sample - start_sample) / sr
                    t = np.linspace(0, duration, end_sample - start_sample)
                    beep = 0.3 * np.sin(2 * np.pi * 1000 * t)
                    censored_audio[start_sample:end_sample] = beep
                elif method == 'noise':
                    noise_length = end_sample - start_sample
                    noise = 0.1 * np.random.normal(0, 1, noise_length)
                    censored_audio[start_sample:end_sample] = noise
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_censored.wav"
        
        # Save censored audio
        sf.write(output_file, censored_audio, sr)
        
        # Print results
        print(f"\nCensoring complete!")
        print(f"Detections found: {len(detections)}")
        if detections:
            print("Detected profanities:")
            for detection in detections:
                print(f"  {detection['time']}: {detection['class']} (confidence: {detection['confidence']})")
        print(f"Censored audio saved to: {output_file}")
        
        return output_file, detections

def main():
    """Command line interface for audio censoring."""
    parser = argparse.ArgumentParser(description='Censor profanities in audio files')
    parser.add_argument('input_file', help='Input audio file path')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('-m', '--method', choices=['silence', 'beep', 'noise'], 
                       default='silence', help='Censoring method (default: silence)')
    parser.add_argument('-t', '--threshold', type=float, default=0.7,
                       help='Confidence threshold for detection (default: 0.7)')
    parser.add_argument('--model-dir', default='./models/simplified_advanced_audio_train',
                       help='Model directory path')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    try:
        # Initialize censoring system
        censor = SimpleCensor(args.model_dir)
        
        # Process the file
        output_file, detections = censor.detect_and_censor(
            args.input_file, args.output, args.method, args.threshold
        )
        
        print(f"\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model directory contains trained models.")

if __name__ == "__main__":
    # If run without command line arguments, use interactive mode
    if len(os.sys.argv) == 1:
        print("Audio Censoring Tool")
        print("=" * 50)
        
        # Interactive mode
        model_dir = input("Model directory [./models/simplified_advanced_audio_train]: ").strip()
        if not model_dir:
            model_dir = "./models/simplified_advanced_audio_train"
        
        input_file = input("Input audio file path: ").strip()
        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            exit(1)
        
        method = input("Censoring method (silence/beep/noise) [silence]: ").strip().lower()
        if method not in ['silence', 'beep', 'noise']:
            method = 'silence'
        
        threshold = input("Confidence threshold [0.7]: ").strip()
        try:
            threshold = float(threshold) if threshold else 0.7
        except ValueError:
            threshold = 0.7
        
        try:
            censor = SimpleCensor(model_dir)
            output_file, detections = censor.detect_and_censor(input_file, method=method, threshold=threshold)
        except Exception as e:
            print(f"Error: {e}")
    else:
        main()

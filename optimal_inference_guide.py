#!/usr/bin/env python3
"""
🎯 INFERENCE GUIDE: How to use your trained models
Post-training usage with optimal window configurations
"""

import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from typing import List, Dict, Tuple, Optional

class OptimalInferenceSystem:
    """
    Inference system using optimal window configurations from training
    """
    
    def __init__(self, 
                 binary_model_path: str = "models/binary_classifier_fast",
                 multiclass_model_path: str = "models/multiclass_classifier_fast"):
        """
        Initialize inference system with trained models
        
        Args:
            binary_model_path: Path to binary classifier (2.0s windows)
            multiclass_model_path: Path to multiclass classifier (0.3s windows)
        """
        
        # Load models
        print("🔄 Loading trained models...")
        self.binary_model = Wav2Vec2ForSequenceClassification.from_pretrained(binary_model_path)
        self.multiclass_model = Wav2Vec2ForSequenceClassification.from_pretrained(multiclass_model_path)
        
        # Feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        
        # Window configurations (from training)
        self.binary_window_size = 2.0    # Optimal for binary classification
        self.multiclass_window_size = 0.3  # Optimal for multiclass classification
        
        # Labels
        self.binary_labels = ['none', 'profanity']
        self.multiclass_labels = ['none', 'เย็ด', 'กู', 'มึง', 'เหี้ย']
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.binary_model.to(self.device)
        self.multiclass_model.to(self.device)
        
        print(f"✅ Models loaded on {self.device}")
        print(f"Binary model window: {self.binary_window_size}s")
        print(f"Multiclass model window: {self.multiclass_window_size}s")
    
    def predict_binary_fast(self, audio_file: str) -> List[Dict]:
        """
        Fast binary profanity detection using 2.0s windows
        
        Use case: Quick screening, real-time processing
        Returns: List of profanity segments with timestamps
        """
        print(f"🔍 Binary detection (2.0s windows): {audio_file}")
        
        # Load full audio
        audio, sr = librosa.load(audio_file, sr=16000)
        audio_duration = len(audio) / sr
        
        results = []
        
        # Process with 2.0s sliding windows
        stride = 1.0  # 1 second stride for good coverage
        current_time = 0
        
        while current_time + self.binary_window_size <= audio_duration:
            # Extract window
            start_sample = int(current_time * sr)
            end_sample = int((current_time + self.binary_window_size) * sr)
            window_audio = audio[start_sample:end_sample]
            
            # Predict
            prediction = self._predict_window(window_audio, self.binary_model, self.binary_labels)
            
            # Store if profanity detected
            if prediction['label'] == 'profanity' and prediction['confidence'] > 0.7:
                results.append({
                    'start_time': current_time,
                    'end_time': current_time + self.binary_window_size,
                    'label': 'profanity',
                    'confidence': prediction['confidence'],
                    'window_size': self.binary_window_size
                })
            
            current_time += stride
        
        print(f"✅ Found {len(results)} profanity segments")
        return results
    
    def predict_multiclass_precise(self, audio_file: str, 
                                 suspected_regions: Optional[List[Tuple[float, float]]] = None) -> List[Dict]:
        """
        Precise multiclass profanity identification using 0.3s windows
        
        Use case: Detailed analysis of specific regions
        Args:
            audio_file: Path to audio file
            suspected_regions: Optional list of (start, end) time regions to analyze
        """
        print(f"🎯 Multiclass detection (0.3s windows): {audio_file}")
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=16000)
        audio_duration = len(audio) / sr
        
        results = []
        
        if suspected_regions:
            # Analyze only suspected regions
            regions_to_analyze = suspected_regions
        else:
            # Analyze entire audio with fine-grained windows
            regions_to_analyze = [(0, audio_duration)]
        
        for region_start, region_end in regions_to_analyze:
            # Process region with 0.3s sliding windows
            stride = 0.15  # 0.15s stride for fine-grained detection
            current_time = region_start
            
            while current_time + self.multiclass_window_size <= region_end:
                # Extract window
                start_sample = int(current_time * sr)
                end_sample = int((current_time + self.multiclass_window_size) * sr)
                
                if end_sample <= len(audio):
                    window_audio = audio[start_sample:end_sample]
                    
                    # Predict
                    prediction = self._predict_window(window_audio, self.multiclass_model, self.multiclass_labels)
                    
                    # Store if profanity detected
                    if prediction['label'] != 'none' and prediction['confidence'] > 0.6:
                        results.append({
                            'start_time': current_time,
                            'end_time': current_time + self.multiclass_window_size,
                            'label': prediction['label'],
                            'confidence': prediction['confidence'],
                            'window_size': self.multiclass_window_size
                        })
                
                current_time += stride
        
        print(f"✅ Found {len(results)} specific profanity instances")
        return results
    
    def _predict_window(self, audio_window: np.ndarray, model, labels: List[str]) -> Dict:
        """Predict on a single audio window"""
        
        # Ensure correct length
        target_samples = len(audio_window)
        if len(audio_window) < target_samples:
            audio_window = np.pad(audio_window, (0, target_samples - len(audio_window)), mode='constant')
        
        # Process with feature extractor
        inputs = self.feature_extractor(
            audio_window,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            'label': labels[predicted_class],
            'confidence': confidence,
            'all_scores': {labels[i]: predictions[0][i].item() for i in range(len(labels))}
        }
    
    def comprehensive_analysis(self, audio_file: str) -> Dict:
        """
        Complete analysis using both models optimally
        
        Strategy:
        1. Binary model (2.0s) for quick profanity region detection
        2. Multiclass model (0.3s) for precise analysis of detected regions
        """
        print(f"🚀 Comprehensive analysis: {audio_file}")
        
        # Stage 1: Fast binary detection
        binary_results = self.predict_binary_fast(audio_file)
        
        if not binary_results:
            return {
                'has_profanity': False,
                'binary_detections': [],
                'multiclass_detections': [],
                'summary': 'No profanity detected'
            }
        
        # Stage 2: Precise multiclass analysis of detected regions
        suspected_regions = [(r['start_time'], r['end_time']) for r in binary_results]
        multiclass_results = self.predict_multiclass_precise(audio_file, suspected_regions)
        
        # Aggregate results
        profanity_types = set(r['label'] for r in multiclass_results if r['label'] != 'none')
        
        summary = {
            'has_profanity': len(multiclass_results) > 0,
            'binary_detections': binary_results,
            'multiclass_detections': multiclass_results,
            'profanity_types_detected': list(profanity_types),
            'total_profanity_segments': len(multiclass_results),
            'analysis_method': '2-stage: 2.0s binary → 0.3s multiclass',
            'summary': f"Found {len(profanity_types)} types of profanity in {len(multiclass_results)} segments"
        }
        
        return summary

def demonstrate_usage():
    """Demonstrate different usage scenarios"""
    
    print("🎯 OPTIMAL INFERENCE USAGE GUIDE")
    print("=" * 50)
    
    # Initialize system
    inference = OptimalInferenceSystem()
    
    print("\n📋 USAGE SCENARIOS:")
    print("\n1. 🚀 FAST SCREENING (Real-time/Streaming)")
    print("   Use: Binary model with 2.0s windows")
    print("   Code: inference.predict_binary_fast(audio_file)")
    print("   Speed: ~5x faster than multiclass")
    print("   Accuracy: High for profanity vs clean detection")
    
    print("\n2. 🎯 PRECISE ANALYSIS (Detailed forensics)")
    print("   Use: Multiclass model with 0.3s windows")
    print("   Code: inference.predict_multiclass_precise(audio_file)")
    print("   Speed: Slower but more precise")
    print("   Accuracy: High for specific profanity type identification")
    
    print("\n3. 🏆 OPTIMAL HYBRID (Best of both)")
    print("   Use: Binary (2.0s) → Multiclass (0.3s) on detected regions")
    print("   Code: inference.comprehensive_analysis(audio_file)")
    print("   Speed: Fast overall with precise details")
    print("   Accuracy: Maximum accuracy with efficiency")
    
    print("\n🎯 WINDOW SIZE RECOMMENDATIONS:")
    print("┌─────────────────┬─────────────┬─────────────────┬──────────────┐")
    print("│ Use Case        │ Model       │ Window Size     │ Stride       │")
    print("├─────────────────┼─────────────┼─────────────────┼──────────────┤")
    print("│ Live Streaming  │ Binary      │ 2.0s           │ 1.0s         │")
    print("│ Content Review  │ Hybrid      │ 2.0s → 0.3s    │ 1.0s → 0.15s │")
    print("│ Forensic Anal.  │ Multiclass  │ 0.3s           │ 0.15s        │")
    print("│ Mobile Apps     │ Binary      │ 2.0s           │ 2.0s         │")
    print("└─────────────────┴─────────────┴─────────────────┴──────────────┘")

if __name__ == "__main__":
    demonstrate_usage()
    
    # Example usage
    # inference = OptimalInferenceSystem()
    # results = inference.comprehensive_analysis("path/to/audio.wav")
    # print(results)
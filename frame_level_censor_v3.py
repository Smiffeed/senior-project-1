#!/usr/bin/env python3
"""
🎯 FRAME-LEVEL PROFANITY CENSORING
Advanced censoring system based on comprehensive evaluation methods

Features:
- Word-level detection using sliding windows
- Ensemble model predictions for better accuracy
- Energy-based VAD for precise boundary detection
- Multiple censoring methods (silence, beep, fade)
- Comprehensive reporting and analysis
"""

import torch
import torchaudio
import numpy as np
import pandas as pd
import librosa
import json
import argparse
import os
import glob
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Constants from evaluation script
LABEL_MAP = {
    'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4
}
NUM_LABELS = len(LABEL_MAP)
CLASS_NAMES = list(LABEL_MAP.keys())

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleAudioPreprocessor:
    """Audio preprocessor aligned with the training pipeline.

    Steps per window (target_length in samples):
    - Pre-emphasis (coef=0.97)
    - Gentle energy-based noise gate (20th percentile x 2.5)
    - Light Hamming edge windowing
    - Mild dynamic range compression (soft clipping around 2*rms)
    - Z-score normalization then scale by 0.5
    - Pad/truncate to exact target_length
    """

    def __init__(self):
        self.sample_rate = 16000

    def preprocess(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        # Ensure 1D float32
        if audio is None:
            audio = np.zeros(target_length, dtype=np.float32)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        # Pre-emphasis
        try:
            audio = librosa.effects.preemphasis(audio, coef=0.97)
        except Exception:
            pass

        # Gentle noise gate
        if audio.size:
            energy = np.abs(audio)
            noise_threshold = np.percentile(energy, 20)
            gate_th = float(noise_threshold) * 2.5
            mask = energy < gate_th
            audio = np.where(mask, audio * 0.1, audio)

        # Edge windowing
        L = len(audio)
        if L > 160:
            w = np.hamming(L).astype(np.float32)
            w = 0.85 + 0.15 * w
            audio = audio * w

        # Mild dynamic range compression
        rms = float(np.sqrt(np.mean(audio**2)) + 1e-8)
        audio = np.tanh(audio / (2.0 * rms)) * (2.0 * rms)

        # Z-score normalization then scale
        mean = float(np.mean(audio))
        std = float(np.std(audio))
        if std > 1e-8:
            audio = (audio - mean) / std
        else:
            audio = audio - mean
        audio = audio * 0.5

        # Pad/truncate to target length
        if len(audio) < target_length:
            pad = target_length - len(audio)
            audio = np.pad(audio, (0, pad), mode='constant')
        elif len(audio) > target_length:
            audio = audio[:target_length]

        return audio.astype(np.float32, copy=False)

class SimpleProfanityDataset:
    """Simple dataset class matching evaluation script"""
    
    def __init__(self, audio_data, feature_extractor, preprocessor, mode='eval', window_size=0.5):
        self.audio_data = audio_data
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
        self.mode = mode
        self.window_size = window_size  # Store window_size for consistent processing
    
    def process_audio_window(self, audio_window, label=0):
        """Process a single audio window"""
        try:
            # Preprocess audio to match training pipeline
            target_length = int(self.window_size * 16000)  # Fixed samples for this window size
            processed_audio = self.preprocessor.preprocess(audio_window, target_length)

            # Extract features; audio already fixed-length
            inputs = self.feature_extractor(
                processed_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=False,
                truncation=False
            )
            
            return {
                'input_values': inputs.input_values.squeeze(),
                'attention_mask': inputs.attention_mask.squeeze() if hasattr(inputs, 'attention_mask') else torch.ones_like(inputs.input_values.squeeze()),
                'label': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error processing audio window: {e}")
            return None

class EnhancedAudioClassifier(torch.nn.Module):
    """Enhanced audio classifier matching evaluation script"""
    
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.wav2vec2 = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.num_labels = num_labels
    
    def forward(self, input_values, attention_mask=None, **kwargs):
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask
        )
        
        return {
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }

class ModelEnsemble:
    """Model ensemble for improved predictions"""
    
    def __init__(self, models):
        self.models = models
        self.num_models = len(models)
    
    def predict(self, inputs):
        """Make ensemble prediction"""
        all_logits = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                all_logits.append(outputs['logits'])
        
        # Average predictions
        ensemble_logits = torch.stack(all_logits).mean(dim=0)
        
        # Calculate uncertainty (standard deviation)
        logits_std = torch.stack(all_logits).std(dim=0)
        uncertainty = logits_std.mean()
        
        return {
            'predictions': ensemble_logits,
            'uncertainty': uncertainty,
            'individual_logits': all_logits
        }

class AdvancedFrameLevelCensor:
    """
    Advanced frame-level censoring system based on comprehensive evaluation methods
    Uses word-level detection with ensemble models for best accuracy
    """

    def __init__(self, model_dir: str, window_size: float = 0.5, overlap: float = 0.25, 
                 confidence_threshold: float = 0.5, model_name: str = "airesearch/wav2vec2-large-xlsr-53-th"):
        """
        Initialize the advanced censoring system
        
        Args:
            model_dir: Directory containing trained models
            window_size: Window size in seconds (0.5s balanced for both word/binary detection)
            overlap: Overlap in seconds (0.25s gives 0.25s stride = 50% overlap)
            confidence_threshold: Minimum confidence for detection
            model_name: Base model name
        """
        self.model_dir = model_dir
        self.window_size = window_size
        self.overlap = overlap
        # 0.5s window - 0.25s overlap = 0.25s stride (50% overlap)
        self.hop_length = window_size - overlap
        self.sample_rate = 16000
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        
        # Initialize class information (will be updated when loading models)
        self.num_labels = NUM_LABELS
        self.class_names = CLASS_NAMES
        
        # Initialize components
        self.feature_extractor = self._load_feature_extractor()
        self.preprocessor = SimpleAudioPreprocessor()
        
        # Load ensemble models
        self.models = self._load_ensemble_models()
        if self.models:
            self.ensemble = ModelEnsemble(self.models) if len(self.models) > 1 else None
            print(f"✅ Loaded {'ensemble of ' if self.ensemble else ''}{len(self.models)} model{'s' if len(self.models) > 1 else ''}")
            print(f"🎯 Configuration: window={window_size}s, hop={self.hop_length}s, overlap={overlap}s, confidence≥{confidence_threshold}")
        else:
            print("❌ No models loaded - trying single model fallback")
            self.ensemble = None
    
    def _load_feature_extractor(self):
        """Load feature extractor from model directory or base model"""
        # Check if preprocessor_config.json exists in model directory
        preprocessor_path = os.path.join(self.model_dir, 'preprocessor_config.json')
        
        if os.path.exists(preprocessor_path):
            print(f"🎯 Loading feature extractor from model directory: {self.model_dir}")
            try:
                return Wav2Vec2FeatureExtractor.from_pretrained(
                    self.model_dir, 
                    return_attention_mask=True, 
                    do_normalize=True,
                    local_files_only=True
                )
            except Exception as e:
                print(f"⚠️  Failed to load feature extractor from model dir: {e}")
        
        # Fallback to base model
        print(f"🎯 Loading feature extractor from base model: {self.model_name}")
        return Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name, 
            return_attention_mask=True, 
            do_normalize=True
        )
    
    def _load_ensemble_models(self, num_folds=5):
        """Load ensemble of models from different folds or single HuggingFace model"""
        models = []

        # Resolve model directory; fallback to ./models/<dir> if needed
        model_path = Path(self.model_dir)
        if not model_path.exists():
            alt_path = Path('models') / self.model_dir
            if alt_path.exists():
                print(f"🔎 Model dir not found, using: {alt_path}")
                model_path = alt_path
                self.model_dir = str(alt_path)

        # Check for a local HuggingFace model directory (allow either safetensors or pytorch bin)
        has_config = (model_path / 'config.json').exists()
        has_preproc = (model_path / 'preprocessor_config.json').exists()
        has_weights = (model_path / 'model.safetensors').exists() or (model_path / 'pytorch_model.bin').exists()
        has_hf_files = model_path.exists() and has_config and has_weights
        
        if has_hf_files:
            print(f"🎯 Detected HuggingFace model format in {self.model_dir}")
            try:
                # Load HuggingFace model directly
                model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    self.model_dir,
                    local_files_only=True
                ).to(device)
                model.eval()
                models.append(model)
                
                # Update class information from the loaded model
                config = model.config
                self.num_labels = config.num_labels if hasattr(config, 'num_labels') else len(config.id2label)
                
                # Create label mapping - use model's labels if available, otherwise use our default
                if hasattr(config, 'id2label') and config.id2label:
                    # Check if model uses meaningful labels or generic LABEL_X
                    model_labels = list(config.id2label.values())
                    if any('LABEL_' in str(label) for label in model_labels):
                        # Generic labels, use our mapping but adjust for model's class count
                        print(f"🏷️  Using default Thai profanity labels (model has {self.num_labels} classes)")
                        self.class_names = CLASS_NAMES[:self.num_labels]
                    else:
                        # Model has meaningful labels
                        print(f"🏷️  Using model's label mapping: {model_labels}")
                        self.class_names = model_labels
                else:
                    # No label mapping, use our default
                    print(f"🏷️  No label mapping found, using default Thai profanity labels")
                    self.class_names = CLASS_NAMES[:self.num_labels]
                
                print(f"✅ Loaded HuggingFace model with {self.num_labels} classes: {self.class_names}")
                return models
                
            except Exception as e:
                print(f"❌ Failed to load HuggingFace model: {e}")
                # Continue to try other loading methods
        
        # Try to load models from different folds (original ensemble approach)
        for fold in range(1, num_folds + 1):
            fold_model_path = os.path.join(self.model_dir, f'fold_{fold}', 'best_model.pt')
            
            if os.path.exists(fold_model_path):
                try:
                    model = EnhancedAudioClassifier(self.model_name, self.num_labels)
                    checkpoint = torch.load(fold_model_path, map_location=device)
                    
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model.to(device)
                    model.eval()
                    models.append(model)
                    print(f"✅ Loaded model from fold {fold}")
                    
                except Exception as e:
                    print(f"⚠️  Failed to load model from fold {fold}: {e}")
            else:
                print(f"⚠️  Model not found: {fold_model_path}")
        
        # If no fold models found, try to load from other common PyTorch paths
        if not models:
            common_paths = [
                'best_model.pt',
                'model.pt',
                'pytorch_model.bin'
            ]
            
            for path in common_paths:
                full_path = os.path.join(self.model_dir, path)
                if os.path.exists(full_path):
                    try:
                        model = EnhancedAudioClassifier(self.model_name, self.num_labels)
                        checkpoint = torch.load(full_path, map_location=device)
                        
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            model.load_state_dict(checkpoint)
                        
                        model.to(device)
                        model.eval()
                        models.append(model)
                        print(f"✅ Loaded single model from {path}")
                        break
                        
                    except Exception as e:
                        print(f"⚠️  Failed to load model from {path}: {e}")

        # Final fallback: try loading a HuggingFace model by ID (requires internet or cached)
        if not models:
            try:
                print(f"🌐 Attempting to load HuggingFace model by ID: {self.model_dir}")
                model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_dir).to(device)
                model.eval()
                models.append(model)
                # Update labels if available
                config = model.config
                self.num_labels = getattr(config, 'num_labels', self.num_labels)
                if hasattr(config, 'id2label') and config.id2label:
                    self.class_names = list(config.id2label.values())
                print(f"✅ Loaded HuggingFace model by ID: {self.model_dir}")
            except Exception as e:
                print(f"❌ Failed to load model by HuggingFace ID: {e}")
        
        return models
    
    def _detect_profanity_windows(self, audio_np: np.ndarray) -> List[Tuple[float, float, str, float]]:
        """
        Detect profanity using sliding window approach (word-level method from evaluation)
        
        Returns:
            List of (start_time, end_time, label, confidence) tuples
        """
        window_samples = int(self.window_size * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)
        
        detected_windows = []
        
        print("🔍 Scanning audio with word-level detection...")
        
        # Create dataset processor
        dataset_processor = SimpleProfanityDataset(None, self.feature_extractor, self.preprocessor, window_size=self.window_size)
        
        for i in tqdm(range(0, len(audio_np) - window_samples, hop_samples), desc="Processing windows"):
            start_sample = i
            end_sample = i + window_samples
            window = audio_np[start_sample:end_sample]
            
            # Process window
            processed_sample = dataset_processor.process_audio_window(window)
            if processed_sample is None:
                continue
            
            # Prepare inputs
            inputs = {
                'input_values': processed_sample['input_values'].unsqueeze(0).to(device),
                'attention_mask': processed_sample['attention_mask'].unsqueeze(0).to(device)
            }
            
            # Get prediction
            if self.ensemble:
                # Use ensemble prediction
                ensemble_output = self.ensemble.predict(inputs)
                logits = ensemble_output['predictions']
                uncertainty = ensemble_output['uncertainty'].item()
            else:
                # Use single model if available
                if self.models:
                    with torch.no_grad():
                        outputs = self.models[0](**inputs)
                        logits = outputs['logits']
                        uncertainty = 0.0
                else:
                    continue
            
            # Get prediction and confidence
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities.max().item()
            
            # Check if profanity detected with sufficient confidence
            if prediction != 0 and confidence >= self.confidence_threshold:  # Not 'none' class
                start_time = start_sample / self.sample_rate
                end_time = end_sample / self.sample_rate
                label = self.class_names[prediction] if prediction < len(self.class_names) else f"class_{prediction}"
                
                detected_windows.append((start_time, end_time, label, confidence))
        
        print(f"🎯 Found {len(detected_windows)} high-confidence profanity windows")
        return self._merge_overlapping_windows(detected_windows)
    
    def _merge_overlapping_windows(self, windows: List[Tuple[float, float, str, float]], max_gap: float = 0.3) -> List[Tuple[float, float, str, float]]:
        """Merge overlapping or nearby profanity windows"""
        if not windows:
            return []
        
        # Sort by start time
        windows.sort(key=lambda x: x[0])
        
        merged = []
        current_start, current_end, current_label, current_conf = windows[0]
        
        for next_start, next_end, next_label, next_conf in windows[1:]:
            # Merge if overlapping or gap is small
            if next_start <= current_end + max_gap:
                # Extend current window
                current_end = max(current_end, next_end)
                # Keep label with higher confidence
                if next_conf > current_conf:
                    current_label = next_label
                    current_conf = next_conf
            else:
                # Add current window and start new one
                merged.append((current_start, current_end, current_label, current_conf))
                current_start, current_end, current_label, current_conf = next_start, next_end, next_label, next_conf
        
        # Add final window
        merged.append((current_start, current_end, current_label, current_conf))
        
        print(f"📋 Merged into {len(merged)} distinct profanity regions")
        return merged
    
    def _refine_boundaries_with_vad(self, audio_np: np.ndarray, region: Tuple[float, float, str, float]) -> Optional[Tuple[float, float, str, float]]:
        """
        Refine boundaries using Voice Activity Detection
        Mirrors the evaluation script's approach:
        - librosa.effects.split with top_db=20
        - merge segments with <=50 ms gaps
        - add +/- 0.1 s padding within the original region bounds
        - choose the longest refined segment; enforce >= 0.1 s min duration
        """
        start_time, end_time, label, confidence = region
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        
        region_audio = audio_np[start_sample:end_sample]
        
        # Apply VAD to find speech segments
        try:
            # 1) basic split
            intervals = librosa.effects.split(region_audio, top_db=20)
            if len(intervals) == 0:
                return None

            # 2) merge close gaps (<= 50 ms)
            merged = []
            max_gap = int(0.05 * self.sample_rate)
            current = list(intervals[0])
            for s, e in intervals[1:]:
                if s - current[1] <= max_gap:
                    current[1] = e
                else:
                    merged.append(tuple(current))
                    current = [s, e]
            merged.append(tuple(current))

            # 3) pad +/- 0.1 s within original bounds
            pad = int(0.1 * self.sample_rate)
            padded = []
            for s, e in merged:
                ps = max(0, s - pad)
                pe = min(len(region_audio), e + pad)
                padded.append((ps, pe))

            # 4) choose the longest interval
            longest = max(padded, key=lambda x: x[1] - x[0])
            speech_start, speech_end = longest

            # Convert back to absolute time
            refined_start = start_time + (speech_start / self.sample_rate)
            refined_end = start_time + (speech_end / self.sample_rate)

            # Ensure minimum duration of 100 ms
            if refined_end - refined_start < 0.1:
                return None

            return (refined_start, refined_end, label, confidence)
            
        except Exception as e:
            print(f"⚠️  VAD refinement failed: {e}")
            return region
    
    def censor_audio_file(self, input_path: str, output_path: str, method: str = "silence", 
                         beep_freq: float = 1000.0, fade_duration: float = 0.05):
        """
        Censor profanity in audio file using advanced detection methods
        
        Args:
            input_path: Input audio file path
            output_path: Output censored audio file path
            method: Censoring method ('silence', 'beep', 'fade')
            beep_freq: Frequency for beep censoring
            fade_duration: Fade duration in seconds
        """
        print(f"🎵 Processing audio file: {input_path}")
        print(f"📊 Using method: {method}")
        
        # Load audio
        try:
            audio, sr = torchaudio.load(input_path)
            
            # Convert to mono if needed
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            
            audio_np = audio.squeeze().numpy()
            
            # Optional: Apply VAD once at file level (not per window) to remove long silent periods
            # This preserves the temporal structure while removing obvious silence
            # Uncomment if you want file-level VAD:
            # audio_np = self.preprocessor.apply_voice_activity_detection(audio_np, top_db=30)
            
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return
        
        # Detect profanity
        profanity_regions = self._detect_profanity_windows(audio_np)
        
        if not profanity_regions:
            print("✅ No profanity detected - audio is clean!")
            # Save original audio
            torchaudio.save(output_path, audio, self.sample_rate)
            self._save_report([], output_path, method)
            return
        
        print(f"🚨 Found {len(profanity_regions)} profanity regions to censor")
        
        # Refine boundaries
        refined_regions = []
        for region in profanity_regions:
            refined = self._refine_boundaries_with_vad(audio_np, region)
            if refined:
                refined_regions.append(refined)
        
        print(f"✨ Refined to {len(refined_regions)} precise regions")
        
        # Apply censoring
        censored_audio = self._apply_censoring(audio_np, refined_regions, method, beep_freq, fade_duration)
        
        # Save censored audio
        torchaudio.save(output_path, torch.from_numpy(censored_audio).unsqueeze(0), self.sample_rate)
        print(f"💾 Censored audio saved to: {output_path}")
        
        # Save report
        self._save_report(refined_regions, output_path, method)
    
    def _apply_censoring(self, audio_np: np.ndarray, regions: List[Tuple[float, float, str, float]], 
                        method: str, beep_freq: float, fade_duration: float) -> np.ndarray:
        """Apply censoring to detected regions"""
        censored_audio = audio_np.copy()
        
        for start_time, end_time, label, confidence in regions:
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            if method == "silence":
                # Replace with silence
                censored_audio[start_sample:end_sample] = 0.0
                
            elif method == "beep":
                # Replace with beep tone
                duration = end_time - start_time
                t = np.linspace(0, duration, end_sample - start_sample, False)
                beep = 0.3 * np.sin(2 * np.pi * beep_freq * t)
                
                # Apply fade in/out to avoid clicks
                fade_samples = int(fade_duration * self.sample_rate)
                if len(beep) > 2 * fade_samples:
                    beep[:fade_samples] *= np.linspace(0, 1, fade_samples)
                    beep[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                censored_audio[start_sample:end_sample] = beep
                
            elif method == "fade":
                # Apply fade out/in
                fade_samples = int(fade_duration * self.sample_rate)
                region_length = end_sample - start_sample
                
                if region_length > 2 * fade_samples:
                    # Fade out
                    censored_audio[start_sample:start_sample + fade_samples] *= np.linspace(1, 0, fade_samples)
                    # Silence in middle
                    censored_audio[start_sample + fade_samples:end_sample - fade_samples] = 0.0
                    # Fade in
                    censored_audio[end_sample - fade_samples:end_sample] *= np.linspace(0, 1, fade_samples)
                else:
                    # Too short for proper fading, just apply silence
                    censored_audio[start_sample:end_sample] = 0.0
            
            print(f"  🔇 Censored {label} at {start_time:.2f}-{end_time:.2f}s (confidence: {confidence:.3f})")
        
        return censored_audio
    
    def _save_report(self, regions: List[Tuple[float, float, str, float]], output_path: str, method: str):
        """Save detailed censoring report"""
        report_path = Path(output_path).with_suffix('.json')
        
        report_data = {
            "censoring_method": method,
            "model_info": {
                "model_dir": self.model_dir,
                "ensemble_size": len(self.models) if self.models else 0,
                "confidence_threshold": self.confidence_threshold,
                "window_size": self.window_size,
                "overlap": self.overlap
            },
            "detections": [],
            "summary": {
                "total_detections": len(regions),
                "total_censored_duration": sum(end - start for start, end, _, _ in regions),
                "detections_by_class": {}
            }
        }
        
        # Process each detection
        for start_time, end_time, label, confidence in regions:
            detection = {
                "label": label,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "duration": round(end_time - start_time, 3),
                "confidence": round(confidence, 3)
            }
            report_data["detections"].append(detection)
            
            # Update class summary
            if label not in report_data["summary"]["detections_by_class"]:
                report_data["summary"]["detections_by_class"][label] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "avg_confidence": 0.0
                }
            
            class_info = report_data["summary"]["detections_by_class"][label]
            class_info["count"] += 1
            class_info["total_duration"] += detection["duration"]
            class_info["avg_confidence"] = (class_info["avg_confidence"] * (class_info["count"] - 1) + confidence) / class_info["count"]
        
        # Round averages
        for label, info in report_data["summary"]["detections_by_class"].items():
            info["total_duration"] = round(info["total_duration"], 3)
            info["avg_confidence"] = round(info["avg_confidence"], 3)
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Censoring report saved to: {report_path}")
        
        # Print summary
        print("\n📊 CENSORING SUMMARY:")
        print(f"   Total detections: {len(regions)}")
        print(f"   Total censored duration: {report_data['summary']['total_censored_duration']:.3f}s")
        if report_data["summary"]["detections_by_class"]:
            print("   Detections by class:")
            for label, info in report_data["summary"]["detections_by_class"].items():
                print(f"     {label}: {info['count']} ({info['total_duration']:.3f}s, avg conf: {info['avg_confidence']:.3f})")

def main():
    """Main function with advanced censoring options"""
    parser = argparse.ArgumentParser(description="Advanced frame-level profanity censoring based on evaluation methods")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input audio file path")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output censored audio file path")
    parser.add_argument("-m", "--model-dir", type=str, required=True, help="Directory containing trained models")
    parser.add_argument("--method", choices=["silence", "beep", "fade"], default="silence", help="Censoring method")
    parser.add_argument("--window-size", type=float, default=2.0, help="Detection window size in seconds (0.5s balanced)")
    parser.add_argument("--overlap", type=float, default=1.0, help="Window overlap in seconds (0.25s = 50% overlap)")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Minimum confidence for detection (lowered from 0.7)")
    parser.add_argument("--beep-freq", type=float, default=1000.0, help="Beep frequency for beep method")
    parser.add_argument("--fade-duration", type=float, default=0.05, help="Fade duration for fade method")
    parser.add_argument("--model-name", type=str, default="airesearch/wav2vec2-large-xlsr-53-th", help="Base model name")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize censoring system
    print("🚀 Initializing Advanced Frame-Level Censoring System")
    print("=" * 60)
    
    censor = AdvancedFrameLevelCensor(
        model_dir=args.model_dir,
        window_size=args.window_size,
        overlap=args.overlap,
        confidence_threshold=args.confidence_threshold,
        model_name=args.model_name
    )
    
    # Check if models were loaded successfully
    if not censor.models:
        print("❌ No models loaded! Please check the model directory path.")
        print(f"   Checked directory: {args.model_dir}")
        print("   Expected structure:")
        print("     models/")
        print("       fold_1/best_model.pt")
        print("       fold_2/best_model.pt")
        print("       ...")
        print("   Or single model files like: best_model.pt, model.pt")
        return
    
    # Process the audio file
    censor.censor_audio_file(
        input_path=args.input,
        output_path=args.output,
        method=args.method,
        beep_freq=args.beep_freq,
        fade_duration=args.fade_duration
    )
    
    print("\n✅ Censoring completed successfully!")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Method: {args.method}")
    print(f"   Report: {Path(args.output).with_suffix('.json')}")

if __name__ == "__main__":
    main()

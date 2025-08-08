#!/usr/bin/env python3
"""
🎯 WINDOWING-BASED EVALUATION SYSTEM
Evaluation system that matches your prototype implementation approach.

Features:
- 0.5 second windows with 0.25 second overlap (50% overlap)
- Evaluates against windowed ground truth labels
- 50% threshold rule for profanity labeling
- Multiple evaluation metrics for production assessment
- Comparison with precise timestamp evaluation

Usage:
    python windowing_evaluation.py --eval-csv csv/eval_windowed.csv --models-dir models/
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import pandas as pd
import librosa
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_recall_fscore_support, accuracy_score
)
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import warnings
warnings.filterwarnings('ignore')

class WindowingEvaluator:
    """
    Windowing-based evaluation that matches your prototype implementation.
    
    This approach:
    1. Slices audio into 0.5 second windows with 0.25 second overlap (50% overlap)
    2. Predicts profanity for each window
    3. Compares against ground truth windows with 50% threshold rule
    4. Provides comprehensive metrics for production assessment
    """
    
    def __init__(self, eval_csv: str, models_dir: str, output_dir: str, 
                 window_length: float = 0.5, overlap: float = 0.25):
        self.eval_csv = eval_csv
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Windowing parameters (now configurable)
        self.window_length = window_length    # Configurable window length
        self.overlap = overlap               # Configurable overlap
        self.hop_length = self.window_length - self.overlap
        self.sample_rate = 16000
        
        # Profanity threshold (50% rule as you described)
        self.profanity_threshold = 0.5
        
        # Label mapping
        self.label_map = {
            'none': 0, 'non_profanity': 0,  # Support both naming conventions
            'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4,
            'ควย': 5, 'สวะ': 6, 'หี': 7, 'แตด': 8,
            'profanity': 1  # Generic profanity label for windowed data
        }
        
        self.id_to_label = {v: k for k, v in self.label_map.items() if k in ['none', 'profanity']}
        
        print(f"🎯 WINDOWING EVALUATION INITIALIZED")
        print(f"Window Length: {self.window_length}s")
        print(f"Overlap: {self.overlap}s ({self.overlap/self.window_length*100:.0f}%)")
        print(f"Hop Length: {self.hop_length}s")
        print(f"50% Profanity Threshold Rule: {self.profanity_threshold}")
    
    def find_available_models(self) -> List[Dict[str, str]]:
        """Find all available trained models."""
        models = []
        
        # Look for different model types
        model_patterns = [
            "*/pytorch_model.bin",    # HuggingFace models
            "*/model.safetensors",    # SafeTensors format
            "*/best_model.pth",       # PyTorch checkpoints
            "*/final_model"           # Final model directories
        ]
        
        for pattern in model_patterns:
            for model_path in self.models_dir.glob(pattern):
                model_dir = model_path.parent if model_path.is_file() else model_path
                
                if model_dir.name not in [m['name'] for m in models]:
                    models.append({
                        'name': model_dir.name,
                        'path': str(model_dir),
                        'type': 'wav2vec2'  # Default to wav2vec2
                    })
        
        print(f"Found {len(models)} available models:")
        for model in models:
            print(f"  - {model['name']} ({model['path']})")
        
        return models
    
    def load_model(self, model_info: Dict[str, str]) -> Tuple[Optional[Any], Optional[Any]]:
        """Load a model and its feature extractor."""
        try:
            print(f"Loading model: {model_info['name']}")
            
            model_path = model_info['path']
            
            # Try loading as HuggingFace model first
            try:
                model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
            except:
                # Try loading with base model and custom weights
                try:
                    model = Wav2Vec2ForSequenceClassification.from_pretrained(
                        "facebook/wav2vec2-base", 
                        num_labels=9  # Adjust based on your classes
                    )
                    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
                    
                    # Load custom weights if available
                    weight_files = list(Path(model_path).glob("*.pth"))
                    if weight_files:
                        weights = torch.load(weight_files[0], map_location='cpu')
                        model.load_state_dict(weights)
                        print(f"Loaded custom weights from {weight_files[0]}")
                except Exception as e:
                    print(f"Failed to load as HuggingFace model: {e}")
                    return None, None
            
            model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            print(f"Model loaded on {device}")
            return model, feature_extractor
            
        except Exception as e:
            print(f"Error loading model {model_info['name']}: {e}")
            return None, None
    
    def load_windowed_ground_truth(self) -> pd.DataFrame:
        """Load the windowed ground truth CSV."""
        try:
            df = pd.read_csv(self.eval_csv)
            print(f"Loaded windowed ground truth: {len(df)} windows")
            
            # Display label distribution
            if 'label' in df.columns:
                label_dist = df['label'].value_counts()
                print(f"Label distribution: {label_dist.to_dict()}")
            
            return df
        except Exception as e:
            print(f"Error loading windowed ground truth: {e}")
            return pd.DataFrame()
    
    def preprocess_audio_window(self, audio: np.ndarray, feature_extractor: Any) -> torch.Tensor:
        """Preprocess audio window for model input (matching your preprocessing)."""
        # Advanced preprocessing (similar to your original approach)
        audio = librosa.effects.preemphasis(audio, coef=0.97)
        audio = audio * np.hamming(len(audio))
        
        # Noise reduction
        noise_threshold = 0.005
        audio = np.where(np.abs(audio) < noise_threshold, 0, audio)
        
        # Normalize
        audio = (audio - audio.mean()) / (audio.std() + 1e-8)
        
        # Feature extraction
        inputs = feature_extractor(
            audio, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.window_length * self.sample_rate)
        )
        
        return inputs['input_values'].squeeze()
    
    def extract_windows_from_audio(self, audio_path: str) -> List[Tuple[float, float, np.ndarray]]:
        """
        Extract overlapping windows from audio file.
        Returns list of (start_time, end_time, audio_window) tuples.
        """
        try:
            # Load full audio file
            audio, sr = torchaudio.load(audio_path)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0)
            
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            
            audio_np = audio.numpy()
            
            # Calculate window parameters
            window_samples = int(self.window_length * self.sample_rate)
            hop_samples = int(self.hop_length * self.sample_rate)
            
            windows = []
            
            # Extract overlapping windows
            start_sample = 0
            while start_sample + window_samples <= len(audio_np):
                end_sample = start_sample + window_samples
                
                start_time = start_sample / self.sample_rate
                end_time = end_sample / self.sample_rate
                
                window_audio = audio_np[start_sample:end_sample]
                
                # Skip silent windows
                if np.max(np.abs(window_audio)) > 0.001:
                    windows.append((start_time, end_time, window_audio))
                
                start_sample += hop_samples
            
            return windows
            
        except Exception as e:
            print(f"Error extracting windows from {audio_path}: {e}")
            return []
    
    def get_ground_truth_for_window(self, window_start: float, window_end: float, 
                                   windowed_df: pd.DataFrame, file_path: str) -> int:
        """
        Get ground truth label for a specific window.
        Uses your 50% threshold rule.
        """
        # Find matching window in ground truth
        file_windows = windowed_df[windowed_df['file_path'] == file_path]
        
        for _, row in file_windows.iterrows():
            gt_start = row['start_time']
            gt_end = row['end_time']
            
            # Check if this window matches (with small tolerance)
            tolerance = 0.05  # 50ms tolerance
            if (abs(window_start - gt_start) < tolerance and 
                abs(window_end - gt_end) < tolerance):
                
                label = row['label']
                return self.label_map.get(label, 0)
        
        # If no exact match found, return none/non_profanity
        return 0
    
    def evaluate_model_on_file(self, model: Any, feature_extractor: Any, 
                              file_path: str, windowed_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model on a single audio file using windowing approach."""
        try:
            # Extract windows from audio
            windows = self.extract_windows_from_audio(file_path)
            
            if not windows:
                return {'predictions': [], 'ground_truth': [], 'confidences': []}
            
            predictions = []
            ground_truth = []
            confidences = []
            processing_times = []
            
            device = next(model.parameters()).device
            
            # Process each window
            for window_start, window_end, window_audio in windows:
                try:
                    start_time = time.time()
                    
                    # Preprocess window
                    audio_input = self.preprocess_audio_window(window_audio, feature_extractor)
                    audio_input = audio_input.to(device)
                    
                    # Get model prediction
                    with torch.no_grad():
                        outputs = model(audio_input.unsqueeze(0))
                        logits = outputs.logits
                        probs = F.softmax(logits, dim=1)
                        
                        # For binary classification (profanity vs non-profanity)
                        if logits.shape[1] == 9:  # Original 9-class model
                            # Convert to binary: class 0 = non-profanity, others = profanity
                            non_prof_prob = probs[0, 0].item()
                            prof_prob = 1 - non_prof_prob
                            pred = 0 if non_prof_prob > prof_prob else 1
                            confidence = max(non_prof_prob, prof_prob)
                        else:  # Binary model
                            pred = torch.argmax(logits, dim=1).item()
                            confidence = torch.max(probs).item()
                    
                    processing_time = time.time() - start_time
                    
                    # Get ground truth for this window
                    gt_label = self.get_ground_truth_for_window(
                        window_start, window_end, windowed_df, file_path
                    )
                    
                    predictions.append(pred)
                    ground_truth.append(gt_label)
                    confidences.append(confidence)
                    processing_times.append(processing_time)
                    
                except Exception as e:
                    print(f"Error processing window {window_start:.2f}-{window_end:.2f}: {e}")
                    continue
            
            return {
                'predictions': predictions,
                'ground_truth': ground_truth,
                'confidences': confidences,
                'processing_times': processing_times,
                'num_windows': len(windows)
            }
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return {'predictions': [], 'ground_truth': [], 'confidences': []}
    
    def calculate_metrics(self, predictions: List[int], ground_truth: List[int], 
                         confidences: List[float]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        if not predictions or not ground_truth:
            return {}
        
        # Basic metrics
        accuracy = accuracy_score(ground_truth, predictions)
        f1_weighted = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
        f1_macro = f1_score(ground_truth, predictions, average='macro', zero_division=0)
        
        # Convert to binary for profanity detection
        binary_gt = [1 if label != 0 else 0 for label in ground_truth]  # 0 = clean, others = profanity
        binary_pred = [1 if label != 0 else 0 for label in predictions]
        
        # Binary profanity detection metrics
        binary_accuracy = accuracy_score(binary_gt, binary_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            binary_gt, binary_pred, average='binary', zero_division=0
        )
        
        # Per-class metrics
        per_class_metrics = {}
        for class_id in set(ground_truth + predictions):
            class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
                ground_truth, predictions, labels=[class_id], average=None, zero_division=0
            )
            if len(class_precision) > 0:
                per_class_metrics[self.id_to_label.get(class_id, f'class_{class_id}')] = {
                    'precision': class_precision[0],
                    'recall': class_recall[0],
                    'f1': class_f1[0],
                    'support': class_support[0]
                }
        
        # Confidence analysis
        confidence_mean = np.mean(confidences) if confidences else 0.0
        confidence_std = np.std(confidences) if confidences else 0.0
        
        return {
            'accuracy': accuracy,
            'binary_accuracy': binary_accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'precision': precision,
            'recall': recall,
            'f1_binary': f1,
            'per_class_metrics': per_class_metrics,
            'confidence_mean': confidence_mean,
            'confidence_std': confidence_std,
            'num_samples': len(predictions)
        }
    
    def evaluate_model(self, model_info: Dict[str, str]) -> Dict[str, Any]:
        """Evaluate a single model using windowing approach."""
        print(f"\n🔍 EVALUATING: {model_info['name']}")
        print("=" * 60)
        
        # Load model
        model, feature_extractor = self.load_model(model_info)
        if model is None:
            return None
        
        # Load windowed ground truth
        windowed_df = self.load_windowed_ground_truth()
        if windowed_df.empty:
            return None
        
        # Get unique files to evaluate
        unique_files = windowed_df['file_path'].unique()
        print(f"Found {len(unique_files)} files to evaluate")
        
        all_predictions = []
        all_ground_truth = []
        all_confidences = []
        all_processing_times = []
        file_results = {}
        
        # Process each file
        for file_path in unique_files:
            print(f"Processing: {os.path.basename(file_path)}")
            
            # Evaluate on this file
            results = self.evaluate_model_on_file(
                model, feature_extractor, file_path, windowed_df
            )
            
            if results['predictions']:
                all_predictions.extend(results['predictions'])
                all_ground_truth.extend(results['ground_truth'])
                all_confidences.extend(results['confidences'])
                all_processing_times.extend(results['processing_times'])
                
                file_results[file_path] = results
                
                print(f"  Windows: {results['num_windows']}, "
                      f"Predictions: {len(results['predictions'])}")
        
        if not all_predictions:
            print(f"❌ No successful predictions for model {model_info['name']}")
            return None
        
        # Calculate overall metrics
        metrics = self.calculate_metrics(all_predictions, all_ground_truth, all_confidences)
        
        # Add timing metrics
        if all_processing_times:
            metrics['avg_processing_time'] = np.mean(all_processing_times)
            metrics['real_time_factor'] = self.window_length / np.mean(all_processing_times)
        
        print(f"📊 RESULTS:")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  F1 (Binary): {metrics.get('f1_binary', 0):.4f}")
        print(f"  F1 (Weighted): {metrics.get('f1_weighted', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0):.4f}")
        print(f"  Windows Processed: {metrics.get('num_samples', 0)}")
        
        if 'real_time_factor' in metrics:
            print(f"  Real-time Factor: {metrics['real_time_factor']:.2f}x")
        
        return {
            'model_name': model_info['name'],
            'metrics': metrics,
            'file_results': file_results,
            'predictions': all_predictions,
            'ground_truth': all_ground_truth,
            'confidences': all_confidences
        }
    
    def create_visualizations(self, results: List[Dict[str, Any]]):
        """Create visualization plots."""
        if not results:
            return
        
        # Set up plotting
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Windowing-Based Evaluation Results', fontsize=16, fontweight='bold')
        
        model_names = [r['model_name'] for r in results]
        
        # Accuracy comparison
        accuracies = [r['metrics']['accuracy'] for r in results]
        axes[0, 0].bar(model_names, accuracies)
        axes[0, 0].set_title('Overall Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        f1_scores = [r['metrics']['f1_binary'] for r in results]
        axes[0, 1].bar(model_names, f1_scores)
        axes[0, 1].set_title('Binary F1 Score (Profanity Detection)')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Precision vs Recall
        precisions = [r['metrics']['precision'] for r in results]
        recalls = [r['metrics']['recall'] for r in results]
        axes[1, 0].scatter(recalls, precisions)
        for i, name in enumerate(model_names):
            axes[1, 0].annotate(name, (recalls[i], precisions[i]))
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision vs Recall')
        
        # Real-time performance
        if 'real_time_factor' in results[0]['metrics']:
            rt_factors = [r['metrics'].get('real_time_factor', 0) for r in results]
            axes[1, 1].bar(model_names, rt_factors)
            axes[1, 1].set_title('Real-time Performance')
            axes[1, 1].set_ylabel('Real-time Factor (>1 = faster than real-time)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'windowing_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Confusion matrix for best model
        best_model = max(results, key=lambda x: x['metrics']['f1_binary'])
        
        cm = confusion_matrix(best_model['ground_truth'], best_model['predictions'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-Profanity', 'Profanity'],
                   yticklabels=['Non-Profanity', 'Profanity'])
        plt.title(f'Confusion Matrix - {best_model["model_name"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix_windowing.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to {self.output_dir}")
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save detailed results to files."""
        if not results:
            return
        
        # Summary report
        with open(self.output_dir / 'windowing_evaluation_summary.txt', 'w', encoding='utf-8') as f:
            f.write("🎯 WINDOWING-BASED EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write("Evaluation Method: Prototype-Matching Windowing\n")
            f.write(f"Window Length: {self.window_length}s\n")
            f.write(f"Overlap: {self.overlap}s ({self.overlap/self.window_length*100:.0f}%)\n")
            f.write(f"Hop Length: {self.hop_length}s\n")
            f.write(f"50% Profanity Threshold Rule Applied\n\n")
            
            for result in results:
                f.write(f"Model: {result['model_name']}\n")
                f.write("-" * 40 + "\n")
                metrics = result['metrics']
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"F1 (Binary): {metrics['f1_binary']:.4f}\n")
                f.write(f"F1 (Weighted): {metrics['f1_weighted']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                f.write(f"Windows Processed: {metrics['num_samples']}\n")
                if 'real_time_factor' in metrics:
                    f.write(f"Real-time Factor: {metrics['real_time_factor']:.2f}x\n")
                f.write("\n")
        
        # Detailed results JSON
        import json
        
        # Convert numpy types to Python types for JSON serialization
        json_results = []
        for result in results:
            json_result = {
                'model_name': result['model_name'],
                'metrics': {k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                           for k, v in result['metrics'].items() if k != 'per_class_metrics'},
                'predictions': [int(p) for p in result['predictions']],
                'ground_truth': [int(gt) for gt in result['ground_truth']],
                'confidences': [float(c) for c in result['confidences']]
            }
            json_results.append(json_result)
        
        with open(self.output_dir / 'windowing_evaluation_detailed.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"📁 Results saved to {self.output_dir}")
    
    def run_evaluation(self):
        """Run complete windowing-based evaluation."""
        print("🎯 STARTING WINDOWING-BASED EVALUATION")
        print("=" * 60)
        print("This evaluation matches your prototype implementation:")
        print("- 0.5 second windows with 0.25 second overlap")
        print("- 50% profanity threshold rule")
        print("- Direct comparison with windowed ground truth")
        print("=" * 60)
        
        # Find available models
        models = self.find_available_models()
        if not models:
            print("❌ No models found for evaluation!")
            return
        
        results = []
        
        # Evaluate each model
        for model_info in models:
            result = self.evaluate_model(model_info)
            if result:
                results.append(result)
                print(f"✅ {model_info['name']} evaluated successfully")
            else:
                print(f"❌ Failed to evaluate {model_info['name']}")
        
        if not results:
            print("❌ No successful evaluations!")
            return
        
        # Create visualizations and save results
        self.create_visualizations(results)
        self.save_results(results)
        
        # Print final summary
        print("\n🎯 EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Models evaluated: {len(results)}")
        print(f"Results saved to: {self.output_dir}")
        
        # Show best model
        best_model = max(results, key=lambda x: x['metrics']['f1_binary'])
        print(f"\n🏆 BEST MODEL: {best_model['model_name']}")
        print(f"F1 Score (Binary): {best_model['metrics']['f1_binary']:.4f}")
        print(f"Accuracy: {best_model['metrics']['accuracy']:.4f}")
        print(f"Precision: {best_model['metrics']['precision']:.4f}")
        print(f"Recall: {best_model['metrics']['recall']:.4f}")
        
        return results

def print_evaluation_analysis():
    """Print detailed analysis of the windowing approach."""
    print("\n" + "="*80)
    print("📊 WINDOWING EVALUATION APPROACH ANALYSIS")
    print("="*80)
    
    print("\n🎯 YOUR CURRENT APPROACH:")
    print("- Window Length: 0.5 seconds")
    print("- Overlap: 0.25 seconds (50% overlap)")
    print("- Hop Length: 0.25 seconds")
    print("- 50% Threshold Rule: If profanity ≥50% of window → profanity label")
    print("- Direct prototype matching evaluation")
    
    print("\n✅ ADVANTAGES:")
    print("1. 🎯 PROTOTYPE MATCHING: Exactly matches your implementation")
    print("2. 🚀 PRACTICAL: Realistic for real-time processing")
    print("3. ⚡ EFFICIENT: Fixed window size allows batch processing")
    print("4. 🔄 OVERLAP: 50% overlap reduces edge effects")
    print("5. 📊 SIMPLE: Easy to understand and implement")
    print("6. 🎚️ TUNABLE: Can adjust window size and overlap")
    print("7. 💻 REAL-TIME: Suitable for streaming applications")
    
    print("\n❌ DISADVANTAGES:")
    print("1. 📏 FIXED RESOLUTION: May miss short profanity words")
    print("2. 🎯 BOUNDARY EFFECTS: Profanity split across windows")
    print("3. 📊 THRESHOLD SENSITIVITY: 50% rule may be too rigid")
    print("4. 🔄 OVERLAP ARTIFACTS: Same profanity detected multiple times")
    print("5. ⏱️ TEMPORAL ACCURACY: Less precise than frame-level detection")
    print("6. 🏷️ LABEL GRANULARITY: Binary classification loses specific profanity types")
    
    print("\n🔍 REAL-WORLD USAGE IMPLICATIONS:")
    print("1. 🎬 VIDEO PROCESSING: Good for continuous video streams")
    print("2. 📞 LIVE AUDIO: Suitable for real-time voice chat")
    print("3. 🎙️ STREAMING: Works well for live broadcasts")
    print("4. ⚡ LATENCY: ~0.5s delay for detection")
    print("5. 💾 MEMORY: Fixed memory usage per window")
    print("6. 🔄 PROCESSING: Can process in parallel")
    
    print("\n⚠️ POTENTIAL ISSUES:")
    print("1. 🗣️ FAST SPEECH: Quick profanity might be missed")
    print("2. 🔇 QUIET PROFANITY: Low volume words in noisy environment")
    print("3. 🎭 CONTEXT: No consideration of surrounding context")
    print("4. 📊 FALSE POSITIVES: Clean speech misclassified in noisy windows")
    print("5. 🎯 EDGE CASES: Profanity exactly at window boundaries")
    
    print("\n🔧 OPTIMIZATION SUGGESTIONS:")
    print("1. 📏 ADAPTIVE WINDOWS: Vary window size based on speech rate")
    print("2. 🎯 CONFIDENCE SMOOTHING: Use temporal smoothing across windows")
    print("3. 🔄 MULTIPLE SCALES: Use multiple window sizes simultaneously")
    print("4. 🎚️ DYNAMIC THRESHOLD: Adjust 50% rule based on context")
    print("5. 📊 POST-PROCESSING: Merge adjacent profanity windows")
    print("6. 🎭 CONTEXT INTEGRATION: Consider surrounding clean/profane windows")
    
    print("\n" + "="*80)

def print_precise_vs_windowing_comparison():
    """Compare precise timestamp vs windowing evaluation."""
    print("\n" + "="*80)
    print("⚖️  PRECISE TIMESTAMPS vs WINDOWING COMPARISON")
    print("="*80)
    
    print("\n📍 PRECISE TIMESTAMP EVALUATION (eval.csv):")
    print("✅ Advantages:")
    print("   - 🎯 EXACT TIMING: Perfect profanity boundaries")
    print("   - 🏷️ DETAILED LABELS: Specific profanity types (กู, เย็ด, etc.)")
    print("   - 📊 HIGH PRECISION: No temporal artifacts")
    print("   - 🔬 RESEARCH QUALITY: Good for model development")
    print("   - 📈 OPTIMISTIC METRICS: Shows model's best potential")
    
    print("❌ Disadvantages:")
    print("   - 🚫 UNREALISTIC: Doesn't match real-world usage")
    print("   - 🎭 MANUAL DEPENDENCY: Requires human annotation")
    print("   - ⚡ NOT REAL-TIME: Can't be used in production")
    print("   - 📊 OVERESTIMATED PERFORMANCE: Inflated metrics")
    print("   - 🔄 NO OVERLAP HANDLING: Doesn't test windowing artifacts")
    
    print("\n🪟 WINDOWING EVALUATION (eval_windowed.csv):")
    print("✅ Advantages:")
    print("   - 🎯 REALISTIC: Matches actual deployment")
    print("   - ⚡ REAL-TIME READY: Direct prototype validation")
    print("   - 🔄 HANDLES OVERLAP: Tests window artifacts")
    print("   - 💻 PRODUCTION METRIC: Meaningful for users")
    print("   - 📊 CONSERVATIVE: More honest performance assessment")
    
    print("❌ Disadvantages:")
    print("   - 📏 LOWER RESOLUTION: May miss fine-grained timing")
    print("   - 🏷️ BINARY LABELS: Loses specific profanity information")
    print("   - 📊 PESSIMISTIC: May underestimate model capability")
    print("   - 🔧 THRESHOLD DEPENDENT: Sensitive to 50% rule")
    print("   - 🎚️ PARAMETER SENSITIVE: Window size affects results")
    
    print("\n🎯 RECOMMENDATION:")
    print("For your use case, I recommend:")
    print("1. 🎯 PRIMARY: Use windowing evaluation (matches your prototype)")
    print("2. 🔬 SECONDARY: Use precise evaluation for model development")
    print("3. 📊 BOTH: Compare both to understand the performance gap")
    print("4. 🎚️ TUNE: Optimize thresholds on windowing performance")
    print("5. 📈 TRACK: Monitor both metrics during training")
    
    print("\n🔄 BETTER WAYS TO USE PRECISE eval.csv:")
    print("1. 🎯 FRAME-LEVEL: 25ms frame classification (research method)")
    print("2. 🔄 SLIDING EVALUATION: Multiple window sizes on precise data")
    print("3. 📊 TEMPORAL LOCALIZATION: Measure timing accuracy")
    print("4. 🎚️ CONFIDENCE CALIBRATION: Tune prediction thresholds")
    print("5. 🔬 ERROR ANALYSIS: Understand model failure modes")
    print("6. 📈 CURRICULUM LEARNING: Progressive training on harder examples")
    
    print("\n" + "="*80)

def create_windowed_from_precise(precise_csv: str, window_length: float, 
                                overlap: float, output_csv: str) -> str:
    """
    Create windowed evaluation data from precise timestamps.
    
    Args:
        precise_csv: Path to eval.csv with precise timestamps
        window_length: Window length in seconds
        overlap: Overlap in seconds
        output_csv: Path to save windowed data
        
    Returns:
        Path to the created windowed CSV file
    """
    print(f"Creating windowed data: {window_length}s windows, {overlap}s overlap")
    
    import librosa
    
    # Read precise evaluation data
    df = pd.read_csv(precise_csv)
    
    windowed_data = []
    threshold = 0.5  # 50% threshold for profanity labeling
    
    # Process each file
    for file_path in df['file_path'].unique():
        file_df = df[df['file_path'] == file_path].copy()
        
        # Get audio duration
        audio_path = file_path.replace("./eval\\", "eval/").replace("./eval/", "eval/")
        if not os.path.exists(audio_path):
            continue
            
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr
        except:
            continue
        
        # Create windows
        start = 0.0
        while start < duration:
            end = min(start + window_length, duration)
            
            # Calculate profanity overlap in this window
            profanity_duration = 0.0
            window_label = 'none'
            profanity_types = {}
            
            for _, row in file_df.iterrows():
                if row['label'] == 'none':
                    continue
                    
                # Calculate overlap between window and profanity segment
                overlap_start = max(start, row['start_time'])
                overlap_end = min(end, row['end_time'])
                
                if overlap_start < overlap_end:
                    segment_overlap = overlap_end - overlap_start
                    profanity_duration += segment_overlap
                    
                    # Track profanity types and their durations
                    if row['label'] not in profanity_types:
                        profanity_types[row['label']] = 0
                    profanity_types[row['label']] += segment_overlap
            
            # Apply 50% threshold rule
            if profanity_duration >= (window_length * threshold):
                # Find the most prominent profanity type in this window
                if profanity_types:
                    window_label = max(profanity_types.items(), key=lambda x: x[1])[0]
            
            windowed_data.append({
                'file_path': file_path,
                'start_time': start,
                'end_time': end,
                'label': window_label
            })
            
            start += overlap
            
            if end >= duration:
                break
    
    # Save windowed data
    windowed_df = pd.DataFrame(windowed_data)
    windowed_df.to_csv(output_csv, index=False)
    
    print(f"Created {len(windowed_df)} windows")
    print(f"Label distribution: {windowed_df['label'].value_counts().to_dict()}")
    print(f"Saved to: {output_csv}")
    
    return output_csv

def main():
    parser = argparse.ArgumentParser(description='Windowing-Based Model Evaluation')
    parser.add_argument('--eval-csv', default='csv/eval_windowed.csv',
                       help='CSV file with windowed ground truth labels')
    parser.add_argument('--models-dir', default='models/',
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', default='evaluation_results/windowing_eval',
                       help='Output directory for results')
    parser.add_argument('--analysis', action='store_true',
                       help='Print detailed analysis of the approach')
    parser.add_argument('--window-length', type=float, default=0.5,
                       help='Window length in seconds (default: 0.5)')
    parser.add_argument('--overlap', type=float, default=0.25,
                       help='Overlap in seconds (default: 0.25)')
    parser.add_argument('--use-precise-csv', action='store_true',
                       help='Create windowed data from precise eval.csv on-the-fly')
    
    args = parser.parse_args()
    
    if args.analysis:
        print_evaluation_analysis()
        print_precise_vs_windowing_comparison()
        return
    
    # Handle different evaluation modes
    eval_csv = args.eval_csv
    
    if args.use_precise_csv:
        # Create windowed data from precise timestamps
        eval_csv = create_windowed_from_precise(
            "csv/eval.csv", 
            args.window_length, 
            args.overlap,
            f"csv/eval_windowed_{args.window_length}s.csv"
        )
    
    # Run windowing evaluation
    evaluator = WindowingEvaluator(
        eval_csv, 
        args.models_dir, 
        args.output_dir,
        window_length=args.window_length,
        overlap=args.overlap
    )
    results = evaluator.run_evaluation()
    
    # Print analysis
    print_evaluation_analysis()
    print_precise_vs_windowing_comparison()

if __name__ == "__main__":
    main()

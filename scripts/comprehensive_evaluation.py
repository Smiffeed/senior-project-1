import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import json
import os
from tqdm import tqdm
import glob

# Import our models and utilities
from ultimate_model_training import (
    AdvancedModelPipeline, MultiModalAudioClassifier, 
    AdvancedProfanityDataset, EnhancedAudioPreprocessor
)
from simplified_ultimate_training import EnhancedAudioClassifier, SimpleProfanityDataset, SimpleAudioPreprocessor
from advanced_model_improvements import AdvancedEvaluator, ModelEnsemble
from transformers import Wav2Vec2FeatureExtractor

# Constants
LABEL_MAP = {
    'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4,
    'ควย': 5, 'สวะ': 6, 'หี': 7, 'แตด': 8
}
NUM_LABELS = len(LABEL_MAP)
CLASS_NAMES = list(LABEL_MAP.keys())

class ComprehensiveEvaluator:
    """Comprehensive evaluation suite for advanced audio models."""
    
    def __init__(self, model_dir, model_name="airesearch/wav2vec2-large-xlsr-53-th"):
        self.model_dir = model_dir
        self.model_name = model_name
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, return_attention_mask=True, do_normalize=True
        )
        self.preprocessor = SimpleAudioPreprocessor()  # Use the simple preprocessor
        self.evaluator = AdvancedEvaluator(CLASS_NAMES)
        
    def load_model(self, fold_num, stage='best'):
        """Load a trained model from specific fold and stage."""
        if stage == 'best':
            model_path = os.path.join(self.model_dir, f'fold_{fold_num}', 'best_model.pt')
        else:
            model_path = os.path.join(self.model_dir, f'fold_{fold_num}', stage, 'model.pt')
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return None
        
        # Create model using the simplified architecture that was actually trained
        model = EnhancedAudioClassifier(
            self.model_name, NUM_LABELS
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def load_ensemble_models(self, num_folds=5, stage='best'):
        """Load ensemble of models from all folds."""
        models = []
        for fold in range(1, num_folds + 1):
            model = self.load_model(fold, stage)
            if model is not None:
                models.append(model)
        
        print(f"Loaded {len(models)} models for ensemble")
        return models
    
    def evaluate_single_model(self, model, test_df, save_prefix="single_model"):
        """Evaluate a single model comprehensively."""
        print(f"Evaluating single model...")
        
        # Create test dataset using the simplified dataset since that's what the model was trained with
        test_dataset = SimpleProfanityDataset(
            test_df, self.feature_extractor, self.preprocessor, mode='test'
        )
        
        # Predictions
        all_predictions = []
        all_labels = []
        all_logits = []
        all_uncertainties = []
        
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
                sample = test_dataset[i]
                if sample is None:
                    continue
                
                # Prepare input - simplified for the basic model
                inputs = {
                    'input_values': sample['input_values'].unsqueeze(0),
                    'attention_mask': sample['attention_mask'].unsqueeze(0)
                }
                
                # The simplified model doesn't use additional features
                
                # Forward pass
                outputs = model(**inputs)
                
                # Collect results
                prediction = torch.argmax(outputs['logits'], dim=-1).item()
                all_predictions.append(prediction)
                all_labels.append(sample['label'].item())
                all_logits.append(outputs['logits'].squeeze().cpu().numpy())
                
                if 'uncertainty' in outputs:
                    all_uncertainties.append(outputs['uncertainty'].item())
        
        # Calculate metrics
        results = self._calculate_comprehensive_metrics(
            all_labels, all_predictions, all_logits, all_uncertainties
        )
        
        # Save results
        self._save_evaluation_results(results, save_prefix)
        
        # Generate plots
        self._generate_evaluation_plots(results, save_prefix)
        
        return results
    
    def evaluate_ensemble(self, test_df, num_folds=5, save_prefix="ensemble"):
        """Evaluate ensemble of models."""
        print(f"Evaluating ensemble of {num_folds} models...")
        
        # Load models
        models = self.load_ensemble_models(num_folds)
        if not models:
            print("No models loaded for ensemble evaluation")
            return None
        
        ensemble = ModelEnsemble(models)
        
        # Create test dataset using simplified dataset
        test_dataset = SimpleProfanityDataset(
            test_df, self.feature_extractor, self.preprocessor, mode='test'
        )
        
        # Predictions
        all_predictions = []
        all_labels = []
        all_logits = []
        all_uncertainties = []
        
        for i in tqdm(range(len(test_dataset)), desc="Ensemble evaluation"):
            sample = test_dataset[i]
            if sample is None:
                continue
            
            # Prepare input - simplified for basic model
            inputs = {
                'input_values': sample['input_values'].unsqueeze(0),
                'attention_mask': sample['attention_mask'].unsqueeze(0)
            }
            
            # The simplified model doesn't use additional features
            
            # Ensemble prediction
            ensemble_output = ensemble.predict(inputs)
            prediction = torch.argmax(ensemble_output['predictions'], dim=-1).item()
            
            all_predictions.append(prediction)
            all_labels.append(sample['label'].item())
            all_logits.append(ensemble_output['predictions'].squeeze().cpu().numpy())
            all_uncertainties.append(ensemble_output['uncertainty'].item())
        
        # Calculate metrics
        results = self._calculate_comprehensive_metrics(
            all_labels, all_predictions, all_logits, all_uncertainties
        )
        
        results['ensemble_size'] = len(models)
        
        # Save results
        self._save_evaluation_results(results, save_prefix)
        
        # Generate plots
        self._generate_evaluation_plots(results, save_prefix)
        
        return results
    
    def _calculate_comprehensive_metrics(self, labels, predictions, logits, uncertainties):
        """Calculate comprehensive evaluation metrics."""
        labels = np.array(labels)
        predictions = np.array(predictions)
        logits = np.array(logits)
        uncertainties = np.array(uncertainties) if uncertainties else None
        
        # Basic metrics
        accuracy = np.mean(labels == predictions)
        
        # Classification report - handle missing classes
        unique_labels = sorted(set(labels))
        present_class_names = [CLASS_NAMES[i] for i in unique_labels]
        
        clf_report = classification_report(
            labels, predictions, labels=unique_labels, target_names=present_class_names, output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Per-class metrics
        per_class_accuracy = {}
        per_class_precision = {}
        per_class_recall = {}
        per_class_f1 = {}
        
        for i, class_name in enumerate(CLASS_NAMES):
            if class_name in clf_report:
                per_class_precision[class_name] = clf_report[class_name]['precision']
                per_class_recall[class_name] = clf_report[class_name]['recall']
                per_class_f1[class_name] = clf_report[class_name]['f1-score']
                
                # Class accuracy
                class_mask = labels == i
                if np.any(class_mask):
                    per_class_accuracy[class_name] = np.mean(predictions[class_mask] == i)
                else:
                    per_class_accuracy[class_name] = 0.0
        
        # ROC curves and AUC (for multi-class)
        roc_auc = {}
        if logits.shape[1] == NUM_LABELS:
            # Binarize labels for multi-class ROC
            labels_bin = label_binarize(labels, classes=range(NUM_LABELS))
            
            for i, class_name in enumerate(CLASS_NAMES):
                if np.any(labels_bin[:, i]):  # Only if class exists in test set
                    fpr, tpr, _ = roc_curve(labels_bin[:, i], logits[:, i])
                    roc_auc[class_name] = auc(fpr, tpr)
        
        # Profanity vs Non-profanity metrics
        profanity_labels = labels != 0  # Non-'none' is profanity
        profanity_predictions = predictions != 0
        profanity_accuracy = np.mean(profanity_labels == profanity_predictions)
        
        # Uncertainty analysis
        uncertainty_metrics = {}
        if uncertainties is not None:
            uncertainty_metrics = {
                'mean_uncertainty': np.mean(uncertainties),
                'std_uncertainty': np.std(uncertainties),
                'uncertainty_accuracy_correlation': np.corrcoef(
                    uncertainties, (labels == predictions).astype(float)
                )[0, 1]
            }
        
        # Error analysis
        error_analysis = self._analyze_errors(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'profanity_accuracy': profanity_accuracy,
            'classification_report': clf_report,
            'confusion_matrix': cm,
            'per_class_accuracy': per_class_accuracy,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'per_class_f1': per_class_f1,
            'roc_auc': roc_auc,
            'uncertainty_metrics': uncertainty_metrics,
            'error_analysis': error_analysis,
            'predictions': predictions.tolist(),
            'labels': labels.tolist(),
            'logits': logits.tolist() if logits.ndim > 1 else [],
            'uncertainties': uncertainties.tolist() if uncertainties is not None else []
        }
    
    def _analyze_errors(self, labels, predictions):
        """Analyze prediction errors."""
        errors = labels != predictions
        error_analysis = {
            'total_errors': int(np.sum(errors)),
            'error_rate': float(np.mean(errors)),
            'error_by_class': {},
            'confusion_patterns': {}
        }
        
        # Error rate by true class
        for i, class_name in enumerate(CLASS_NAMES):
            class_mask = labels == i
            if np.any(class_mask):
                class_errors = errors[class_mask]
                error_analysis['error_by_class'][class_name] = {
                    'count': int(np.sum(class_errors)),
                    'rate': float(np.mean(class_errors))
                }
        
        # Common confusion patterns
        for true_class in range(NUM_LABELS):
            for pred_class in range(NUM_LABELS):
                if true_class != pred_class:
                    confusion_count = np.sum((labels == true_class) & (predictions == pred_class))
                    if confusion_count > 0:
                        pattern = f"{CLASS_NAMES[true_class]} -> {CLASS_NAMES[pred_class]}"
                        error_analysis['confusion_patterns'][pattern] = int(confusion_count)
        
        return error_analysis
    
    def _save_evaluation_results(self, results, prefix):
        """Save evaluation results to files."""
        save_dir = os.path.join(self.model_dir, 'evaluation_results')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save JSON results (excluding non-serializable items)
        json_results = {k: v for k, v in results.items() 
                       if k not in ['confusion_matrix']}
        
        with open(os.path.join(save_dir, f'{prefix}_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save confusion matrix separately
        np.save(os.path.join(save_dir, f'{prefix}_confusion_matrix.npy'), 
                results['confusion_matrix'])
        
        # Save classification report
        unique_labels = sorted(set(results['labels']))
        present_class_names = [CLASS_NAMES[i] for i in unique_labels]
        
        with open(os.path.join(save_dir, f'{prefix}_classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write(classification_report(
                results['labels'], results['predictions'], 
                labels=unique_labels, target_names=present_class_names
            ))
        
        print(f"Results saved to {save_dir}")
    
    def _generate_evaluation_plots(self, results, prefix):
        """Generate comprehensive evaluation plots."""
        save_dir = os.path.join(self.model_dir, 'evaluation_plots')
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                   cmap='Blues')
        plt.title(f'Confusion Matrix - {prefix}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_confusion_matrix.png'), dpi=300)
        plt.close()
        
        # 2. Per-class metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy
        classes = list(results['per_class_accuracy'].keys())
        accuracies = list(results['per_class_accuracy'].values())
        axes[0, 0].bar(classes, accuracies)
        axes[0, 0].set_title('Per-Class Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precision
        precisions = [results['per_class_precision'].get(c, 0) for c in classes]
        axes[0, 1].bar(classes, precisions)
        axes[0, 1].set_title('Per-Class Precision')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Recall
        recalls = [results['per_class_recall'].get(c, 0) for c in classes]
        axes[1, 0].bar(classes, recalls)
        axes[1, 0].set_title('Per-Class Recall')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # F1-score
        f1_scores = [results['per_class_f1'].get(c, 0) for c in classes]
        axes[1, 1].bar(classes, f1_scores)
        axes[1, 1].set_title('Per-Class F1-Score')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_per_class_metrics.png'), dpi=300)
        plt.close()
        
        # 3. ROC Curves (if available)
        if results['roc_auc']:
            plt.figure(figsize=(12, 8))
            for class_name, auc_score in results['roc_auc'].items():
                # Note: This is simplified - in practice you'd need to store FPR/TPR
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                plt.text(0.5, 0.3 + len(results['roc_auc']) * 0.05, 
                        f'{class_name}: AUC = {auc_score:.3f}')
            
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves - {prefix}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, f'{prefix}_roc_curves.png'), dpi=300)
            plt.close()
        
        # 4. Uncertainty analysis (if available)
        if results['uncertainties']:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Uncertainty distribution
            axes[0].hist(results['uncertainties'], bins=30, alpha=0.7)
            axes[0].set_title('Uncertainty Distribution')
            axes[0].set_xlabel('Uncertainty')
            axes[0].set_ylabel('Frequency')
            
            # Uncertainty vs Accuracy
            correct = np.array(results['labels']) == np.array(results['predictions'])
            axes[1].scatter(results['uncertainties'], correct.astype(int), alpha=0.6)
            axes[1].set_title('Uncertainty vs Accuracy')
            axes[1].set_xlabel('Uncertainty')
            axes[1].set_ylabel('Correct Prediction')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{prefix}_uncertainty_analysis.png'), dpi=300)
            plt.close()
        
        print(f"Plots saved to {save_dir}")
    
    def compare_models(self, test_df, model_configs):
        """Compare multiple model configurations."""
        print("Comparing multiple models...")
        
        comparison_results = {}
        
        for config_name, config in model_configs.items():
            print(f"\nEvaluating {config_name}...")
            
            if config['type'] == 'single':
                model = self.load_model(config['fold'], config.get('stage', 'best'))
                if model is not None:
                    results = self.evaluate_single_model(
                        model, test_df, save_prefix=f"comparison_{config_name}"
                    )
                    comparison_results[config_name] = results
            
            elif config['type'] == 'ensemble':
                results = self.evaluate_ensemble(
                    test_df, config.get('num_folds', 5), 
                    save_prefix=f"comparison_{config_name}"
                )
                if results is not None:
                    comparison_results[config_name] = results
        
        # Generate comparison plots
        self._generate_comparison_plots(comparison_results)
        
        return comparison_results
    
    def _generate_comparison_plots(self, comparison_results):
        """Generate plots comparing different models."""
        save_dir = os.path.join(self.model_dir, 'comparison_plots')
        os.makedirs(save_dir, exist_ok=True)
        
        if not comparison_results:
            return
        
        # Overall accuracy comparison
        model_names = list(comparison_results.keys())
        accuracies = [results['accuracy'] for results in comparison_results.values()]
        profanity_accuracies = [results['profanity_accuracy'] for results in comparison_results.values()]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall accuracy
        axes[0].bar(model_names, accuracies)
        axes[0].set_title('Overall Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Profanity detection accuracy
        axes[1].bar(model_names, profanity_accuracies)
        axes[1].set_title('Profanity Detection Accuracy')
        axes[1].set_ylabel('Accuracy')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), dpi=300)
        plt.close()
        
        # Per-class F1 score comparison
        plt.figure(figsize=(15, 8))
        
        for i, (model_name, results) in enumerate(comparison_results.items()):
            f1_scores = [results['per_class_f1'].get(class_name, 0) for class_name in CLASS_NAMES]
            x_pos = np.arange(len(CLASS_NAMES)) + i * 0.25
            plt.bar(x_pos, f1_scores, width=0.25, label=model_name, alpha=0.8)
        
        plt.xlabel('Classes')
        plt.ylabel('F1-Score')
        plt.title('Per-Class F1-Score Comparison')
        plt.xticks(np.arange(len(CLASS_NAMES)) + 0.25, CLASS_NAMES, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'f1_score_comparison.png'), dpi=300)
        plt.close()
        
        print(f"Comparison plots saved to {save_dir}")

    def evaluate_full_audio_files(self, audio_files, save_prefix="full_audio_scan"):
        """Evaluate entire audio files using sliding windows to detect all profanities."""
        print(f"Scanning {len(audio_files)} audio files for all profanities...")
        
        all_detections = []
        window_size = 0.5  # seconds
        hop_length = 0.25  # seconds
        
        for audio_file in tqdm(audio_files, desc="Scanning audio files"):
            if not os.path.exists(audio_file):
                print(f"Warning: File not found: {audio_file}")
                continue
            
            try:
                # Load entire audio file
                import librosa
                audio, sr = librosa.load(audio_file, sr=16000)
                audio_length = len(audio) / sr
                
                file_detections = []
                
                # Scan with sliding windows
                for window_start in np.arange(0, audio_length - window_size, hop_length):
                    window_end = window_start + window_size
                    
                    # Extract audio segment
                    start_sample = int(window_start * sr)
                    end_sample = int(window_end * sr)
                    audio_segment = audio[start_sample:end_sample]
                    
                    # Ensure correct length
                    if len(audio_segment) < int(window_size * sr):
                        audio_segment = np.pad(audio_segment, 
                                             (0, int(window_size * sr) - len(audio_segment)), 
                                             'constant')
                    
                    # Preprocess
                    audio_segment = self.preprocessor.preprocess_audio_simple(audio_segment)
                    
                    # Get model prediction
                    inputs = self.feature_extractor(
                        audio_segment, 
                        sampling_rate=16000, 
                        return_tensors="pt", 
                        padding=True
                    )
                    
                    # Load a model for prediction (use fold 1 by default)
                    if not hasattr(self, '_scan_model'):
                        self._scan_model = self.load_model(1, 'best')
                    
                    if self._scan_model is not None:
                        with torch.no_grad():
                            outputs = self._scan_model(**inputs)
                            probs = torch.softmax(outputs['logits'], dim=-1)
                            prediction = torch.argmax(outputs['logits'], dim=-1).item()
                            confidence = probs.max().item()
                        
                        # Store detection if profanity detected (not 'none')
                        if prediction != 0:  # 0 is 'none'
                            detection = {
                                'file': os.path.basename(audio_file),
                                'start_time': window_start,
                                'end_time': window_end,
                                'predicted_class': CLASS_NAMES[prediction],
                                'predicted_id': prediction,
                                'confidence': confidence,
                                'all_probabilities': probs.squeeze().cpu().numpy().tolist()
                            }
                            file_detections.append(detection)
                            all_detections.append(detection)
                
                print(f"Found {len(file_detections)} potential profanity windows in {os.path.basename(audio_file)}")
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        # Save results
        results = {
            'total_files_scanned': len(audio_files),
            'total_detections': len(all_detections),
            'detections_by_file': {},
            'detections_by_class': {},
            'all_detections': all_detections
        }
        
        # Group by file
        for detection in all_detections:
            file_name = detection['file']
            if file_name not in results['detections_by_file']:
                results['detections_by_file'][file_name] = []
            results['detections_by_file'][file_name].append(detection)
        
        # Group by class
        for detection in all_detections:
            class_name = detection['predicted_class']
            if class_name not in results['detections_by_class']:
                results['detections_by_class'][class_name] = []
            results['detections_by_class'][class_name].append(detection)
        
        # Save results
        self._save_full_audio_results(results, save_prefix)
        
        return results
    
    def _save_full_audio_results(self, results, prefix):
        """Save full audio scanning results."""
        save_dir = os.path.join(self.model_dir, 'full_audio_scan_results')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save JSON results
        with open(os.path.join(save_dir, f'{prefix}_scan_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV of all detections
        if results['all_detections']:
            detections_df = pd.DataFrame(results['all_detections'])
            detections_df.to_csv(os.path.join(save_dir, f'{prefix}_detections.csv'), index=False)
        
        # Generate summary report
        with open(os.path.join(save_dir, f'{prefix}_summary.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Full Audio Scan Summary\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Files scanned: {results['total_files_scanned']}\n")
            f.write(f"Total detections: {results['total_detections']}\n\n")
            
            f.write(f"Detections by class:\n")
            for class_name, detections in results['detections_by_class'].items():
                f.write(f"  {class_name}: {len(detections)} detections\n")
            
            f.write(f"\nDetections by file:\n")
            for file_name, detections in results['detections_by_file'].items():
                f.write(f"  {file_name}: {len(detections)} detections\n")
        
        print(f"Full audio scan results saved to {save_dir}")

    def scan_audio_directory(self, audio_dir, audio_extensions=None):
        """Scan all audio files in a directory for profanities."""
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(audio_dir, f"**/*{ext}"), recursive=True))
        
        print(f"Found {len(audio_files)} audio files to scan")
        
        return self.evaluate_full_audio_files(audio_files)

def main():
    """Main evaluation function."""
    # Configuration
    MODEL_DIR = './models/simplified_advanced_audio_train'
    TEST_CSV = './csv/eval_windowed.csv'  # You might want to use a separate test set
    
    # Load test data (using a subset for demo - in practice use separate test set)
    df = pd.read_csv(TEST_CSV)
    test_df = df.sample(n=min(200, len(df)), random_state=42)  # Sample for quick evaluation
    
    print(f"Evaluating on {len(test_df)} samples")
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(MODEL_DIR)
    
    # Evaluate single best model
    print("\n" + "="*50)
    print("SINGLE MODEL EVALUATION")
    print("="*50)
    
    best_model = evaluator.load_model(1, 'best')  # Load best model from fold 1
    if best_model is not None:
        single_results = evaluator.evaluate_single_model(best_model, test_df)
        print(f"Single model accuracy: {single_results['accuracy']:.4f}")
    
    # Evaluate ensemble
    print("\n" + "="*50)
    print("ENSEMBLE EVALUATION")
    print("="*50)
    
    ensemble_results = evaluator.evaluate_ensemble(test_df, num_folds=5)
    if ensemble_results is not None:
        print(f"Ensemble accuracy: {ensemble_results['accuracy']:.4f}")
        print(f"Ensemble size: {ensemble_results['ensemble_size']}")
    
    # Model comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    model_configs = {
        'fold_1_best': {'type': 'single', 'fold': 1, 'stage': 'best'},
        'fold_2_best': {'type': 'single', 'fold': 2, 'stage': 'best'},
        'ensemble_all': {'type': 'ensemble', 'num_folds': 5},
        'ensemble_3': {'type': 'ensemble', 'num_folds': 3}
    }
    
    comparison_results = evaluator.compare_models(test_df, model_configs)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for model_name, results in comparison_results.items():
        print(f"{model_name}:")
        print(f"  Overall Accuracy: {results['accuracy']:.4f}")
        print(f"  Profanity Accuracy: {results['profanity_accuracy']:.4f}")
        
        if results['uncertainty_metrics']:
            print(f"  Mean Uncertainty: {results['uncertainty_metrics']['mean_uncertainty']:.4f}")
        
        print(f"  Total Errors: {results['error_analysis']['total_errors']}")
        print()

    # Demonstrate full audio file scanning
    print("\n" + "="*50)
    print("FULL AUDIO FILE SCANNING DEMO")
    print("="*50)
    
    # Example: Scan some audio files for all profanities
    eval_dir = './eval'
    if os.path.exists(eval_dir):
        # Get a few example files
        audio_files = []
        for ext in ['.wav', '.mp3']:
            import glob
            found_files = glob.glob(os.path.join(eval_dir, f"*{ext}"))
            audio_files.extend(found_files[:3])  # Take first 3 files
        
        if audio_files:
            print(f"Scanning {len(audio_files)} example audio files for all profanities...")
            scan_results = evaluator.evaluate_full_audio_files(
                audio_files, save_prefix="demo_full_scan"
            )
            
            print(f"Full audio scan results:")
            print(f"  Files scanned: {scan_results['total_files_scanned']}")
            print(f"  Total detections: {scan_results['total_detections']}")
            
            if scan_results['detections_by_class']:
                print(f"  Detections by class:")
                for class_name, detections in scan_results['detections_by_class'].items():
                    print(f"    {class_name}: {len(detections)} detections")
        else:
            print("No audio files found in ./eval directory for demo")
    else:
        print("./eval directory not found - skipping full audio scan demo")
        print("You can use evaluator.scan_audio_directory('./your_audio_dir') to scan audio files")

if __name__ == "__main__":
    main()

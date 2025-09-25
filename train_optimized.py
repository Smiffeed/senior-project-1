#!/usr/bin/env python3
"""
🚀 OPTIMIZED MULTI-STAGE TRAINING
High-performance training with aggressive optimizations for RTX 5070

Performance improvements:
- Pre-computed audio features (no real-time loading)
- Optimized batch sizes and data loading
- Memory-efficient processing
- Reduced I/O operations
"""

import os
import sys
# Disable wandb logging completely
os.environ["WANDB_DISABLED"] = "true"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import pandas as pd
import librosa
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
from transformers import (
    Wav2Vec2ForSequenceClassification, 
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Config,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
)
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class FastDataCollator:
    """Fast data collator for Windows compatibility"""
    def __call__(self, features):
        batch = {}
        batch['input_values'] = torch.stack([f['input_values'] for f in features])
        batch['labels'] = torch.stack([f['labels'] for f in features])
        return batch

class CustomTrainer(Trainer):
    """Enhanced trainer with class weights and focal loss"""
    def __init__(self, class_weights=None, use_focal_loss=False, gamma=3.0, alpha=0.25, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        self.use_focal_loss = use_focal_loss
        self.gamma = gamma
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.use_focal_loss:
            # Focal loss implementation
            ce_loss = torch.nn.functional.cross_entropy(logits, labels, weight=self.class_weights, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            loss = focal_loss.mean()
        else:
            # Standard weighted cross entropy
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=self.class_weights)
            
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    """Enhanced metrics computation"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Overall accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Macro F1 (gives equal weight to all classes)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    
    # Weighted F1 (considers class distribution)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Focus on profanity classes (exclude 'none' class which is label 0)
    profanity_mask = labels != 0
    if np.any(profanity_mask):
        profanity_accuracy = accuracy_score(labels[profanity_mask], predictions[profanity_mask])
        profanity_f1 = f1_score(labels[profanity_mask], predictions[profanity_mask], average='macro', zero_division=0)
    else:
        profanity_accuracy = 0.0
        profanity_f1 = 0.0
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "profanity_accuracy": profanity_accuracy,
        "profanity_f1": profanity_f1,
    }

@dataclass
class OptimizedConfig:
    """Optimized configuration for fast training"""
    
    # Data paths
    binary_data_path: str = "csv/fixed_windows_adaptive/balanced_train_2.0s.csv"
    multiclass_data_path: str = "csv/fixed_windows_context_aware/balanced_train_0.3s.csv"
    
    # Model paths
    binary_model_path: str = "models/binary_classifier_fast"
    multiclass_model_path: str = "models/multiclass_classifier_fast"
    
    # Window configurations
    binary_window_size: float = 2.0
    multiclass_window_size: float = 0.3
    
    # OPTIMIZED training parameters for RTX 5070
    binary_epochs: int = 15  # Reduced from 25
    multiclass_epochs: int = 10  # Reduced from 20
    batch_size: int = 16  # Optimized for RTX 5070
    gradient_accumulation_steps: int = 4  # Effective batch size = 64
    
    # Learning rates
    binary_lr: float = 2e-4  # Slightly higher for faster convergence
    multiclass_lr: float = 1e-4
    
    # Performance optimizations (Windows-compatible)
    num_workers: int = 0  # Disable multiprocessing for Windows compatibility
    pin_memory: bool = True
    mixed_precision: bool = True
    dataloader_drop_last: bool = True
    
    # Model parameters
    wav2vec_model: str = "facebook/wav2vec2-base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class PrecomputedDataset(Dataset):
    """Dataset with precomputed features for maximum speed"""
    
    def __init__(self, csv_file: str, window_size: float, stage: str = "multiclass", max_samples: int = None):
        """
        Initialize with precomputed features
        
        Args:
            csv_file: Path to CSV file
            window_size: Window size in seconds
            stage: "binary" or "multiclass"
            max_samples: Limit dataset size for faster training
        """
        self.data = pd.read_csv(csv_file)
        
        # Limit dataset size for faster training if specified
        if max_samples and len(self.data) > max_samples:
            self.data = self.data.sample(n=max_samples, random_state=42).reset_index(drop=True)
            print(f"⚡ Limited dataset to {max_samples} samples for faster training")
        
        self.window_size = window_size
        self.stage = stage
        
        # Label mappings
        if stage == "binary":
            self.label_map = {
                'none': 0,
                'เย็ด': 1, 'กู': 1, 'มึง': 1, 'เหี้ย': 1
            }
            self.num_classes = 2
        else:
            self.label_map = {
                'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4
            }
            self.num_classes = 5
        
        # Feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base"
        )
        
        print(f"⚡ Dataset loaded: {len(self.data)} samples for {stage} training")
        print(f"Label distribution: {self.data['label'].value_counts().to_dict()}")
        
        # Precompute audio features for speed
        self._precompute_features()
    
    def _precompute_features(self):
        """Precompute audio features to avoid I/O during training"""
        print("⚡ Precomputing audio features for maximum training speed...")
        
        self.precomputed_audio = []
        self.labels = []
        
        # Process in batches to manage memory
        batch_size = 100
        processed = 0
        
        for i in range(0, len(self.data), batch_size):
            batch_data = self.data.iloc[i:i+batch_size]
            batch_audio = []
            batch_labels = []
            
            for _, row in batch_data.iterrows():
                try:
                    # Load and process audio
                    audio = self._load_audio_window_fast(
                        row['file_path'], 
                        row['start_time'], 
                        self.window_size
                    )
                    
                    if audio is not None:
                        batch_audio.append(audio)
                        batch_labels.append(self.label_map[row['label']])
                    
                except Exception as e:
                    # Use dummy data for failed loads
                    dummy_audio = np.zeros(int(self.window_size * 16000))
                    batch_audio.append(dummy_audio)
                    batch_labels.append(0)
                
                processed += 1
                if processed % 1000 == 0:
                    print(f"  Processed {processed}/{len(self.data)} samples...")
            
            self.precomputed_audio.extend(batch_audio)
            self.labels.extend(batch_labels)
        
        print(f"✅ Precomputed {len(self.precomputed_audio)} audio features")
    
    def _enhance_audio(self, audio):
        """Enhanced audio preprocessing for better feature extraction"""
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Apply slight noise reduction using spectral gating
        # Compute short-time energy
        frame_length = 512
        hop_length = 256
        energy = np.array([
            np.sum(audio[i:i+frame_length]**2) 
            for i in range(0, len(audio)-frame_length, hop_length)
        ])
        
        # Noise gate threshold (remove very quiet sections)
        threshold = np.percentile(energy, 10)  # Bottom 10% considered noise
        
        # Create mask for non-noise sections
        mask = energy > threshold
        
        # Expand mask to original audio length
        expanded_mask = np.repeat(mask, hop_length)
        if len(expanded_mask) > len(audio):
            expanded_mask = expanded_mask[:len(audio)]
        elif len(expanded_mask) < len(audio):
            expanded_mask = np.pad(expanded_mask, (0, len(audio) - len(expanded_mask)), mode='edge')
        
        # Apply noise gate
        audio = audio * expanded_mask
        
        return audio
    
    def _load_audio_window_fast(self, file_path: str, start_time: float, window_size: float):
        """Fast audio loading with enhanced preprocessing"""
        try:
            # Load only the needed segment
            audio, sr = librosa.load(
                file_path, 
                sr=16000, 
                offset=start_time,
                duration=window_size
            )
            
            # Apply enhanced audio preprocessing
            audio = self._enhance_audio(audio)
            
            # Ensure exact size
            target_samples = int(window_size * 16000)
            if len(audio) < target_samples:
                audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
            elif len(audio) > target_samples:
                audio = audio[:target_samples]
            
            return audio
            
        except Exception:
            return None
    
    def __len__(self):
        return len(self.precomputed_audio)
    
    def __getitem__(self, idx):
        # Return precomputed features
        audio = self.precomputed_audio[idx]
        label = self.labels[idx]
        
        # Process with feature extractor
        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        return {
            'input_values': inputs.input_values.squeeze(),
            'labels': torch.tensor(label, dtype=torch.long),
        }

class OptimizedTrainer:
    """Optimized trainer for maximum speed on RTX 5070"""
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        os.makedirs("models", exist_ok=True)
        
        print(f"⚡ Optimized Trainer initialized for RTX 5070")
        print(f"Device: {self.device}")
        print(f"Batch size: {config.batch_size} (effective: {config.batch_size * config.gradient_accumulation_steps})")
        
    def compute_class_weights(self, labels, num_classes):
        """Compute class weights for imbalanced datasets"""
        unique_labels = np.unique(labels)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_labels,
            y=labels
        )
        
        # Create tensor with weights matching the actual number of classes
        weight_tensor = torch.ones(num_classes)
        for i, weight in zip(unique_labels, class_weights):
            if i < num_classes:  # Ensure we don't exceed tensor size
                weight_tensor[i] = weight
            
        # Label mappings for display
        if num_classes == 2:
            id2label = {0: 'none', 1: 'profanity'}
        else:
            id2label = {0: 'none', 1: 'เย็ด', 2: 'กู', 3: 'มึง', 4: 'เหี้ย'}
            
        print(f"Class weights computed for {num_classes} classes:")
        for i, weight in enumerate(weight_tensor):
            if i < len(id2label):
                print(f"  {id2label[i]}: {weight:.3f}")
        
        return weight_tensor
    
    def create_fast_model(self, num_classes: int, pretrained_path: Optional[str] = None):
        """Create optimized model"""
        if pretrained_path and os.path.exists(pretrained_path):
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                pretrained_path,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.config.wav2vec_model,
                num_labels=num_classes
            )
        
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        
        return model.to(self.device)
    
    def train_stage_fast(self, stage: str, max_samples: int = None):
        """Fast training with optimizations"""
        
        print(f"\n⚡ FAST {stage.upper()} training")
        
        # Configuration
        if stage == "binary":
            data_path = self.config.binary_data_path
            window_size = self.config.binary_window_size
            epochs = self.config.binary_epochs
            lr = self.config.binary_lr
            num_classes = 2
            model_path = self.config.binary_model_path
            pretrained_path = None
        else:
            data_path = self.config.multiclass_data_path
            window_size = self.config.multiclass_window_size
            epochs = self.config.multiclass_epochs
            lr = self.config.multiclass_lr
            num_classes = 5
            model_path = self.config.multiclass_model_path
            pretrained_path = self.config.binary_model_path
        
        # Create optimized dataset
        dataset = PrecomputedDataset(
            data_path, 
            window_size, 
            stage, 
            max_samples=max_samples
        )
        
        # Create optimized dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.dataloader_drop_last
        )
        
        # Create model
        model = self.create_fast_model(num_classes, pretrained_path)
        
        # Compute class weights for imbalanced datasets
        class_weights = self.compute_class_weights(dataset.labels, num_classes)
        
        # Optimized training arguments
        training_args = TrainingArguments(
            output_dir=f"./models/{stage}_training_fast",
            num_train_epochs=epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=lr,
            weight_decay=0.01,
            logging_steps=50,  # Less frequent logging
            save_steps=1000,
            save_total_limit=1,  # Keep only 1 checkpoint
            fp16=self.config.mixed_precision,
            dataloader_pin_memory=self.config.pin_memory,
            dataloader_num_workers=self.config.num_workers,
            remove_unused_columns=False,
            report_to=[],  # Disable wandb for speed (empty list)
            warmup_steps=100,
            lr_scheduler_type="cosine",
            eval_strategy="steps" if stage == "multiclass" else "no",
            eval_steps=500 if stage == "multiclass" else None,
            load_best_model_at_end=True if stage == "multiclass" else False,
            metric_for_best_model="f1_macro" if stage == "multiclass" else None,
        )
        
        # Data collator (moved outside to avoid pickle issues)
        data_collator = FastDataCollator()
        
        # Create enhanced trainer with class weights and focal loss
        use_focal_loss = stage == "multiclass"  # Use focal loss for multiclass only
        trainer = CustomTrainer(
            class_weights=class_weights,
            use_focal_loss=use_focal_loss,
            gamma=2.0,  # Focal loss gamma parameter
            alpha=0.25,  # Focal loss alpha parameter
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
            eval_dataset=dataset if stage == "multiclass" else None,  # Use same dataset for eval in multiclass
            compute_metrics=compute_metrics if stage == "multiclass" else None,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if stage == "multiclass" else None,
        )
        
        # Train
        print(f"🚀 Training {stage} model for {epochs} epochs (FAST MODE)...")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        trainer.train()
        end_time.record()
        
        torch.cuda.synchronize()
        training_time = start_time.elapsed_time(end_time) / 1000 / 60  # Convert to minutes
        
        print(f"✅ {stage.capitalize()} training completed in {training_time:.1f} minutes")
        
        # Save model
        model.save_pretrained(model_path)
        print(f"💾 Model saved to {model_path}")
        
        return model
    
    def cross_validate_model(self, stage: str, k_folds: int = 3, max_samples: int = None):
        """Perform k-fold cross validation for model robustness"""
        print(f"\n🔄 {k_folds}-fold Cross Validation for {stage} model")
        
        # Configuration
        if stage == "binary":
            data_path = self.config.binary_data_path
            window_size = self.config.binary_window_size
            epochs = max(1, self.config.binary_epochs // 2)  # Reduce epochs for CV
            lr = self.config.binary_lr
            num_classes = 2
        else:
            data_path = self.config.multiclass_data_path
            window_size = self.config.multiclass_window_size
            epochs = max(1, self.config.multiclass_epochs // 2)  # Reduce epochs for CV
            lr = self.config.multiclass_lr
            num_classes = 5
        
        # Load and prepare data
        full_dataset = PrecomputedDataset(
            data_path, 
            window_size, 
            stage, 
            max_samples=max_samples
        )
        
        # Perform k-fold CV
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_dataset)))):
            print(f"\n📊 Fold {fold + 1}/{k_folds}")
            
            # Create fold datasets
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)
            
            # Create model for this fold
            model = self.create_fast_model(num_classes)
            
            # Compute class weights for training set
            train_labels = [full_dataset.labels[i] for i in train_idx]
            class_weights = self.compute_class_weights(train_labels, num_classes)
            
            # Training arguments for CV
            training_args = TrainingArguments(
                output_dir=f"./models/{stage}_cv_fold_{fold}",
                num_train_epochs=epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=lr,
                weight_decay=0.01,
                logging_steps=100,
                eval_strategy="epoch",
                save_strategy="no",  # Don't save CV models
                fp16=self.config.mixed_precision,
                dataloader_pin_memory=self.config.pin_memory,
                dataloader_num_workers=self.config.num_workers,
                remove_unused_columns=False,
                report_to=[],
            )
            
            # Create trainer for this fold
            trainer = CustomTrainer(
                class_weights=class_weights,
                use_focal_loss=(stage == "multiclass"),
                gamma=2.0,
                alpha=0.25,
                model=model,
                args=training_args,
                data_collator=FastDataCollator(),
                train_dataset=train_subset,
                eval_dataset=val_subset,
                compute_metrics=compute_metrics,
            )
            
            # Train and evaluate
            trainer.train()
            eval_results = trainer.evaluate()
            
            cv_scores.append({
                'fold': fold + 1,
                'accuracy': eval_results.get('eval_accuracy', 0),
                'f1_macro': eval_results.get('eval_f1_macro', 0),
                'f1_weighted': eval_results.get('eval_f1_weighted', 0),
                'profanity_f1': eval_results.get('eval_profanity_f1', 0),
            })
            
            print(f"Fold {fold + 1} Results:")
            print(f"  Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
            print(f"  F1 Macro: {eval_results.get('eval_f1_macro', 0):.4f}")
            print(f"  F1 Weighted: {eval_results.get('eval_f1_weighted', 0):.4f}")
            print(f"  Profanity F1: {eval_results.get('eval_profanity_f1', 0):.4f}")
        
        # Calculate cross-validation statistics
        cv_df = pd.DataFrame(cv_scores)
        mean_scores = cv_df.mean()
        std_scores = cv_df.std()
        
        print(f"\n🎯 Cross-Validation Results Summary ({k_folds} folds):")
        print(f"Accuracy: {mean_scores['accuracy']:.4f} ± {std_scores['accuracy']:.4f}")
        print(f"F1 Macro: {mean_scores['f1_macro']:.4f} ± {std_scores['f1_macro']:.4f}")
        print(f"F1 Weighted: {mean_scores['f1_weighted']:.4f} ± {std_scores['f1_weighted']:.4f}")
        print(f"Profanity F1: {mean_scores['profanity_f1']:.4f} ± {std_scores['profanity_f1']:.4f}")
        
        return cv_scores, mean_scores, std_scores

    def train_fast_pipeline(self):
        """Execute fast multi-stage training"""
        
        print("🚀 FAST MULTI-STAGE TRAINING PIPELINE")
        print("=" * 60)
        print("Optimizations enabled:")
        print("✅ Precomputed audio features")
        print("✅ Optimized batch sizes")
        print("✅ Reduced epochs with higher learning rates")
        print("✅ Memory-efficient processing")
        print("✅ Disabled wandb logging")
        print("=" * 60)
        
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record()
        
        # Stage 1: Binary (limit to 10k samples for speed)
        print("\n📊 STAGE 1: Fast Binary Classification")
        binary_model = self.train_stage_fast("binary", max_samples=10000)
        
        # Stage 2: Multiclass (use full dataset - it's smaller)
        print("\n📊 STAGE 2: Fast Multiclass Fine-tuning")
        multiclass_model = self.train_stage_fast("multiclass")
        
        total_end.record()
        torch.cuda.synchronize()
        total_time = total_start.elapsed_time(total_end) / 1000 / 60
        
        print(f"\n🎉 FAST TRAINING COMPLETED!")
        print(f"⏱️  Total time: {total_time:.1f} minutes")
        print(f"🎯 Expected performance: Binary F1 ~0.82, Multiclass F1 ~0.70")
        
        return multiclass_model
    
    def train_full_pipeline(self):
        """Execute full multi-stage training with complete datasets"""
        
        print("🔥 FULL MULTI-STAGE TRAINING PIPELINE")
        print("=" * 60)
        print("Full training configuration:")
        print(f"✅ Binary epochs: {self.config.binary_epochs}")
        print(f"✅ Multiclass epochs: {self.config.multiclass_epochs}")
        print("✅ Complete datasets (no sample limits)")
        print("✅ All optimizations still enabled")
        print("=" * 60)
        
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record()
        
        # Stage 1: Binary (full dataset)
        print("\n📊 STAGE 1: Full Binary Classification")
        binary_model = self.train_stage_fast("binary", max_samples=None)
        
        # Stage 2: Multiclass (full dataset)
        print("\n📊 STAGE 2: Full Multiclass Fine-tuning")
        multiclass_model = self.train_stage_fast("multiclass", max_samples=None)
        
        total_end.record()
        torch.cuda.synchronize()
        total_time = total_start.elapsed_time(total_end) / 1000 / 60
        
        print(f"\n🎉 FULL TRAINING COMPLETED!")
        print(f"⏱️  Total time: {total_time:.1f} minutes")
        print(f"🎯 Expected performance: Binary F1 ~0.85+, Multiclass F1 ~0.75+")
        
        return multiclass_model

def main():
    """Main function with speed options"""
    parser = argparse.ArgumentParser(description="Optimized training for RTX 5070")
    parser.add_argument("--mode", choices=["fast", "full", "cv"], default="fast",
                       help="Training mode: fast (optimized), full (original), or cv (cross-validation)")
    parser.add_argument("--stage", choices=["binary", "multiclass", "both"], default="both")
    parser.add_argument("--max_samples", type=int, help="Limit training samples for speed")
    parser.add_argument("--cv_folds", type=int, default=3, help="Number of folds for cross-validation")
    
    args = parser.parse_args()
    
    config = OptimizedConfig()
    
    # Adjust config for full mode
    if args.mode == "full":
        print("🔄 FULL TRAINING MODE")
        print("Using complete datasets with original epoch counts...")
        config.binary_epochs = 25  # Original epochs
        config.multiclass_epochs = 20  # Original epochs
        config.binary_model_path = "models/binary_classifier_full"
        config.multiclass_model_path = "models/multiclass_classifier_full"
    
    trainer = OptimizedTrainer(config)
    
    if args.mode == "fast":
        if args.stage == "both":
            trainer.train_fast_pipeline()
        elif args.stage == "binary":
            trainer.train_stage_fast("binary", args.max_samples)
        elif args.stage == "multiclass":
            trainer.train_stage_fast("multiclass", args.max_samples)
    
    elif args.mode == "full":
        if args.stage == "both":
            trainer.train_full_pipeline()
        elif args.stage == "binary":
            trainer.train_stage_fast("binary", None)  # No sample limit
        elif args.stage == "multiclass":
            trainer.train_stage_fast("multiclass", None)  # No sample limit
    
    elif args.mode == "cv":
        print("🔄 CROSS-VALIDATION MODE")
        if args.stage == "both":
            print("Running cross-validation for both stages...")
            trainer.cross_validate_model("binary", args.cv_folds, args.max_samples)
            trainer.cross_validate_model("multiclass", args.cv_folds, args.max_samples)
        elif args.stage == "binary":
            trainer.cross_validate_model("binary", args.cv_folds, args.max_samples)
        elif args.stage == "multiclass":
            trainer.cross_validate_model("multiclass", args.cv_folds, args.max_samples)
    
    print("\n🎯 NEXT STEPS:")
    if args.mode == "fast":
        print("1. Test your model:")
        print("   python evaluate_advanced_models.py --model_path models/multiclass_classifier_fast")
    elif args.mode == "full":
        print("1. Test your model:")
        print("   python evaluate_advanced_models.py --model_path models/multiclass_classifier_full")
    elif args.mode == "cv":
        print("1. Cross-validation completed. Consider training with best parameters.")
        print("2. Use --mode fast or --mode full to train final model")
    print("2. Compare performance vs baseline")
    print("3. For detailed analysis, run comprehensive evaluation scripts")

if __name__ == "__main__":
    main()
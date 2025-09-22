#!/usr/bin/env python3
"""
🎯 MULTI-STAGE TRAINING SYSTEM
Implements optimal training strategy using fixed-window datasets.

Based on evaluation results:
- Stage 1: Binary classification with 2.0s windows (optimal F1)
- Stage 2: Multiclass fine-tuning with 0.3s windows (optimal class separation)

Usage:
    python train_multi_stage.py
    python train_multi_stage.py --stage binary
    python train_multi_stage.py --stage multiclass
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchaudio
import numpy as np
import pandas as pd
import librosa
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    Wav2Vec2ForSequenceClassification, 
    Wav2Vec2FeatureExtractor,
    Trainer, 
    TrainingArguments,
    TrainerCallback
)
import warnings
warnings.filterwarnings('ignore')

# Configuration
@dataclass
class MultiStageConfig:
    """Configuration for multi-stage training"""
    
    # Data paths
    binary_data_path: str = "csv/fixed_windows_adaptive/balanced_train_2.0s.csv"
    multiclass_data_path: str = "csv/fixed_windows_context_aware/balanced_train_0.3s.csv"
    eval_data_path: str = "csv/eval_5labels.csv"
    
    # Model paths
    binary_model_path: str = "models/binary_classifier.pt"
    multiclass_model_path: str = "models/multiclass_classifier.pt"
    final_model_path: str = "models/multi_stage_final.pt"
    
    # Window configurations (based on evaluation results)
    binary_window_size: float = 2.0
    multiclass_window_size: float = 0.3
    
    # Training parameters
    binary_epochs: int = 25
    multiclass_epochs: int = 20
    binary_lr: float = 1e-4
    multiclass_lr: float = 5e-5  # Lower for fine-tuning
    batch_size: int = 32
    
    # Model parameters
    wav2vec_model: str = "facebook/wav2vec2-base"
    dropout_rate: float = 0.1
    hidden_size: int = 768
    
    # Device and optimization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 2

class FixedWindowDataset(Dataset):
    """Dataset for fixed-window training"""
    
    def __init__(self, csv_file: str, window_size: float, stage: str = "multiclass"):
        """
        Initialize dataset
        
        Args:
            csv_file: Path to CSV file with fixed windows
            window_size: Window size in seconds
            stage: "binary" or "multiclass"
        """
        self.data = pd.read_csv(csv_file)
        self.window_size = window_size
        self.stage = stage
        
        # Label mappings
        if stage == "binary":
            # Binary: profanity vs none
            self.label_map = {
                'none': 0,
                'เย็ด': 1,
                'กู': 1,
                'มึง': 1,
                'เหี้ย': 1
            }
            self.num_classes = 2
            self.class_names = ['none', 'profanity']
        else:
            # Multiclass: all 5 classes
            self.label_map = {
                'none': 0,
                'เย็ด': 1,
                'กู': 2,
                'มึง': 3,
                'เหี้ย': 4
            }
            self.num_classes = 5
            self.class_names = ['none', 'เย็ด', 'กู', 'มึง', 'เหี้ย']
        
        # Feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base"
        )
        
        print(f"Loaded {len(self.data)} samples for {stage} training")
        print(f"Label distribution: {self.data['label'].value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        try:
            # Load audio window
            audio = self._load_audio_window(
                row['file_path'], 
                row['start_time'], 
                self.window_size
            )
            
            # Process with feature extractor
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            )
            
            # Map label
            label = self.label_map[row['label']]
            
            return {
                'input_values': inputs.input_values.squeeze(),
                'labels': torch.tensor(label, dtype=torch.long),
                'profanity_coverage': row.get('profanity_coverage', 0.0)
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy sample
            dummy_audio = np.zeros(int(self.window_size * 16000))
            inputs = self.feature_extractor(
                dummy_audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            return {
                'input_values': inputs.input_values.squeeze(),
                'labels': torch.tensor(0, dtype=torch.long),
                'profanity_coverage': 0.0
            }
    
    def _load_audio_window(self, file_path: str, start_time: float, window_size: float):
        """Load audio window with exact timing"""
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=16000)
            
            # Extract window
            start_sample = int(start_time * sr)
            end_sample = int((start_time + window_size) * sr)
            
            # Handle bounds
            if start_sample >= len(audio):
                start_sample = max(0, len(audio) - int(window_size * sr))
            
            if end_sample > len(audio):
                end_sample = len(audio)
            
            window_audio = audio[start_sample:end_sample]
            
            # Ensure exact size
            target_samples = int(window_size * sr)
            if len(window_audio) < target_samples:
                # Pad if too short
                padding = target_samples - len(window_audio)
                window_audio = np.pad(window_audio, (0, padding), mode='constant')
            elif len(window_audio) > target_samples:
                # Trim if too long
                window_audio = window_audio[:target_samples]
            
            return window_audio
            
        except Exception as e:
            print(f"Error loading audio from {file_path}: {e}")
            # Return silence as fallback
            return np.zeros(int(window_size * 16000))

class MultiStageTrainer:
    """Multi-stage training system"""
    
    def __init__(self, config: MultiStageConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create model directory
        os.makedirs("models", exist_ok=True)
        
        print(f"🚀 Multi-Stage Trainer initialized")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {config.mixed_precision}")
    
    def setup_data_paths(self):
        """Set up training data paths"""
        # Copy datasets to expected locations
        binary_source = self.config.binary_data_path
        multiclass_source = self.config.multiclass_data_path
        
        if os.path.exists(binary_source) and os.path.exists(multiclass_source):
            # Copy to expected locations
            os.makedirs("csv", exist_ok=True)
            
            import shutil
            shutil.copy(binary_source, "csv/train_binary.csv")
            shutil.copy(multiclass_source, "csv/train_multiclass.csv")
            
            print("✅ Training data paths set up successfully")
            return "csv/train_binary.csv", "csv/train_multiclass.csv"
        else:
            raise FileNotFoundError(f"Dataset files not found: {binary_source}, {multiclass_source}")
    
    def create_model(self, num_classes: int, pretrained_model_path: Optional[str] = None):
        """Create Wav2Vec2 model for classification"""
        
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"Loading pretrained model from {pretrained_model_path}")
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                pretrained_model_path,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            print(f"Creating new model with {num_classes} classes")
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.config.wav2vec_model,
                num_labels=num_classes
            )
        
        return model.to(self.device)
    
    def train_stage(self, stage: str, model_save_path: str, pretrained_model_path: Optional[str] = None):
        """Train a single stage"""
        
        print(f"\n🎯 Starting {stage.upper()} training stage")
        
        # Set up data
        if stage == "binary":
            data_path = "csv/train_binary.csv"
            window_size = self.config.binary_window_size
            epochs = self.config.binary_epochs
            learning_rate = self.config.binary_lr
            num_classes = 2
        else:  # multiclass
            data_path = "csv/train_multiclass.csv"
            window_size = self.config.multiclass_window_size
            epochs = self.config.multiclass_epochs
            learning_rate = self.config.multiclass_lr
            num_classes = 5
        
        # Create dataset
        dataset = FixedWindowDataset(data_path, window_size, stage)
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Create model
        model = self.create_model(num_classes, pretrained_model_path)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=f"./models/{stage}_training",
            num_train_epochs=epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"./logs/{stage}",
            logging_steps=100,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=False,
            fp16=self.config.mixed_precision,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
        )
        
        # Custom data collator
        def data_collator(features):
            batch = {}
            batch['input_values'] = torch.stack([f['input_values'] for f in features])
            batch['labels'] = torch.stack([f['labels'] for f in features])
            return batch
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        
        # Train
        print(f"Training {stage} model for {epochs} epochs...")
        trainer.train()
        
        # Save model
        model.save_pretrained(model_save_path)
        print(f"✅ {stage.capitalize()} model saved to {model_save_path}")
        
        return model
    
    def train_multi_stage(self):
        """Execute complete multi-stage training"""
        
        print("🚀 Starting Multi-Stage Training Pipeline")
        print("=" * 50)
        
        # Set up data paths
        binary_path, multiclass_path = self.setup_data_paths()
        
        # Stage 1: Binary Classification
        print("\n📊 STAGE 1: Binary Classification (profanity vs none)")
        print(f"Dataset: {binary_path}")
        print(f"Window size: {self.config.binary_window_size}s")
        print(f"Expected improvement: +13% F1 score")
        
        binary_model = self.train_stage(
            stage="binary",
            model_save_path=self.config.binary_model_path
        )
        
        # Stage 2: Multiclass Fine-tuning
        print("\n📊 STAGE 2: Multiclass Fine-tuning (4 profanity classes)")
        print(f"Dataset: {multiclass_path}")
        print(f"Window size: {self.config.multiclass_window_size}s")
        print(f"Expected improvement: +15% multiclass F1, +33% IoU")
        
        multiclass_model = self.train_stage(
            stage="multiclass",
            model_save_path=self.config.multiclass_model_path,
            pretrained_model_path=self.config.binary_model_path
        )
        
        # Save final model
        multiclass_model.save_pretrained(self.config.final_model_path)
        print(f"\n✅ Multi-stage training completed!")
        print(f"Final model saved to: {self.config.final_model_path}")
        
        # Training summary
        print("\n" + "=" * 50)
        print("🎯 TRAINING SUMMARY")
        print("=" * 50)
        print(f"Stage 1 (Binary): {self.config.binary_epochs} epochs, {self.config.binary_window_size}s windows")
        print(f"Stage 2 (Multiclass): {self.config.multiclass_epochs} epochs, {self.config.multiclass_window_size}s windows")
        print(f"Total training time: Estimated 2-4 hours on RTX 5070")
        
        print("\n🎉 NEXT STEPS:")
        print("1. Evaluate model performance:")
        print("   python evaluate_advanced_models.py --model_path models/multi_stage_final")
        print("2. Run comprehensive evaluation:")
        print("   python comprehensive_evaluation_analysis.py")
        print("3. Compare with baseline results")
        
        return multiclass_model

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Multi-stage training for Thai profanity detection")
    parser.add_argument("--stage", choices=["binary", "multiclass", "both"], default="both",
                       help="Training stage to run")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--binary_epochs", type=int, default=25, help="Binary training epochs")
    parser.add_argument("--multiclass_epochs", type=int, default=20, help="Multiclass training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate (will use defaults if not specified)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = MultiStageConfig()
    
    # Override with command line arguments
    if args.binary_epochs:
        config.binary_epochs = args.binary_epochs
    if args.multiclass_epochs:
        config.multiclass_epochs = args.multiclass_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.binary_lr = args.learning_rate
        config.multiclass_lr = args.learning_rate * 0.5  # Half for fine-tuning
    
    # Create trainer
    trainer = MultiStageTrainer(config)
    
    try:
        if args.stage == "both":
            # Full multi-stage training
            trainer.train_multi_stage()
        elif args.stage == "binary":
            # Binary only
            trainer.setup_data_paths()
            trainer.train_stage("binary", config.binary_model_path)
        elif args.stage == "multiclass":
            # Multiclass only (requires binary model)
            trainer.setup_data_paths()
            if not os.path.exists(config.binary_model_path):
                print("❌ Binary model not found. Run binary stage first or use --stage both")
                return
            trainer.train_stage("multiclass", config.multiclass_model_path, config.binary_model_path)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("🔧 Troubleshooting tips:")
        print("1. Check that dataset files exist:")
        print(f"   - {config.binary_data_path}")
        print(f"   - {config.multiclass_data_path}")
        print("2. Ensure CUDA is available for GPU training")
        print("3. Check disk space (models require ~2GB)")
        print("4. Try reducing batch size if OOM errors occur")
        
        raise

if __name__ == "__main__":
    main()
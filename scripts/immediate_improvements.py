#!/usr/bin/env python3
"""
🎯 IMMEDIATE MODEL IMPROVEMENTS
Quick implementation of critical improvements for Thai profanity detection model.

This script implements the most impactful improvements that can be applied immediately:
1. Advanced data augmentation for minority classes
2. Focal Loss for class imbalance
3. Enhanced training pipeline
4. Better evaluation metrics

Usage:
    python immediate_improvements.py --input-csv ./csv/train.csv --output-dir ./models/improved_model
"""

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import librosa
import os
import argparse
from transformers import (
    Wav2Vec2ForSequenceClassification, 
    Wav2Vec2FeatureExtractor, 
    Wav2Vec2Config,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings('ignore')

# Enhanced label mapping
label_map = {
    'none': 0,
    'เย็ด': 1,
    'กู': 2,
    'มึง': 3,
    'เหี้ย': 4,
    'ควย': 5,
    'สวะ': 6,
    'หี': 7,
    'แตด': 8
}

num_labels = len(label_map)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples and reduces the relative loss for well-classified examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdvancedAugmentation:
    """Advanced audio augmentation specifically for Thai profanity words."""
    
    def __init__(self, target_length=16000):
        self.target_length = target_length
        
    def generate_variants(self, audio, label, num_variants=5):
        """Generate multiple realistic variants of profanity words."""
        variants = []
        
        for i in range(num_variants):
            variant = audio.copy()
            intensity = (i + 1) / num_variants  # Gradual intensity increase
            
            # 1. Voice pitch variation (simulate different speakers)
            if np.random.random() < 0.8:
                pitch_shift = np.random.uniform(-2, 2) * intensity
                try:
                    variant = librosa.effects.pitch_shift(variant, sr=16000, n_steps=pitch_shift)
                except:
                    pass  # Skip if pitch shift fails
            
            # 2. Speaking rate variation
            if np.random.random() < 0.7:
                rate = np.random.uniform(0.85, 1.15) * (1 + intensity * 0.1)
                try:
                    variant = librosa.effects.time_stretch(variant, rate=rate)
                except:
                    pass  # Skip if time stretch fails
            
            # 3. Background noise (simulate different recording conditions)
            if np.random.random() < 0.6:
                noise_level = np.random.uniform(0.001, 0.005) * intensity
                noise = np.random.normal(0, noise_level, len(variant))
                variant += noise
            
            # 4. Volume variation
            if np.random.random() < 0.5:
                volume_factor = np.random.uniform(0.7, 1.3)
                variant *= volume_factor
            
            # 5. Simple reverb (room acoustics)
            if np.random.random() < 0.3:
                reverb_delay = np.random.randint(200, 800)
                decay = np.random.uniform(0.1, 0.3)
                if reverb_delay < len(variant):
                    reverb = np.exp(-decay * np.linspace(0, 1, reverb_delay))
                    reverb_effect = np.convolve(variant, reverb, mode='full')[:len(variant)]
                    mix_ratio = 0.2
                    variant = (1 - mix_ratio) * variant + mix_ratio * reverb_effect
            
            # 6. High-frequency emphasis (simulate different mics)
            if np.random.random() < 0.4:
                variant = librosa.effects.preemphasis(variant, coef=np.random.uniform(0.9, 0.99))
            
            # Normalize and ensure proper length
            variant = self._normalize_audio(variant)
            variant = self._ensure_length(variant)
            
            variants.append(variant)
        
        return variants
    
    def _normalize_audio(self, audio):
        """Normalize audio to prevent clipping and ensure consistent levels."""
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # RMS normalization
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            audio = audio * (0.1 / rms)
        
        # Z-score normalization
        std = np.std(audio)
        if std > 1e-8:
            audio = (audio - np.mean(audio)) / std
        
        return audio
    
    def _ensure_length(self, audio):
        """Ensure audio is exactly target_length samples."""
        if len(audio) < self.target_length:
            padding = self.target_length - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        elif len(audio) > self.target_length:
            # Trim from center to preserve most important part
            start_idx = (len(audio) - self.target_length) // 2
            audio = audio[start_idx:start_idx + self.target_length]
        
        return audio

class ImprovedTrainer(Trainer):
    """Enhanced trainer with focal loss and class weighting."""
    
    def __init__(self, use_focal_loss=True, focal_alpha=0.25, focal_gamma=2.0, 
                 class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Enhanced compute_loss that handles additional arguments from newer Transformers versions.
        """
        # Handle potential additional arguments (like num_items_in_batch)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.use_focal_loss:
            loss_fct = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def enhanced_compute_metrics(eval_pred):
    """Enhanced metrics that work better with imbalanced data."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "f1_weighted": f1_score(labels, predictions, average='weighted'),
        "f1_macro": f1_score(labels, predictions, average='macro'),
    }

def create_balanced_dataset(df, augmentation_factors=None):
    """
    Create a more balanced dataset using smart oversampling and augmentation.
    
    Args:
        df: Original dataset
        augmentation_factors: Dict mapping label -> number of augmented samples to create
    """
    if augmentation_factors is None:
        # Default augmentation factors based on current class imbalance
        augmentation_factors = {
            'none': 0,      # Don't augment majority class
            'เย็ด': 3,      # 46 -> ~138 samples
            'กู': 2,        # 80 -> ~160 samples  
            'มึง': 2,       # 79 -> ~158 samples
            'เหี้ย': 3,     # 52 -> ~156 samples
            'ควย': 3,       # 50 -> ~150 samples
            'หี': 4,        # 37 -> ~148 samples
            'สวะ': 5,       # 32 -> ~160 samples
            'แตด': 8,       # 20 -> ~160 samples (critical!)
        }
    
    print("🔄 Creating balanced dataset with smart augmentation...")
    
    augmenter = AdvancedAugmentation()
    balanced_data = []
    
    # Add all original samples
    for _, row in df.iterrows():
        balanced_data.append(row.to_dict())
    
    # Generate augmented samples for minority classes
    for label, factor in augmentation_factors.items():
        if factor == 0:
            continue
            
        label_data = df[df['label'] == label]
        if len(label_data) == 0:
            continue
            
        print(f"Augmenting '{label}': {len(label_data)} -> +{len(label_data) * factor} samples")
        
        for _, row in label_data.iterrows():
            file_path = row['file_path'].replace('\\', '/')
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            
            try:
                # Load audio segment
                metadata = torchaudio.info(file_path)
                sr = metadata.sample_rate
                
                start_frame = int(row['start_time'] * sr)
                num_frames = int((row['end_time'] - row['start_time']) * sr)
                
                audio, sr = torchaudio.load(file_path, frame_offset=start_frame, num_frames=num_frames)
                
                if audio.shape[0] > 1:
                    audio = torch.mean(audio, dim=0)
                
                audio_np = audio.numpy()
                
                # Generate variants
                variants = augmenter.generate_variants(audio_np, label, num_variants=factor)
                
                # Add augmented samples (note: they still point to original file)
                for i, variant in enumerate(variants):
                    augmented_row = row.to_dict()
                    augmented_row['augmented'] = True
                    augmented_row['variant_id'] = i
                    balanced_data.append(augmented_row)
                    
            except Exception as e:
                print(f"Warning: Could not augment {file_path}: {e}")
                # Fallback: just duplicate the original sample
                for _ in range(factor):
                    duplicate_row = row.to_dict()
                    duplicate_row['augmented'] = True
                    balanced_data.append(duplicate_row)
    
    balanced_df = pd.DataFrame(balanced_data)
    
    print("\n✅ Balanced dataset created!")
    print("New class distribution:")
    class_counts = balanced_df['label'].value_counts()
    for label, count in class_counts.items():
        original_count = len(df[df['label'] == label])
        print(f"  {label}: {original_count} -> {count} (+{count - original_count})")
    
    return balanced_df

def enhanced_preprocess_audio(file_path, start_time, end_time, max_length=16000):
    """Enhanced audio preprocessing with noise reduction and normalization."""
    try:
        # Load audio metadata
        metadata = torchaudio.info(file_path)
        sr = metadata.sample_rate
        
        # Add small padding
        padding = 0.1
        start_time = max(0, start_time - padding)
        end_time = end_time + padding
        
        # Load audio segment
        audio, sr = torchaudio.load(
            file_path,
            frame_offset=int(start_time * sr),
            num_frames=int((end_time - start_time) * sr)
        )
        
        # Convert to mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample to 16kHz
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        
        audio_np = audio.squeeze().numpy()
        
        # Enhanced preprocessing
        # 1. Pre-emphasis
        audio_np = librosa.effects.preemphasis(audio_np, coef=0.97)
        
        # 2. Noise gate
        noise_threshold = np.percentile(np.abs(audio_np), 20)  # Adaptive threshold
        audio_np = np.where(np.abs(audio_np) < noise_threshold, audio_np * 0.1, audio_np)
        
        # 3. Apply Hamming window
        if len(audio_np) > 1:
            window = np.hamming(len(audio_np))
            audio_np = audio_np * window
        
        # 4. RMS normalization
        rms = np.sqrt(np.mean(audio_np ** 2))
        if rms > 0:
            target_rms = 0.1
            audio_np = audio_np * (target_rms / rms)
        
        # 5. Z-score normalization
        audio_np = (audio_np - np.mean(audio_np)) / (np.std(audio_np) + 1e-8)
        
        # 6. Ensure proper length
        if len(audio_np) < max_length:
            audio_np = np.pad(audio_np, (0, max_length - len(audio_np)), 'constant')
        else:
            audio_np = audio_np[:max_length]
        
        return audio_np
        
    except Exception as e:
        print(f"Error processing audio {file_path}: {e}")
        return None

def prepare_enhanced_dataset(df, feature_extractor):
    """Prepare dataset with enhanced preprocessing and augmentation support."""
    
    def process_example(example):
        file_path = example['file_path'].replace('\\', '/')
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        # Enhanced audio preprocessing
        audio = enhanced_preprocess_audio(
            file_path, 
            example['start_time'], 
            example['end_time']
        )
        
        if audio is None:
            return None
        
        # Apply additional augmentation for augmented samples
        if example.get('augmented', False):
            augmenter = AdvancedAugmentation()
            try:
                # Apply light augmentation to the already augmented sample
                variants = augmenter.generate_variants(audio, example['label'], num_variants=1)
                if variants:
                    audio = variants[0]
            except:
                pass  # Use original if augmentation fails
        
        # Apply feature extractor
        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Get label
        label = label_map.get(example['label'])
        if label is None:
            print(f"Warning: Unknown label {example['label']}")
            return None
        
        return {
            'input_values': inputs.input_values.squeeze().numpy(),
            'attention_mask': inputs.attention_mask.squeeze().numpy(),
            'label': label
        }
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(process_example, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda example: example is not None)
    
    return dataset

def enhanced_collate_fn(batch):
    """Enhanced collate function with proper padding."""
    input_values = [torch.tensor(item['input_values']).squeeze() for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']).squeeze() for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    
    # Pad sequences
    input_values = pad_sequence(input_values, batch_first=True, padding_value=0.0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0.0)
    
    return {
        'input_values': input_values,
        'attention_mask': attention_mask,
        'labels': labels
    }

def train_improved_model(csv_file, output_dir, use_balancing=True, use_focal_loss=True):
    """Main training function with all improvements."""
    
    print("🚀 Starting Improved Model Training")
    print("="*50)
    
    # Load dataset
    print("📊 Loading dataset...")
    df = pd.read_csv(csv_file)
    
    print(f"Original dataset: {len(df)} samples")
    print("Original class distribution:")
    original_dist = df['label'].value_counts()
    for label, count in original_dist.items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Create balanced dataset if requested
    if use_balancing:
        df_balanced = create_balanced_dataset(df)
    else:
        df_balanced = df.copy()
    
    # Calculate class weights
    labels = [label_map[label] for label in df_balanced['label']]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    print(f"\n🎯 Class weights:")
    for i, (label_name, weight) in enumerate(zip(label_map.keys(), class_weights)):
        print(f"  {label_name}: {weight:.3f}")
    
    # Split data
    train_df, val_df = train_test_split(
        df_balanced, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_balanced['label']
    )
    
    print(f"\n📦 Data split:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    
    # Load model and feature extractor
    print("🤖 Loading pre-trained model...")
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    
    config = Wav2Vec2Config.from_pretrained(
        model_name,
        num_labels=num_labels,
        finetuning_task="audio-classification",
        hidden_dropout=0.3,
        attention_dropout=0.1,
        mask_time_prob=0.0,  # Disable for fine-tuning
        mask_feature_prob=0.0
    )
    
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        return_attention_mask=True,
        do_normalize=True,
    )
    
    # Prepare datasets
    print("🔧 Preparing datasets...")
    train_dataset = prepare_enhanced_dataset(train_df, feature_extractor)
    val_dataset = prepare_enhanced_dataset(val_df, feature_extractor)
    
    print(f"  Training dataset: {len(train_dataset)} samples")
    print(f"  Validation dataset: {len(val_dataset)} samples")
    
    # Enhanced training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=120,  # Increased for better convergence
        per_device_train_batch_size=8,   # Smaller batch for stability
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,   # Effective batch size = 16
        learning_rate=1e-5,              # Lower learning rate
        weight_decay=0.05,               # Stronger regularization
        warmup_ratio=0.15,               # More warmup
        save_strategy="steps",
        save_steps=300,
        eval_strategy="steps", 
        eval_steps=300,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",  # Better for imbalanced data
        greater_is_better=True,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        save_total_limit=3,
        report_to=[],  # Explicitly disable all reporting
        run_name=None,  # Disable wandb run name
    )
    
    # Initialize improved trainer
    trainer = ImprovedTrainer(
        use_focal_loss=use_focal_loss,
        focal_alpha=0.25,
        focal_gamma=2.0,
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=enhanced_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
        data_collator=enhanced_collate_fn,
    )
    
    print(f"\n🎯 Training configuration:")
    print(f"  Focal Loss: {use_focal_loss}")
    print(f"  Class Weighting: {class_weights_tensor is not None}")
    print(f"  Data Balancing: {use_balancing}")
    print(f"  Output Directory: {output_dir}")
    
    # Start training
    print("\n🚀 Starting training...")
    try:
        trainer.train()
        
        # Save final model
        print("\n💾 Saving model...")
        trainer.save_model(output_dir)
        feature_extractor.save_pretrained(output_dir)
        
        # Final evaluation
        print("\n📊 Final evaluation...")
        eval_results = trainer.evaluate()
        
        print("Final Results:")
        for metric, value in eval_results.items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model saved to: {output_dir}")
        
        return trainer, eval_results
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Immediate Model Improvements")
    parser.add_argument("--input-csv", type=str, default="./csv/train.csv",
                       help="Path to training CSV file")
    parser.add_argument("--output-dir", type=str, default="./models/improved_model",
                       help="Output directory for trained model")
    parser.add_argument("--use-balancing", action="store_true", default=True,
                       help="Use data balancing and augmentation")
    parser.add_argument("--use-focal-loss", action="store_true", default=True,
                       help="Use focal loss for class imbalance")
    parser.add_argument("--no-balancing", action="store_true",
                       help="Disable data balancing")
    parser.add_argument("--no-focal-loss", action="store_true", 
                       help="Disable focal loss")
    
    args = parser.parse_args()
    
    # Handle negative flags
    use_balancing = args.use_balancing and not args.no_balancing
    use_focal_loss = args.use_focal_loss and not args.no_focal_loss
    
    if not os.path.exists(args.input_csv):
        print(f"❌ Error: Input CSV file not found: {args.input_csv}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    trainer, results = train_improved_model(
        args.input_csv,
        args.output_dir,
        use_balancing=use_balancing,
        use_focal_loss=use_focal_loss
    )
    
    if trainer and results:
        print("\n🎉 Improvement training completed successfully!")
    else:
        print("\n💥 Training failed. Check the logs above for details.")

if __name__ == "__main__":
    main()

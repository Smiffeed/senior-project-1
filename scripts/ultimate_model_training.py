import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Config, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import librosa
from sklearn.model_selection import KFold, train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import random
from tqdm import tqdm
import json

# Import our advanced techniques - with fallbacks
try:
    from advanced_model_improvements import (
        SpectralAugmentation, AdvancedFeatureExtractor, TransformerAudioClassifier,
        ContrastiveLoss, MetaLearningOptimizer, ActiveLearningSelector,
        ModelEnsemble, AdversarialTraining, AdvancedEvaluator
    )
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced features not available: {e}")
    print("Falling back to basic implementation")
    ADVANCED_FEATURES_AVAILABLE = False
    
    # Create dummy classes to prevent errors
    class SpectralAugmentation:
        def spec_augment(self, x): return x
    class AdvancedFeatureExtractor:
        def __init__(self, sr=16000): pass
        def extract_spectral_features(self, audio): return {}
    class TransformerAudioClassifier:
        def __init__(self, *args, **kwargs): pass
    class ContrastiveLoss:
        def __init__(self, *args, **kwargs): pass
    class AdvancedEvaluator:
        def __init__(self, *args, **kwargs): pass

# =====================================================================================
# Configuration and Constants
# =====================================================================================

# Define label mapping and number of labels
LABEL_MAP = {
    'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4,
    'ควย': 5, 'สวะ': 6, 'หี': 7, 'แตด': 8
}
NUM_LABELS = len(LABEL_MAP)
CLASS_NAMES = list(LABEL_MAP.keys())

# Advanced training configuration - simplified for stability
ADVANCED_CONFIG = {
    'use_spectral_augmentation': False,  # Disabled for stability
    'use_contrastive_learning': False,   # Disabled for stability
    'use_meta_learning': False,          # Disabled for stability
    'use_adversarial_training': False,   # Disabled for stability
    'use_ensemble': True,
    'use_knowledge_distillation': False, # Disabled for stability
    'use_progressive_training': False,   # Disabled for stability
    'num_ensemble_models': 3,
    'contrastive_weight': 0.1,
    'adversarial_weight': 0.2,
    'knowledge_distillation_weight': 0.3,
}

# Environment setup for verbosity
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["DATASETS_VERBOSITY"] = "info"

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =====================================================================================
# Enhanced Audio Preprocessing
# =====================================================================================

class EnhancedAudioPreprocessor:
    """Enhanced audio preprocessing with multiple advanced techniques."""
    
    def __init__(self, sr=16000):
        self.sr = sr
        self.feature_extractor = AdvancedFeatureExtractor(sr=sr)
        self.spectral_augmentation = SpectralAugmentation()
    
    def preprocess_audio(self, audio_path, start_time=None, end_time=None, augment=False):
        """Comprehensive audio preprocessing pipeline."""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sr)
        
        # Extract segment if specified
        if start_time is not None and end_time is not None:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio = audio[start_sample:end_sample]
        
        # Basic preprocessing
        audio = self._apply_pre_emphasis(audio)
        audio = self._apply_noise_reduction(audio)
        audio = self._apply_normalization(audio)
        
        # Extract multiple feature types
        features = self.feature_extractor.extract_spectral_features(audio)
        
        # Apply spectral augmentation if training
        if augment and ADVANCED_CONFIG['use_spectral_augmentation']:
            features['mel_spec'] = self.spectral_augmentation.spec_augment(
                torch.tensor(features['mel_spec']).unsqueeze(0)
            ).squeeze(0).numpy()
        
        return audio, features
    
    def _apply_pre_emphasis(self, audio, coeff=0.97):
        """Apply pre-emphasis filter."""
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])
    
    def _apply_noise_reduction(self, audio):
        """Simple spectral subtraction noise reduction."""
        # Estimate noise from first 0.1 seconds
        noise_sample_length = int(0.1 * self.sr)
        if len(audio) > noise_sample_length:
            # Get noise and audio spectra
            noise_segment = audio[:noise_sample_length]
            noise_spectrum = np.abs(np.fft.fft(noise_segment))
            audio_spectrum = np.fft.fft(audio)
            
            # Spectral subtraction
            magnitude = np.abs(audio_spectrum)
            phase = np.angle(audio_spectrum)
            
            # Ensure noise spectrum matches magnitude length by repeating/truncating
            if len(noise_spectrum) != len(magnitude):
                if len(noise_spectrum) < len(magnitude):
                    # Repeat noise spectrum to match audio length
                    repeat_factor = len(magnitude) // len(noise_spectrum) + 1
                    noise_spectrum = np.tile(noise_spectrum, repeat_factor)[:len(magnitude)]
                else:
                    # Truncate noise spectrum
                    noise_spectrum = noise_spectrum[:len(magnitude)]
            
            # Subtract noise spectrum
            clean_magnitude = magnitude - 0.5 * noise_spectrum
            clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
            
            # Reconstruct audio
            clean_spectrum = clean_magnitude * np.exp(1j * phase)
            audio = np.real(np.fft.ifft(clean_spectrum))
        
        return audio
    
    def _apply_normalization(self, audio):
        """Apply normalization and windowing."""
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio / (rms + 1e-8)
        
        # Apply Hamming window for spectral analysis
        if len(audio) > 1:
            window = np.hamming(len(audio))
            audio = audio * window
        
        return audio

# =====================================================================================
# Enhanced Dataset Class
# =====================================================================================

class AdvancedProfanityDataset(Dataset):
    """Advanced dataset with multiple feature types and augmentation strategies."""
    
    def __init__(self, df, wav2vec_feature_extractor, preprocessor, augmentor=None, max_length=16000, mode='train'):
        self.df = df
        self.wav2vec_feature_extractor = wav2vec_feature_extractor
        self.preprocessor = preprocessor
        self.augmentor = augmentor
        self.max_length = max_length
        self.mode = mode
        
        # Pre-compute difficulty scores for curriculum learning
        self.difficulty_scores = self._compute_difficulty_scores()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['file_path'].replace('\\', '/')
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        # Enhanced preprocessing
        augment = (self.mode == 'train' and self.augmentor is not None)
        audio_np, additional_features = self.preprocessor.preprocess_audio(
            file_path, row['start_time'], row['end_time'], augment=augment
        )
        
        # Apply traditional augmentation if specified
        if augment:
            audio_np = self.augmentor(audio_np)
        
        # Extract Wav2Vec2 features
        wav2vec_inputs = self.wav2vec_feature_extractor(
            audio_np, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Get label
        label = LABEL_MAP[row['label']]
        
        return {
            'input_values': wav2vec_inputs.input_values.squeeze(),
            'attention_mask': wav2vec_inputs.attention_mask.squeeze(),
            'additional_features': additional_features,
            'label': torch.tensor(label, dtype=torch.long),
            'difficulty_score': torch.tensor(self.difficulty_scores[idx], dtype=torch.float32),
            'file_path': file_path,
            'start_time': row['start_time'],
            'end_time': row['end_time']
        }
    
    def _compute_difficulty_scores(self):
        """Compute difficulty scores for curriculum learning."""
        scores = []
        for _, row in self.df.iterrows():
            # Base difficulty on label frequency (rare classes are harder)
            label_count = (self.df['label'] == row['label']).sum()
            frequency_score = 1.0 / (label_count + 1)
            
            # Add duration-based difficulty (very short/long segments are harder)
            duration = row['end_time'] - row['start_time']
            duration_score = 1.0 if duration < 0.5 or duration > 10.0 else 0.0
            
            # Combine scores
            total_score = 0.7 * frequency_score + 0.3 * duration_score
            scores.append(total_score)
        
        return np.array(scores)

# =====================================================================================
# Enhanced Model Architecture
# =====================================================================================

class MultiModalAudioClassifier(nn.Module):
    """Multi-modal audio classifier combining Wav2Vec2 with additional features."""
    
    def __init__(self, wav2vec_model_name, num_labels, additional_feature_dim=128):
        super().__init__()
        
        # Load pre-trained Wav2Vec2
        self.wav2vec2 = Wav2Vec2ForSequenceClassification.from_pretrained(
            wav2vec_model_name, num_labels=num_labels
        )
        
        # Get hidden dimension
        hidden_dim = self.wav2vec2.config.hidden_size
        
        # Additional feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(additional_feature_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Advanced transformer classifier
        self.advanced_classifier = TransformerAudioClassifier(
            input_dim=hidden_dim,
            num_labels=num_labels
        )
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Final classifier
        self.final_classifier = nn.Linear(hidden_dim, num_labels)
        
        # Teacher model for knowledge distillation (if enabled)
        self.teacher_model = None
    
    def forward(self, input_values, attention_mask=None, additional_features=None, return_all_outputs=False):
        # Get Wav2Vec2 hidden states
        wav2vec_outputs = self.wav2vec2.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = wav2vec_outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        
        # Process additional features if provided
        if additional_features is not None:
            # Flatten additional features (assuming they're spectral features)
            additional_flat = additional_features.view(additional_features.size(0), -1)
            processed_additional = self.feature_processor(additional_flat)
            
            # Expand to match sequence length
            processed_additional = processed_additional.unsqueeze(1).expand(
                -1, hidden_states.size(1), -1
            )
            
            # Fuse features
            fused_features = torch.cat([hidden_states, processed_additional], dim=-1)
            fused_features = self.fusion_layer(fused_features)
        else:
            fused_features = hidden_states
        
        # Apply advanced transformer classifier
        transformer_outputs = self.advanced_classifier(fused_features, attention_mask)
        
        # Final classification
        final_logits = self.final_classifier(transformer_outputs['features'])
        
        outputs = {
            'logits': final_logits,
            'hidden_states': fused_features,
            'transformer_features': transformer_outputs['features'],
            'uncertainty': transformer_outputs.get('uncertainty', None)
        }
        
        if return_all_outputs:
            outputs.update(transformer_outputs)
        
        return outputs
    
    def set_teacher_model(self, teacher_model):
        """Set teacher model for knowledge distillation."""
        self.teacher_model = teacher_model
        for param in self.teacher_model.parameters():
            param.requires_grad = False

# =====================================================================================
# Advanced Training Strategy
# =====================================================================================

class AdvancedTrainingStrategy:
    """Implements multiple advanced training techniques."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Initialize training components
        if config['use_contrastive_learning']:
            self.contrastive_loss = ContrastiveLoss()
        
        if config['use_meta_learning']:
            self.meta_optimizer = MetaLearningOptimizer(model)
        
        if config['use_adversarial_training']:
            self.adversarial_trainer = AdversarialTraining()
        
        # Loss weights
        self.contrastive_weight = config.get('contrastive_weight', 0.1)
        self.adversarial_weight = config.get('adversarial_weight', 0.2)
        self.kd_weight = config.get('knowledge_distillation_weight', 0.3)
    
    def compute_loss(self, model, inputs, labels, epoch=None):
        """Compute combined loss with multiple objectives."""
        # Standard classification loss
        outputs = model(**inputs)
        classification_loss = F.cross_entropy(outputs['logits'], labels)
        
        total_loss = classification_loss
        loss_components = {'classification': classification_loss.item()}
        
        # Contrastive learning
        if self.config['use_contrastive_learning'] and 'transformer_features' in outputs:
            contrastive_loss = self.contrastive_loss(outputs['transformer_features'], labels)
            total_loss += self.contrastive_weight * contrastive_loss
            loss_components['contrastive'] = contrastive_loss.item()
        
        # Knowledge distillation
        if (self.config['use_knowledge_distillation'] and 
            hasattr(model, 'teacher_model') and 
            model.teacher_model is not None):
            
            with torch.no_grad():
                teacher_outputs = model.teacher_model(**inputs)
            
            kd_loss = F.kl_div(
                F.log_softmax(outputs['logits'] / 3.0, dim=-1),
                F.softmax(teacher_outputs['logits'] / 3.0, dim=-1),
                reduction='batchmean'
            )
            total_loss += self.kd_weight * kd_loss
            loss_components['knowledge_distillation'] = kd_loss.item()
        
        return total_loss, loss_components
    
    def adversarial_training_step(self, model, optimizer, inputs, labels):
        """Perform adversarial training step."""
        if self.config['use_adversarial_training']:
            return self.adversarial_trainer.train_step(model, optimizer, inputs, labels)
        else:
            # Standard training step
            outputs = model(**inputs)
            loss = F.cross_entropy(outputs['logits'], labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            return loss.item()

# =====================================================================================
# Progressive Training Implementation
# =====================================================================================

class ProgressiveTrainer:
    """Implements progressive training with increasing model complexity."""
    
    def __init__(self, base_model_name, output_dir, config):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.config = config
        self.current_stage = 0
        
        # Define training stages
        self.stages = [
            {'name': 'basic', 'epochs': 10, 'lr': 5e-5, 'complexity': 'low'},
            {'name': 'intermediate', 'epochs': 15, 'lr': 3e-5, 'complexity': 'medium'},
            {'name': 'advanced', 'epochs': 20, 'lr': 1e-5, 'complexity': 'high'},
        ]
    
    def train_stage(self, stage_config, train_dataset, val_dataset, fold_num):
        """Train a single stage of progressive training."""
        print(f"Training stage: {stage_config['name']} (Fold {fold_num})")
        
        # Create model based on complexity
        if stage_config['complexity'] == 'low':
            # Simple Wav2Vec2 model
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.base_model_name, num_labels=NUM_LABELS
            )
        elif stage_config['complexity'] == 'medium':
            # Wav2Vec2 with basic enhancements
            model = MultiModalAudioClassifier(
                self.base_model_name, NUM_LABELS, additional_feature_dim=64
            )
        else:  # high complexity
            # Full advanced model
            model = MultiModalAudioClassifier(
                self.base_model_name, NUM_LABELS, additional_feature_dim=128
            )
        
        # Training setup
        training_strategy = AdvancedTrainingStrategy(model, self.config)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, 
            collate_fn=self._collate_fn, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False,
            collate_fn=self._collate_fn, num_workers=0
        )
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=stage_config['lr'],
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=stage_config['epochs']
        )
        
        # Training loop
        model.train()
        best_val_loss = float('inf')
        
        for epoch in range(stage_config['epochs']):
            print(f"Epoch {epoch + 1}/{stage_config['epochs']}")
            
            # Training
            train_loss = self._train_epoch(
                model, train_loader, optimizer, training_strategy, epoch
            )
            
            # Validation
            val_loss, val_metrics = self._validate_epoch(model, val_loader)
            
            # Update scheduler
            scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model(model, stage_config['name'], fold_num)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        return model
    
    def _train_epoch(self, model, dataloader, optimizer, training_strategy, epoch):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")
        
        for batch in progress_bar:
            # Prepare inputs
            inputs = {
                'input_values': batch['input_values'],
                'attention_mask': batch['attention_mask']
            }
            
            if 'additional_features' in batch:
                inputs['additional_features'] = batch['additional_features']
            
            labels = batch['labels']
            
            # Compute loss
            loss, loss_components = training_strategy.compute_loss(
                model, inputs, labels, epoch
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def _validate_epoch(self, model, dataloader):
        """Validate for one epoch."""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {
                    'input_values': batch['input_values'],
                    'attention_mask': batch['attention_mask']
                }
                
                if 'additional_features' in batch:
                    inputs['additional_features'] = batch['additional_features']
                
                labels = batch['labels']
                
                # Forward pass
                outputs = model(**inputs)
                loss = F.cross_entropy(outputs['logits'], labels)
                
                total_loss += loss.item()
                
                # Collect predictions
                predictions = torch.argmax(outputs['logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        return total_loss / len(dataloader), {'accuracy': accuracy}
    
    def _collate_fn(self, batch):
        """Custom collate function for batching."""
        # Filter out None samples
        batch = [item for item in batch if item is not None]
        if not batch:
            return {}
        
        # Prepare batch tensors
        result = {
            'input_values': torch.stack([item['input_values'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['label'] for item in batch]),
        }
        
        # Handle additional features if present
        if 'additional_features' in batch[0] and batch[0]['additional_features'] is not None:
            # Convert dict of numpy arrays to tensor
            additional_features = []
            for item in batch:
                # Flatten all spectral features
                features = item['additional_features']
                feature_vector = np.concatenate([
                    features['mfcc'].flatten(),
                    features['spectral_centroids'].flatten(),
                    features['spectral_rolloff'].flatten(),
                    features['spectral_bandwidth'].flatten(),
                    features['zero_crossing_rate'].flatten(),
                    features['chroma'].flatten(),
                    features['rms'].flatten()
                ])
                additional_features.append(feature_vector)
            
            # Pad to same length
            max_len = max(len(f) for f in additional_features)
            padded_features = []
            for f in additional_features:
                if len(f) < max_len:
                    f = np.pad(f, (0, max_len - len(f)), 'constant')
                padded_features.append(f[:max_len])  # Truncate if too long
            
            result['additional_features'] = torch.tensor(
                np.array(padded_features), dtype=torch.float32
            )
        
        return result
    
    def _save_model(self, model, stage_name, fold_num):
        """Save model checkpoint."""
        save_dir = os.path.join(self.output_dir, f'fold_{fold_num}', stage_name)
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'stage': stage_name,
            'fold': fold_num
        }, os.path.join(save_dir, 'model.pt'))

# =====================================================================================
# Main Training Pipeline
# =====================================================================================

class AdvancedModelPipeline:
    """Main pipeline orchestrating all advanced training techniques."""
    
    def __init__(self, model_name, output_dir, config):
        self.model_name = model_name
        self.output_dir = output_dir
        self.config = config
        
        # Initialize components
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, return_attention_mask=True, do_normalize=True
        )
        self.preprocessor = EnhancedAudioPreprocessor()
        self.progressive_trainer = ProgressiveTrainer(model_name, output_dir, config)
        self.evaluator = AdvancedEvaluator(CLASS_NAMES)
        
        # Storage for ensemble models
        self.ensemble_models = []
    
    def train_fold(self, train_df, val_df, fold_num):
        """Train a single fold with all advanced techniques."""
        print(f"\n=== Training Fold {fold_num} ===")
        
        # Create datasets
        train_dataset = AdvancedProfanityDataset(
            train_df, self.wav2vec_feature_extractor, self.preprocessor, mode='train'
        )
        val_dataset = AdvancedProfanityDataset(
            val_df, self.wav2vec_feature_extractor, self.preprocessor, mode='val'
        )
        
        # Progressive training
        if self.config['use_progressive_training']:
            final_model = None
            for stage_config in self.progressive_trainer.stages:
                model = self.progressive_trainer.train_stage(
                    stage_config, train_dataset, val_dataset, fold_num
                )
                final_model = model
        else:
            # Standard training
            final_model = self._standard_training(train_dataset, val_dataset, fold_num)
        
        # Add to ensemble
        if self.config['use_ensemble']:
            self.ensemble_models.append(final_model)
        
        return final_model
    
    def _standard_training(self, train_dataset, val_dataset, fold_num):
        """Standard training without progressive stages."""
        # Create model
        model = MultiModalAudioClassifier(
            self.model_name, NUM_LABELS, additional_feature_dim=128
        )
        
        # Training strategy
        training_strategy = AdvancedTrainingStrategy(model, self.config)
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True,
            collate_fn=self.progressive_trainer._collate_fn, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False,
            collate_fn=self.progressive_trainer._collate_fn, num_workers=0
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-5, weight_decay=0.01
        )
        
        # Training loop
        num_epochs = 30
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.progressive_trainer._train_epoch(
                model, train_loader, optimizer, training_strategy, epoch
            )
            
            # Validation
            val_loss, val_metrics = self.progressive_trainer._validate_epoch(
                model, val_loader
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_best_model(model, fold_num)
            
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        return model
    
    def _save_best_model(self, model, fold_num):
        """Save the best model."""
        save_path = os.path.join(self.output_dir, f'fold_{fold_num}', 'best_model.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
    
    def evaluate_ensemble(self, test_df):
        """Evaluate ensemble of models."""
        if not self.ensemble_models:
            print("No ensemble models available for evaluation.")
            return
        
        print(f"\n=== Ensemble Evaluation ===")
        print(f"Ensemble size: {len(self.ensemble_models)}")
        
        # Create test dataset
        test_dataset = AdvancedProfanityDataset(
            test_df, self.wav2vec_feature_extractor, self.preprocessor, mode='test'
        )
        test_loader = DataLoader(
            test_dataset, batch_size=16, shuffle=False,
            collate_fn=self.progressive_trainer._collate_fn, num_workers=0
        )
        
        # Create ensemble
        ensemble = ModelEnsemble(self.ensemble_models)
        
        # Evaluate
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        
        for batch in tqdm(test_loader, desc="Evaluating ensemble"):
            inputs = {
                'input_values': batch['input_values'],
                'attention_mask': batch['attention_mask']
            }
            
            if 'additional_features' in batch:
                inputs['additional_features'] = batch['additional_features']
            
            labels = batch['labels']
            
            # Ensemble prediction
            ensemble_output = ensemble.predict(inputs)
            predictions = torch.argmax(ensemble_output['predictions'], dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_uncertainties.extend(ensemble_output['uncertainty'].cpu().numpy())
        
        # Calculate metrics
        results = {
            'predictions': all_predictions,
            'labels': all_labels,
            'uncertainties': all_uncertainties,
            'classification_report': classification_report(
                all_labels, all_predictions, target_names=CLASS_NAMES, output_dict=True
            )
        }
        
        # Print results
        print("Ensemble Classification Report:")
        print(classification_report(all_labels, all_predictions, target_names=CLASS_NAMES))
        
        # Save results
        results_path = os.path.join(self.output_dir, 'ensemble_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'classification_report': results['classification_report'],
                'accuracy': np.mean(np.array(all_predictions) == np.array(all_labels)),
                'ensemble_size': len(self.ensemble_models)
            }, f, indent=2)
        
        # Advanced evaluation and visualization
        self.evaluator.plot_uncertainty_analysis(
            results, save_path=os.path.join(self.output_dir, 'uncertainty_analysis.png')
        )
        self.evaluator.calibration_analysis(results)
        
        return results

# =====================================================================================
# Main Execution
# =====================================================================================

def main():
    """Main training function."""
    # Configuration
    CSV_FILE = './csv/main.csv'
    MODEL_NAME = "airesearch/wav2vec2-large-xlsr-53-th"
    BASE_OUTPUT_DIR = './models/advanced_audio_train'
    NUM_FOLDS = 5
    
    # Create output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Load and prepare data
    print("Loading data...")
    df = pd.read_csv(CSV_FILE)
    
    # Handle class imbalance for splitting
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        if count < NUM_FOLDS:
            rows_to_add = df[df['label'] == label]
            for _ in range(NUM_FOLDS - count):
                df = pd.concat([df, rows_to_add], ignore_index=True)
    
    print(f"Dataset size after balancing: {len(df)}")
    print("Label distribution:")
    print(df['label'].value_counts())
    
    # Create train/test split
    train_val_df, test_df = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df['label']
    )
    
    # Initialize pipeline
    pipeline = AdvancedModelPipeline(MODEL_NAME, BASE_OUTPUT_DIR, ADVANCED_CONFIG)
    
    # K-Fold Cross-Validation
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    print(f"\nStarting {NUM_FOLDS}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_df)):
        train_fold_df = train_val_df.iloc[train_idx]
        val_fold_df = train_val_df.iloc[val_idx]
        
        # Oversample minority classes
        train_fold_df = oversample_minority_classes(train_fold_df)
        
        # Train fold
        model = pipeline.train_fold(train_fold_df, val_fold_df, fold + 1)
        
        # Memory cleanup
        torch.cuda.empty_cache()
    
    # Final ensemble evaluation
    print("\n" + "="*50)
    print("FINAL ENSEMBLE EVALUATION")
    print("="*50)
    
    ensemble_results = pipeline.evaluate_ensemble(test_df)
    
    print(f"\nFinal ensemble accuracy: {np.mean(np.array(ensemble_results['predictions']) == np.array(ensemble_results['labels'])):.4f}")
    print(f"Results saved to: {BASE_OUTPUT_DIR}")

def oversample_minority_classes(train_df):
    """Oversample minority profanity classes."""
    profanity_labels = [label for label in CLASS_NAMES if label != 'none']
    profanity_counts = train_df[train_df['label'].isin(profanity_labels)]['label'].value_counts()
    
    if not profanity_counts.empty:
        max_count = profanity_counts.max()
        oversampled_dfs = [train_df]
        
        for label, count in profanity_counts.items():
            if count < max_count:
                oversample_size = max_count - count
                label_df = train_df[train_df['label'] == label]
                oversampled_dfs.append(label_df.sample(n=oversample_size, replace=True, random_state=42))
        
        train_df = pd.concat(oversampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Oversampled training data. New size: {len(train_df)}")
    
    return train_df

if __name__ == "__main__":
    main()

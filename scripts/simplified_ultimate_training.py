import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Config
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
from sklearn.metrics import classification_report
import random
from tqdm import tqdm
import json

# =====================================================================================
# Configuration and Constants
# =====================================================================================

LABEL_MAP = {
    'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4,
    'ควย': 5, 'สวะ': 6, 'หี': 7, 'แตด': 8
}
NUM_LABELS = len(LABEL_MAP)
CLASS_NAMES = list(LABEL_MAP.keys())

# Simplified configuration - we'll enable features one by one to debug
ADVANCED_CONFIG = {
    'use_spectral_augmentation': False,  # Disabled for now
    'use_contrastive_learning': False,   # Disabled for now
    'use_adversarial_training': False,   # Disabled for now
    'use_ensemble': True,                # Keep this simple one
    'use_progressive_training': False,   # Disabled for now
    'use_focal_loss': True,              # This is a good one to keep
    'contrastive_weight': 0.1,
    'adversarial_weight': 0.2,
}

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =====================================================================================
# Simplified Audio Preprocessing
# =====================================================================================

class SimpleAudioPreprocessor:
    """Simplified but effective audio preprocessing."""
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def preprocess_audio(self, audio_path, start_time=None, end_time=None):
        """Simple but effective audio preprocessing."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Extract segment if specified
            if start_time is not None and end_time is not None:
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                audio = audio[start_sample:end_sample]
            
            # Basic preprocessing
            audio = self._apply_pre_emphasis(audio)
            audio = self._apply_simple_noise_reduction(audio)
            audio = self._apply_normalization(audio)
            
            return audio
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            # Return a small silent audio segment as fallback
            return np.zeros(int(0.5 * self.sr))
    
    def _apply_pre_emphasis(self, audio, coeff=0.97):
        """Apply pre-emphasis filter."""
        if len(audio) < 2:
            return audio
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])
    
    def _apply_simple_noise_reduction(self, audio):
        """Simple noise reduction using threshold."""
        # Simple threshold-based noise reduction
        noise_threshold = 0.005
        audio = np.where(np.abs(audio) < noise_threshold, 0, audio)
        return audio
    
    def _apply_normalization(self, audio):
        """Apply normalization and windowing."""
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio / (rms + 1e-8)
        
        # Apply Hamming window
        if len(audio) > 1:
            window = np.hamming(len(audio))
            audio = audio * window
        
        return audio

# =====================================================================================
# Focal Loss Implementation
# =====================================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# =====================================================================================
# Enhanced Dataset
# =====================================================================================

class SimpleProfanityDataset(Dataset):
    """Simplified dataset with better error handling."""
    
    def __init__(self, df, feature_extractor, preprocessor, max_length=16000, mode='train'):
        self.df = df
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
        self.max_length = max_length
        self.mode = mode
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            file_path = row['file_path'].replace('\\', '/')
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None
            
            # Preprocess audio
            audio_np = self.preprocessor.preprocess_audio(
                file_path, row['start_time'], row['end_time']
            )
            
            # Extract Wav2Vec2 features
            wav2vec_inputs = self.feature_extractor(
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
                'label': torch.tensor(label, dtype=torch.long),
                'file_path': file_path
            }
            
        except Exception as e:
            print(f"Error in dataset __getitem__ at index {idx}: {e}")
            return None

# =====================================================================================
# Enhanced Model Architecture
# =====================================================================================

class EnhancedAudioClassifier(nn.Module):
    """Enhanced audio classifier with improved architecture."""
    
    def __init__(self, wav2vec_model_name, num_labels):
        super().__init__()
        
        # Load pre-trained Wav2Vec2
        self.wav2vec2 = Wav2Vec2ForSequenceClassification.from_pretrained(
            wav2vec_model_name, 
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Don't replace the classifier - use the default one
        # The shape mismatch error suggests we should keep the original architecture
        
    def forward(self, input_values, attention_mask=None, labels=None):
        return self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels
        )

# =====================================================================================
# Custom Trainer with Focal Loss
# =====================================================================================

class AdvancedTrainer:
    """Custom trainer with advanced loss functions."""
    
    def __init__(self, model, train_loader, val_loader, device, class_weights=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Initialize loss function
        if ADVANCED_CONFIG['use_focal_loss'] and class_weights is not None:
            self.loss_fn = FocalLoss(alpha=1, gamma=2, weight=class_weights.to(device))
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=3e-5, 
            weight_decay=0.01
        )
        
        # Scheduler
        total_steps = len(train_loader) * 30  # Assume 30 epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch in progress_bar:
            if batch is None or len(batch['input_values']) == 0:
                continue
                
            # Move to device
            input_values = batch['input_values'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_values=input_values,
                attention_mask=attention_mask
            )
            
            # Compute loss
            loss = self.loss_fn(outputs.logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / max(num_batches, 1)
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None or len(batch['input_values']) == 0:
                    continue
                
                # Move to device
                input_values = batch['input_values'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_values=input_values,
                    attention_mask=attention_mask
                )
                
                # Compute loss
                loss = self.loss_fn(outputs.logits, labels)
                total_loss += loss.item()
                
                # Collect predictions
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        return total_loss / len(self.val_loader), {'accuracy': accuracy}

# =====================================================================================
# Collate Function
# =====================================================================================

def collate_fn(batch):
    """Custom collate function for batching."""
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    if not batch:
        return {'input_values': torch.empty(0), 'attention_mask': torch.empty(0), 'labels': torch.empty(0)}
    
    # Prepare batch tensors
    result = {
        'input_values': torch.stack([item['input_values'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch]),
    }
    
    return result

# =====================================================================================
# Training Pipeline
# =====================================================================================

class SimplifiedTrainingPipeline:
    """Simplified training pipeline focusing on stability and performance."""
    
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Initialize components
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, return_attention_mask=True, do_normalize=True
        )
        self.preprocessor = SimpleAudioPreprocessor()
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Storage for ensemble models
        self.trained_models = []
    
    def train_fold(self, train_df, val_df, fold_num):
        """Train a single fold."""
        print(f"\n=== Training Fold {fold_num} ===")
        
        # Create datasets
        train_dataset = SimpleProfanityDataset(
            train_df, self.feature_extractor, self.preprocessor, mode='train'
        )
        val_dataset = SimpleProfanityDataset(
            val_df, self.feature_extractor, self.preprocessor, mode='val'
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True,
            collate_fn=collate_fn, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )
        
        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(NUM_LABELS),
            y=[LABEL_MAP[label] for label in train_df['label']]
        )
        class_weights = torch.FloatTensor(class_weights)
        print(f"Class weights: {class_weights}")
        
        # Create model
        model = EnhancedAudioClassifier(self.model_name, NUM_LABELS)
        
        # Create trainer
        trainer = AdvancedTrainer(
            model, train_loader, val_loader, self.device, class_weights
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_val_accuracy = 0
        patience = 0
        max_patience = 10
        
        for epoch in range(30):  # Maximum 30 epochs
            # Training
            train_loss = trainer.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = trainer.validate_epoch()
            val_accuracy = val_metrics['accuracy']
            
            print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                self._save_model(model, fold_num)
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        print(f"Best validation accuracy for fold {fold_num}: {best_val_accuracy:.4f}")
        
        # Load best model and add to ensemble
        model = self._load_model(fold_num)
        if ADVANCED_CONFIG['use_ensemble']:
            self.trained_models.append(model)
        
        return model
    
    def _save_model(self, model, fold_num):
        """Save model checkpoint."""
        save_dir = os.path.join(self.output_dir, f'fold_{fold_num}')
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'fold': fold_num
        }, os.path.join(save_dir, 'best_model.pt'))
    
    def _load_model(self, fold_num):
        """Load model checkpoint."""
        model_path = os.path.join(self.output_dir, f'fold_{fold_num}', 'best_model.pt')
        
        model = EnhancedAudioClassifier(self.model_name, NUM_LABELS)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def evaluate_ensemble(self, test_df):
        """Evaluate ensemble of models."""
        if not self.trained_models:
            print("No ensemble models available for evaluation.")
            return
        
        print(f"\n=== Ensemble Evaluation ===")
        print(f"Ensemble size: {len(self.trained_models)}")
        
        # Create test dataset
        test_dataset = SimpleProfanityDataset(
            test_df, self.feature_extractor, self.preprocessor, mode='test'
        )
        test_loader = DataLoader(
            test_dataset, batch_size=16, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )
        
        # Evaluate
        all_predictions = []
        all_labels = []
        
        for batch in tqdm(test_loader, desc="Evaluating ensemble"):
            if batch is None or len(batch['input_values']) == 0:
                continue
            
            input_values = batch['input_values'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels']
            
            # Ensemble prediction
            ensemble_logits = []
            
            for model in self.trained_models:
                with torch.no_grad():
                    outputs = model(
                        input_values=input_values,
                        attention_mask=attention_mask
                    )
                    ensemble_logits.append(F.softmax(outputs.logits, dim=-1))
            
            # Average predictions
            avg_logits = torch.stack(ensemble_logits).mean(dim=0)
            predictions = torch.argmax(avg_logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        print("Ensemble Classification Report:")
        print(classification_report(all_labels, all_predictions, target_names=CLASS_NAMES))
        
        # Save results
        results = {
            'accuracy': accuracy,
            'classification_report': classification_report(
                all_labels, all_predictions, target_names=CLASS_NAMES, output_dict=True
            ),
            'ensemble_size': len(self.trained_models)
        }
        
        results_path = os.path.join(self.output_dir, 'ensemble_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

# =====================================================================================
# Main Execution
# =====================================================================================

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

def main():
    """Main training function."""
    # Configuration
    CSV_FILE = './csv/main.csv'
    MODEL_NAME = "airesearch/wav2vec2-large-xlsr-53-th"
    BASE_OUTPUT_DIR = './models/simplified_advanced_audio_train'
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
    pipeline = SimplifiedTrainingPipeline(MODEL_NAME, BASE_OUTPUT_DIR)
    
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
    
    if ensemble_results:
        print(f"\nFinal ensemble accuracy: {ensemble_results['accuracy']:.4f}")
        print(f"Ensemble size: {ensemble_results['ensemble_size']}")
        print(f"Results saved to: {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()

import torch
import torchaudio
import torch.nn.functional as F
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Config, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import librosa
from sklearn.model_selection import KFold, train_test_split
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

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

# Environment setup for verbosity
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["DATASETS_VERBOSITY"] = "info"

# =====================================================================================
# Audio Augmentation
# =====================================================================================

class AudioAugmentor:
    """
    Applies a chain of augmentations to audio data.
    """
    def __init__(self, sampling_rate=16000):
        self.sampling_rate = sampling_rate

    def __call__(self, audio_np):
        augmented = audio_np.copy()
        
        augmentation_types = np.random.choice(
            ['noise', 'pitch', 'speed', 'reverb', 'time_mask'],
            size=np.random.randint(1, 4),
            replace=False
        )
        
        for aug_type in augmentation_types:
            if aug_type == 'noise':
                noise_type = np.random.choice(['gaussian', 'pink', 'uniform'])
                if noise_type == 'gaussian':
                    noise_level = np.random.uniform(0.001, 0.005)
                    noise = np.random.normal(0, noise_level, len(augmented))
                elif noise_type == 'pink':
                    noise_level = np.random.uniform(0.001, 0.003)
                    # librosa.core.pink_noise is deprecated, using an alternative
                    noise = self._pink_noise(len(augmented)) * noise_level
                else:  # uniform
                    noise = np.random.uniform(-0.002, 0.002, len(augmented))
                augmented += noise

            elif aug_type == 'pitch':
                pitch_shift = np.random.uniform(-300, 300)
                augmented = librosa.effects.pitch_shift(
                    augmented, sr=self.sampling_rate, n_steps=pitch_shift / 100, bins_per_octave=12
                )

            elif aug_type == 'speed':
                speed_factor = np.random.uniform(0.8, 1.2)
                augmented = librosa.effects.time_stretch(augmented, rate=speed_factor)

            elif aug_type == 'reverb':
                reverb_delay = np.random.randint(1000, 3000)
                decay = np.random.uniform(0.1, 0.5)
                reverb = np.exp(-decay * np.linspace(0, 1, reverb_delay))
                augmented = np.convolve(augmented, reverb, mode='full')[:len(augmented)]

            elif aug_type == 'time_mask':
                mask_size = int(len(augmented) * np.random.uniform(0.05, 0.15))
                mask_start = np.random.randint(0, len(augmented) - mask_size)
                augmented[mask_start:mask_start + mask_size] = 0
        
        return augmented

    def _pink_noise(self, size):
        """Generate pink noise."""
        # This is a simple implementation. For more accuracy, a dedicated library might be better.
        uneven = size % 2
        X = np.random.randn(size // 2 + 1 + uneven) + 1j * np.random.randn(size // 2 + 1 + uneven)
        S = np.sqrt(np.arange(len(X)) + 1.)
        y = (np.fft.irfft(X / S)).real
        if uneven:
            y = y[:-1]
        return y / np.sqrt(np.mean(y**2))

# =====================================================================================
# Advanced Audio Preprocessing
# =====================================================================================

class AdvancedAudioPreprocessor:
    """Advanced audio preprocessing with spectral features and noise reduction."""
    
    def __init__(self, sampling_rate=16000):
        self.sr = sampling_rate
    
    def spectral_subtraction(self, audio, noise_factor=0.02):
        """Advanced noise reduction using spectral subtraction."""
        # Estimate noise from the first 0.1 seconds
        noise_sample = audio[:int(0.1 * self.sr)]
        noise_power = np.mean(noise_sample ** 2)
        
        # Apply spectral subtraction
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Subtract noise estimate
        clean_magnitude = magnitude - noise_factor * np.sqrt(noise_power)
        clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
        
        # Reconstruct audio
        clean_stft = clean_magnitude * np.exp(1j * phase)
        return librosa.istft(clean_stft)
    
    def apply_dynamic_range_compression(self, audio, threshold=0.1, ratio=4.0):
        """Apply dynamic range compression to enhance quiet sounds."""
        # Convert to dB
        audio_db = librosa.amplitude_to_db(np.abs(audio))
        
        # Apply compression
        compressed = np.where(
            audio_db > threshold,
            threshold + (audio_db - threshold) / ratio,
            audio_db
        )
        
        # Convert back to linear scale
        return librosa.db_to_amplitude(compressed) * np.sign(audio)
    
    def extract_mfcc_features(self, audio, n_mfcc=13):
        """Extract MFCC features for additional discriminative power."""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=n_mfcc)
        return np.mean(mfccs, axis=1)  # Take mean across time
    
    def voice_activity_detection(self, audio, frame_length=2048, hop_length=512):
        """Detect voice activity and focus on speech regions."""
        # Compute energy
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Threshold for voice activity (adjust based on your data)
        threshold = np.percentile(energy, 30)
        voice_frames = energy > threshold
        
        # Convert frame indices to sample indices
        voice_samples = []
        for i, is_voice in enumerate(voice_frames):
            if is_voice:
                start_sample = i * hop_length
                end_sample = min(start_sample + hop_length, len(audio))
                voice_samples.extend(range(start_sample, end_sample))
        
        if voice_samples:
            return audio[voice_samples]
        return audio

# =====================================================================================
# Custom Dataset
# =====================================================================================

class ProfanityAudioDataset(Dataset):
    """
    Enhanced PyTorch Dataset with advanced preprocessing and augmentation.
    """
    def __init__(self, df, feature_extractor, augmentor=None, max_length=16000, use_advanced_preprocessing=True):
        self.df = df
        self.feature_extractor = feature_extractor
        self.augmentor = augmentor
        self.max_length = max_length
        self.use_advanced_preprocessing = use_advanced_preprocessing
        
        # Initialize advanced preprocessor
        if use_advanced_preprocessing:
            self.advanced_preprocessor = AdvancedAudioPreprocessor()
            self.spec_augment = SpecAugment()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['file_path'].replace('\\', '/')
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None

        audio_np = self._load_and_preprocess_audio(file_path, row['start_time'], row['end_time'])
        
        if self.augmentor:
            audio_np = self.augmentor(audio_np)

        inputs = self.feature_extractor(
            audio_np, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        label = LABEL_MAP[row['label']]
        
        return {
            'input_values': inputs.input_values.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }    def _load_and_preprocess_audio(self, file_path, start_time, end_time):
        metadata = torchaudio.info(file_path)
        sr = metadata.sample_rate
        audio_length_sec = metadata.num_frames / sr

        padding = 0.2
        start_time = max(0, start_time - padding)
        end_time = min(end_time + padding, audio_length_sec)

        audio, sr = torchaudio.load(
            file_path, 
            frame_offset=int(start_time * sr), 
            num_frames=int((end_time - start_time) * sr)
        )

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        audio_np = audio.squeeze().numpy()
        
        if self.use_advanced_preprocessing:
            # Apply advanced preprocessing
            audio_np = self.advanced_preprocessor.spectral_subtraction(audio_np)
            audio_np = self.advanced_preprocessor.apply_dynamic_range_compression(audio_np)
            audio_np = self.advanced_preprocessor.voice_activity_detection(audio_np)
        
        # Apply original preprocessing
        if len(audio_np) > 0:
            audio_np = audio_np * np.hamming(len(audio_np))
            audio_np = librosa.effects.preemphasis(audio_np)

            # Simple noise reduction
            noise_threshold = 0.005
            audio_np = np.where(np.abs(audio_np) < noise_threshold, 0, audio_np)
            
            # Normalize
            audio_np = (audio_np - audio_np.mean()) / (audio_np.std() + 1e-8)
        else:
            # Fallback if advanced preprocessing removes all audio
            audio_np = audio.squeeze().numpy()
            if len(audio_np) > 0:
                audio_np = audio_np * np.hamming(len(audio_np))
                audio_np = librosa.effects.preemphasis(audio_np)
                noise_threshold = 0.005
                audio_np = np.where(np.abs(audio_np) < noise_threshold, 0, audio_np)
                audio_np = (audio_np - audio_np.mean()) / (audio_np.std() + 1e-8)
        
        return audio_np

# =====================================================================================
# Ensemble Model Architecture
# =====================================================================================

class MultiHeadClassifier(nn.Module):
    """Multi-head classifier with attention mechanism."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Multiple classification heads
        self.temporal_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_size // 2, config.num_labels)
        )
        
        self.spectral_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_size // 2, config.num_labels)
        )
        
        # Attention mechanism to combine heads
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config.num_labels * 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size // 4, config.num_labels)
        )
    
    def forward(self, features, **kwargs):
        batch_size = features.shape[0]
        
        # Apply attention to features
        attended_features, _ = self.attention(features.unsqueeze(1), features.unsqueeze(1), features.unsqueeze(1))
        attended_features = attended_features.squeeze(1)
        
        # Get predictions from multiple heads
        temporal_pred = self.temporal_head(features)
        spectral_pred = self.spectral_head(attended_features)
        
        # Combine predictions
        combined = torch.cat([temporal_pred, spectral_pred], dim=1)
        final_pred = self.fusion(combined)
        
        return final_pred

# =====================================================================================
# Custom Model and Trainer
# =====================================================================================

class ProfanityClassificationHead(nn.Module):
    """Custom classification head"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CustomTrainer(Trainer):
    """
    A custom trainer to handle class weights in the loss function.
    """
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}

    input_values = [item['input_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]

    input_values = pad_sequence(input_values, batch_first=True, padding_value=0.0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return {
        'input_values': input_values,
        'attention_mask': attention_mask,
        'labels': labels
    }

# =====================================================================================
# Advanced Training Strategies
# =====================================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance more effectively than weighted CE."""
    
    def __init__(self, alpha=1, gamma=2, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class AdvancedTrainer(Trainer):
    """Enhanced trainer with advanced loss functions and learning strategies."""
    
    def __init__(self, class_weights=None, use_focal_loss=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)
            
        # Initialize loss function
        if use_focal_loss:
            self.loss_fn = FocalLoss(alpha=1, gamma=2, weight=self.class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss = self.loss_fn(logits.view(-1, NUM_LABELS), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    
    def create_scheduler(self, num_training_steps, optimizer):
        """Create a custom learning rate scheduler."""
        from transformers import get_cosine_schedule_with_warmup
        
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )

# =====================================================================================
# Data Augmentation with SpecAugment
# =====================================================================================

class SpecAugment:
    """SpecAugment implementation for frequency domain augmentation."""
    
    def __init__(self, freq_mask_param=15, time_mask_param=35, num_freq_masks=1, num_time_masks=1):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, mel_spectrogram):
        """Apply SpecAugment to mel spectrogram."""
        spec = mel_spectrogram.copy()
        
        # Frequency masking
        for _ in range(self.num_freq_masks):
            freq_mask_size = np.random.randint(0, self.freq_mask_param)
            freq_mask_start = np.random.randint(0, spec.shape[0] - freq_mask_size)
            spec[freq_mask_start:freq_mask_start + freq_mask_size, :] = 0
        
        # Time masking
        for _ in range(self.num_time_masks):
            time_mask_size = np.random.randint(0, min(self.time_mask_param, spec.shape[1]))
            time_mask_start = np.random.randint(0, spec.shape[1] - time_mask_size)
            spec[:, time_mask_start:time_mask_start + time_mask_size] = 0
        
        return spec

# =====================================================================================
# Main Training Orchestrator
# =====================================================================================

class ModelTrainingPipeline:
    def __init__(self, model_name, base_output_dir):
        self.model_name = model_name
        self.base_output_dir = base_output_dir
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, return_attention_mask=True, do_normalize=True
        )
        self.augmentor = AudioAugmentor()

    def train_fold(self, train_df, val_df, fold_num):
        output_dir_fold = os.path.join(self.base_output_dir, f'fold_{fold_num}')
        print(f"--- Training Fold {fold_num} ---")
        print(f"Output directory: {output_dir_fold}")

        # Create datasets
        train_dataset = ProfanityAudioDataset(train_df, self.feature_extractor, augmentor=self.augmentor)
        val_dataset = ProfanityAudioDataset(val_df, self.feature_extractor)

        # Calculate class weights for the training fold
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(NUM_LABELS),
            y=[LABEL_MAP[label] for label in train_df['label']]
        )
        class_weights = torch.FloatTensor(class_weights)
        print("Class weights for this fold:", class_weights)        # Model configuration with advanced architecture
        config = Wav2Vec2Config.from_pretrained(
            self.model_name, num_labels=NUM_LABELS, finetuning_task="audio-classification"
        )
        config.pooling_mode = 'mean'
        model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name, config=config)
        
        # Replace classifier with advanced multi-head classifier
        model.projector = nn.Identity()
        model.classifier = MultiHeadClassifier(config)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir_fold,
            num_train_epochs=100,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=1,
            save_strategy="steps",
            save_steps=100,
            eval_steps=100,
            logging_steps=50,
            learning_rate=3e-5,
            warmup_ratio=0.1,
            weight_decay=0.01,
            fp16=True,
            gradient_checkpointing=True,
            # dataloader_num_workers=8,
            dataloader_pin_memory=True,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            eval_strategy="steps",
            save_total_limit=2,
        )        # Initialize advanced trainer with focal loss
        trainer = AdvancedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=15)],  # Increased patience
            data_collator=collate_fn,
            class_weights=class_weights,
            use_focal_loss=True  # Enable focal loss for better class imbalance handling
        )

        # Train
        trainer.train()
        trainer.save_model(output_dir_fold)
        self.feature_extractor.save_pretrained(output_dir_fold)
        print(f"--- Fold {fold_num} training complete ---")

    def evaluate_and_save_best_model(self, test_df, num_folds):
        fold_performances = {}
        best_accuracy = 0
        best_model_path = ""

        test_dataset = ProfanityAudioDataset(test_df, self.feature_extractor)

        for fold in range(1, num_folds + 1):
            model_path = os.path.join(self.base_output_dir, f'fold_{fold}')
            print(f"Evaluating model from: {model_path}")

            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()

            predictions, labels = [], []
            for item in test_dataset:
                if item is None: continue
                with torch.no_grad():
                    inputs = {k: v.unsqueeze(0).to(model.device) for k, v in item.items() if k != 'label'}
                    logits = model(**inputs).logits
                    pred = torch.argmax(logits, dim=-1).item()
                    predictions.append(pred)
                    labels.append(item['label'].item())
            
            accuracy = np.mean(np.array(predictions) == np.array(labels))
            report = classification_report(labels, predictions, target_names=CLASS_NAMES, output_dict=True)
            
            fold_performances[fold] = {'accuracy': accuracy, 'report': report}
            print(f"Fold {fold} Accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_path = model_path

        print(f"Best model is from {best_model_path} with accuracy: {best_accuracy:.4f}")
        
        # Save the best model to a dedicated directory
        best_model_dir = os.path.join(self.base_output_dir, 'best_model')
        os.makedirs(best_model_dir, exist_ok=True)
        
        best_model = Wav2Vec2ForSequenceClassification.from_pretrained(best_model_path)
        best_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(best_model_path)
        
        best_model.save_pretrained(best_model_dir)
        best_feature_extractor.save_pretrained(best_model_dir)
        print(f"Best model saved to: {best_model_dir}")

# =====================================================================================
# Main Execution
# =====================================================================================

if __name__ == "__main__":
    # Configuration
    CSV_FILE = './csv/main.csv'
    MODEL_NAME = "airesearch/wav2vec2-large-xlsr-53-th"
    BASE_OUTPUT_DIR = './models/audio_train_refactored'
    NUM_FOLDS = 5

    # Load and prepare data
    df = pd.read_csv(CSV_FILE)
    
    # Handle class imbalance for splitting
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        if count < NUM_FOLDS:
            rows_to_add = df[df['label'] == label]
            for _ in range(NUM_FOLDS - count):
                df = pd.concat([df, rows_to_add], ignore_index=True)

    # K-Fold Cross-Validation
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    # Use a held-out test set from the original data before augmentation/duplication
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])

    pipeline = ModelTrainingPipeline(MODEL_NAME, BASE_OUTPUT_DIR)

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_df)):
        train_fold_df = train_val_df.iloc[train_idx]
        val_fold_df = train_val_df.iloc[val_idx]

        # Oversample minority profanity classes in the training fold
        profanity_labels = [label for label in CLASS_NAMES if label != 'none']
        profanity_counts = train_fold_df[train_fold_df['label'].isin(profanity_labels)]['label'].value_counts()
        
        if not profanity_counts.empty:
            max_count = profanity_counts.max()
            oversampled_dfs = [train_fold_df]

            for label, count in profanity_counts.items():
                if count < max_count:
                    oversample_size = max_count - count
                    label_df = train_fold_df[train_fold_df['label'] == label]
                    oversampled_dfs.append(label_df.sample(n=oversample_size, replace=True, random_state=42))
            
            train_fold_df = pd.concat(oversampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"Fold {fold + 1}: Oversampled training data. New size: {len(train_fold_df)}")
        
        pipeline.train_fold(train_fold_df, val_fold_df, fold + 1)

    # Evaluate all folds on the test set and save the best one
    print("\n--- Final Evaluation on Test Set ---")
    pipeline.evaluate_and_save_best_model(test_df, NUM_FOLDS)

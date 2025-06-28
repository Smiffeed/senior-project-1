import torch
import torchaudio
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
# Custom Dataset
# =====================================================================================

class ProfanityAudioDataset(Dataset):
    """
    PyTorch Dataset for loading, preprocessing, and augmenting profanity audio data.
    """
    def __init__(self, df, feature_extractor, augmentor=None, max_length=16000):
        self.df = df
        self.feature_extractor = feature_extractor
        self.augmentor = augmentor
        self.max_length = max_length

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
        }

    def _load_and_preprocess_audio(self, file_path, start_time, end_time):
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
        
        # Apply Hamming window and pre-emphasis
        audio_np = audio_np * np.hamming(len(audio_np))
        audio_np = librosa.effects.preemphasis(audio_np)

        # Simple noise reduction
        noise_threshold = 0.005  # Adjust this value based on your needs
        audio_np = np.where(np.abs(audio_np) < noise_threshold, 0, audio_np)
        
        # Normalize
        audio_np = (audio_np - audio_np.mean()) / (audio_np.std() + 1e-8)
        
        return audio_np

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
# Self-Training
# =====================================================================================

def self_training_step(model, unlabeled_data, confidence_threshold=0.95):
    """Add confident predictions to training set."""
    pseudo_labels = []
    for batch in unlabeled_data:
        predictions = model(batch)
        confident_samples = predictions.max(dim=-1)[0] > confidence_threshold
        pseudo_labels.extend(confident_samples)
    return pseudo_labels

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
        print("Class weights for this fold:", class_weights)

        # Model configuration
        config = Wav2Vec2Config.from_pretrained(
            self.model_name, num_labels=NUM_LABELS, finetuning_task="audio-classification"
        )
        config.pooling_mode = 'mean'
        model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name, config=config)
        
        # Replace classifier head and bypass the projector
        model.projector = nn.Identity()
        model.classifier = ProfanityClassificationHead(config)

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
        )

        # Initialize trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
            data_collator=collate_fn,
            class_weights=class_weights
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
# Curriculum Learning
# =====================================================================================

def create_curriculum_dataset(df, difficulty_scores):
    """Create curriculum based on difficulty."""
    sorted_indices = np.argsort(difficulty_scores)
    return df.iloc[sorted_indices]

# =====================================================================================
# Advanced Model Improvement Techniques
# =====================================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling severe class imbalance - much better than weighted CE."""
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

class AdvancedAudioPreprocessor:
    """State-of-the-art audio preprocessing techniques."""
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def spectral_gating_denoising(self, audio, stationary_noise_reduction=6, non_stationary_noise_reduction=6):
        """Advanced noise reduction using spectral gating."""
        import scipy.signal
        
        # Convert to STFT
        f, t, stft = scipy.signal.stft(audio, fs=self.sr, nperseg=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise floor from quietest 10% of frames
        noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
        
        # Create spectral gate
        gate = magnitude / (noise_floor + 1e-10)
        gate = np.clip(gate, 0.1, 1.0)  # Soft gating
        
        # Apply gating
        denoised_stft = magnitude * gate * np.exp(1j * phase)
        
        # Convert back to time domain
        _, denoised_audio = scipy.signal.istft(denoised_stft, fs=self.sr)
        return denoised_audio
    
    def dynamic_range_compression(self, audio, threshold=-20, ratio=4, attack=0.003, release=0.1):
        """Professional audio compression for consistent levels."""
        # Convert to dB
        rms = np.sqrt(np.mean(audio**2))
        audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
        
        # Apply compression
        compressed = np.where(
            audio_db > threshold,
            threshold + (audio_db - threshold) / ratio,
            audio_db
        )
        
        # Convert back to linear
        return np.sign(audio) * (10 ** (compressed / 20))
    
    def voice_activity_detection_advanced(self, audio, frame_length=512, hop_length=160):
        """Advanced VAD using multiple features."""
        # Energy-based VAD
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames**2, axis=0)
        
        # Zero crossing rate
        zcr = np.sum(np.diff(np.sign(frames), axis=0) != 0, axis=0)
        
        # Spectral features
        spec_centroid = []
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            if len(frame) > 0:
                fft = np.abs(np.fft.rfft(frame))
                freqs = np.fft.rfftfreq(len(frame), 1/self.sr)
                centroid = np.sum(freqs * fft) / (np.sum(fft) + 1e-10)
                spec_centroid.append(centroid)
        
        spec_centroid = np.array(spec_centroid)
        
        # Combine features for VAD decision
        energy_thresh = np.percentile(energy, 30)
        zcr_thresh = np.percentile(zcr, 70)
        centroid_thresh = np.percentile(spec_centroid, 40)
        
        voice_mask = (energy > energy_thresh) & (zcr < zcr_thresh) & (spec_centroid > centroid_thresh)
        
        # Expand mask to original audio length
        voice_samples = np.repeat(voice_mask, hop_length)[:len(audio)]
        return audio * voice_samples

class MultiScaleAudioEncoder(nn.Module):
    """Multi-scale audio encoder for capturing different temporal patterns."""
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.scales = [1, 2, 4, 8]  # Different temporal scales
        
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim // len(self.scales), 
                         kernel_size=3*scale, stride=scale, padding=scale),
                nn.BatchNorm1d(hidden_dim // len(self.scales)),
                nn.ReLU(),
                nn.Conv1d(hidden_dim // len(self.scales), hidden_dim // len(self.scales),
                         kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(hidden_dim // len(self.scales)),
                nn.ReLU()
            ) for scale in self.scales
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape: (batch, input_dim, time)
        scale_outputs = []
        
        for encoder, scale in zip(self.scale_encoders, self.scales):
            # Encode at this scale
            scaled_out = encoder(x)
            # Upsample back to original resolution
            upsampled = F.interpolate(scaled_out, size=x.shape[-1], mode='linear', align_corners=False)
            scale_outputs.append(upsampled)
        
        # Concatenate all scales
        multi_scale = torch.cat(scale_outputs, dim=1)
        
        # Fuse information
        fused = self.fusion(multi_scale)
        return fused

class AttentionPooling(nn.Module):
    """Attention-based pooling for better sequence representation."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, mask=None):
        # x shape: (batch, seq_len, hidden_dim)
        # mask shape: (batch, seq_len)
        
        attention_weights = self.attention(x).squeeze(-1)  # (batch, seq_len)
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(~mask.bool(), float('-inf'))
        
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        return pooled, attention_weights

class ImprovedProfanityClassifier(nn.Module):
    """Advanced classifier with multiple improvements."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_size
        
        # Multi-scale audio encoder
        self.multi_scale_encoder = MultiScaleAudioEncoder(hidden_dim, hidden_dim)
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2, 
            num_layers=2, batch_first=True, 
            bidirectional=True, dropout=0.3
        )
        
        # Self-attention mechanism
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Attention pooling
        self.attention_pooling = AttentionPooling(hidden_dim)
        
        # Multi-layer classifier with residual connections
        self.classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            nn.Linear(hidden_dim // 2, config.num_labels)
        ])
        
        # Auxiliary tasks for better representation learning
        self.emotion_classifier = nn.Linear(hidden_dim, 5)  # anger, neutral, surprise, etc.
        self.intensity_regressor = nn.Linear(hidden_dim, 1)  # profanity intensity
        
    def forward(self, features, attention_mask=None, return_auxiliary=False):
        batch_size, seq_len, hidden_dim = features.shape
        
        # Multi-scale encoding
        features_transposed = features.transpose(1, 2)  # (batch, hidden_dim, seq_len)
        multi_scale_features = self.multi_scale_encoder(features_transposed)
        multi_scale_features = multi_scale_features.transpose(1, 2)  # Back to (batch, seq_len, hidden_dim)
        
        # LSTM processing
        lstm_out, _ = self.lstm(multi_scale_features)
        
        # Self-attention
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out, key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        
        # Residual connection
        features_enhanced = lstm_out + attn_out
        
        # Attention pooling
        pooled_features, attention_weights = self.attention_pooling(features_enhanced, attention_mask)
        
        # Multi-layer classification with residual connections
        x = pooled_features
        for i, layer in enumerate(self.classifier[:-1]):
            residual = x
            x = layer(x)
            if x.shape == residual.shape:  # Add residual connection when dimensions match
                x = x + residual
        
        logits = self.classifier[-1](x)
        
        result = {'logits': logits}
        
        if return_auxiliary:
            emotion_logits = self.emotion_classifier(pooled_features)
            intensity_score = self.intensity_regressor(pooled_features)
            result.update({
                'emotion_logits': emotion_logits,
                'intensity_score': intensity_score,
                'attention_weights': attention_weights
            })
        
        return result

class AdvancedTrainer(Trainer):
    """Enhanced trainer with multiple loss functions and advanced training strategies."""
    
    def __init__(self, class_weights=None, use_focal_loss=True, auxiliary_loss_weight=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.auxiliary_loss_weight = auxiliary_loss_weight
        
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)
        
        # Initialize loss functions
        if use_focal_loss:
            self.main_loss_fn = FocalLoss(alpha=1, gamma=2, weight=self.class_weights)
        else:
            self.main_loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            
        self.emotion_loss_fn = nn.CrossEntropyLoss()
        self.intensity_loss_fn = nn.MSELoss()
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        
        # Get model outputs
        outputs = model(**inputs, return_auxiliary=True)
        logits = outputs['logits']
        
        # Main classification loss
        main_loss = self.main_loss_fn(logits.view(-1, NUM_LABELS), labels.view(-1))
        
        # Auxiliary losses (if available)
        total_loss = main_loss
        
        if 'emotion_logits' in outputs and self.auxiliary_loss_weight > 0:
            # Create pseudo emotion labels based on profanity type
            emotion_labels = self._create_emotion_labels(labels)
            emotion_loss = self.emotion_loss_fn(outputs['emotion_logits'], emotion_labels)
            total_loss += self.auxiliary_loss_weight * emotion_loss
            
        if 'intensity_score' in outputs and self.auxiliary_loss_weight > 0:
            # Create intensity labels (0 for none, increasing for profanity severity)
            intensity_labels = self._create_intensity_labels(labels)
            intensity_loss = self.intensity_loss_fn(outputs['intensity_score'].squeeze(), intensity_labels)
            total_loss += self.auxiliary_loss_weight * intensity_loss
        
        # Create outputs object for compatibility
        from transformers.modeling_outputs import SequenceClassifierOutput
        final_outputs = SequenceClassifierOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=None,
            attentions=outputs.get('attention_weights', None)
        )
        
        return (total_loss, final_outputs) if return_outputs else total_loss
    
    def _create_emotion_labels(self, profanity_labels):
        """Create emotion labels based on profanity type."""
        # Map profanity types to emotions: 0=neutral, 1=anger, 2=disgust, 3=surprise, 4=emphasis
        emotion_mapping = {
            0: 0,  # none -> neutral
            1: 1,  # เย็ด -> anger
            2: 1,  # กู -> anger
            3: 1,  # มึง -> anger  
            4: 2,  # เหี้ย -> disgust
            5: 1,  # ควย -> anger
            6: 2,  # สวะ -> disgust
            7: 1,  # หี -> anger
            8: 2,  # แตด -> disgust
        }
        
        emotion_labels = torch.tensor([emotion_mapping[label.item()] for label in profanity_labels])
        return emotion_labels.to(profanity_labels.device)
    
    def _create_intensity_labels(self, profanity_labels):
        """Create intensity labels for regression."""
        # Assign intensity scores: none=0, mild=0.3, moderate=0.6, severe=1.0
        intensity_mapping = {
            0: 0.0,  # none
            1: 0.8,  # เย็ด (severe)
            2: 0.6,  # กู (moderate)
            3: 0.5,  # มึง (mild-moderate)
            4: 0.7,  # เหี้ย (moderate-severe)
            5: 0.9,  # ควย (severe)
            6: 0.4,  # สวะ (mild)
            7: 0.8,  # หี (severe)
            8: 0.6,  # แตด (moderate)
        }
        
        intensity_labels = torch.tensor([intensity_mapping[label.item()] for label in profanity_labels], dtype=torch.float32)
        return intensity_labels.to(profanity_labels.device)

class CurriculumLearningScheduler:
    """Implement curriculum learning for gradual difficulty increase."""
    
    def __init__(self, total_epochs=100):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def get_difficulty_threshold(self):
        """Return current difficulty threshold (0=easy, 1=hard)."""
        return min(1.0, self.current_epoch / (self.total_epochs * 0.7))
    
    def update_epoch(self, epoch):
        self.current_epoch = epoch
        
    def filter_dataset_by_difficulty(self, dataset, difficulty_scores):
        """Filter dataset based on current curriculum stage."""
        threshold = self.get_difficulty_threshold()
        easy_samples = difficulty_scores <= threshold
        return dataset[easy_samples]

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

def ensemble_predict(models, inputs):
    """Ensemble prediction from multiple models."""
    predictions = []
    for model in models:
        with torch.no_grad():
            logits = model(**inputs).logits
            predictions.append(F.softmax(logits, dim=-1))
    
    # Average predictions
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred

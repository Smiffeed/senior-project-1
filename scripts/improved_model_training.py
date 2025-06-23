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
        model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name, config=config)
        
        # Replace classifier head and bypass the projector
        model.projector = nn.Identity()
        model.classifier = ProfanityClassificationHead(config)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir_fold,
            num_train_epochs=100,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
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
            dataloader_num_workers=4,
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
        
        pipeline.train_fold(train_fold_df, val_fold_df, fold + 1)

    # Evaluate all folds on the test set and save the best one
    print("\n--- Final Evaluation on Test Set ---")
    pipeline.evaluate_and_save_best_model(test_df, NUM_FOLDS)

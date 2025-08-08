import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Config
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import EarlyStoppingCallback
import librosa
from sklearn.model_selection import KFold
import torch.nn as nn
import matplotlib.pyplot as plt
import tempfile
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel, Wav2Vec2Model
from collections import Counter
from sklearn.model_selection import train_test_split
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import EarlyStoppingCallback
import librosa
from sklearn.model_selection import KFold
import torch.nn as nn
import matplotlib.pyplot as plt
import tempfile
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel, Wav2Vec2Model

# Set these environment variables before running your Python script
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["DATASETS_VERBOSITY"] = "info"
os.environ["PYTHONPATH"] = "."

# Define label mapping
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

# Define the number of labels
num_labels = len(label_map)

def calculate_class_weights(df):
    """Calculate balanced class weights"""
    labels = [label_map[label] for label in df['label']]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.FloatTensor(class_weights)

class CustomTrainer(Trainer):
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
            # Compute cross entropy
            ce_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')(
                logits.view(-1, num_labels), labels.view(-1)
            )
            
            # Compute probabilities and focal weight
            pt = torch.exp(-ce_loss)
            
            # Apply alpha weighting for class balance
            if self.alpha is not None:
                # Create alpha tensor for each class
                alpha_t = torch.ones_like(labels, dtype=torch.float)
                for i in range(num_labels):
                    mask = (labels == i)
                    if i == 0:  # 'none' class gets lower alpha
                        alpha_t[mask] = 1.0 - self.alpha
                    else:  # profanity classes get higher alpha
                        alpha_t[mask] = self.alpha
                alpha_t = alpha_t.to(self.args.device)
                focal_weight = alpha_t * ((1 - pt) ** self.gamma)
            else:
                focal_weight = (1 - pt) ** self.gamma
                
            loss = (focal_weight * ce_loss).mean()
        else:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss

class SafeSavingTrainer(CustomTrainer):
    def _save_optimizer_and_scheduler(self, output_dir):
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"), _use_new_zipfile_serialization=False)
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"), _use_new_zipfile_serialization=False)

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
    
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
        # Add per-class F1 scores for monitoring
        "f1_เย็ด": f1[1] if len(f1) > 1 else 0.0,
        "f1_กู": f1[2] if len(f1) > 2 else 0.0,
        "f1_มึง": f1[3] if len(f1) > 3 else 0.0,
        "f1_เหี้ย": f1[4] if len(f1) > 4 else 0.0,
        "f1_ควย": f1[5] if len(f1) > 5 else 0.0,
        "f1_สวะ": f1[6] if len(f1) > 6 else 0.0,
        "f1_หี": f1[7] if len(f1) > 7 else 0.0,
        "f1_แตด": f1[8] if len(f1) > 8 else 0.0,
    }

# Load and preprocess the dataset
def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    return df

def normalize_file_path(file_path):
    """
    Normalize file paths for cross-platform compatibility.
    Convert Windows absolute paths to relative paths when running on non-Windows systems.
    """
    import platform
    
    # Replace backslashes with forward slashes
    file_path = file_path.replace('\\', '/')
    
    # If this is a Windows absolute path but we're not on Windows, convert to relative
    if file_path.startswith('C:/Users/muldi/Documents/Playground/University/senior-project-1/'):
        # Extract the relative path from the project root
        relative_path = file_path.replace('C:/Users/muldi/Documents/Playground/University/senior-project-1/', '')
        return relative_path
    
    # Handle other possible absolute path formats
    if file_path.startswith('/teamspace/studios/this_studio/'):
        # Already in Lightning.ai format
        return file_path
    
    # If it's already a relative path, keep it as is
    if not file_path.startswith('/') and not file_path.startswith('C:'):
        return file_path
    
    return file_path
def preprocess_audio(file_path, start_time, end_time, max_length=16000):
    """Load and preprocess audio segment with enhanced noise reduction and feature preservation"""
    try:
        # Get the sample rate
        metadata = torchaudio.info(file_path)
        sr = metadata.sample_rate
        
        # Load the specific segment
        audio, sr = torchaudio.load(file_path, 
                                  frame_offset=int(start_time * sr), 
                                  num_frames=int((end_time - start_time) * sr))
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Convert to numpy array for processing
        audio_np = audio.squeeze().numpy()
        
        # Enhanced preprocessing pipeline
        # 1. Apply pre-emphasis filter first (preserves high-frequency content)
        audio_np = librosa.effects.preemphasis(audio_np, coef=0.97)
        
        # 2. Noise reduction using spectral gating
        # Estimate noise floor from quiet portions
        audio_energy = np.abs(audio_np)
        noise_threshold = np.percentile(audio_energy, 20)  # Bottom 20% as noise estimate
        
        # Apply gentle noise gate (preserve low-amplitude speech)
        noise_gate_threshold = noise_threshold * 2.5
        audio_np = np.where(audio_energy < noise_gate_threshold, 
                           audio_np * 0.1,  # Reduce but don't eliminate
                           audio_np)
        
        # 3. Apply Hamming window (but preserve more of the signal)
        window_length = len(audio_np)
        if window_length > 160:  # Only apply if reasonable length
            # Use a gentler window to preserve more information
            tukey_window = np.blackman(window_length)
            audio_np = audio_np * (0.7 + 0.3 * tukey_window)  # Blend with original
        
        # 4. Dynamic range compression to enhance weak profanity
        # Compress dynamic range while preserving relative loudness
        audio_rms = np.sqrt(np.mean(audio_np**2))
        if audio_rms > 1e-6:
            compression_ratio = 0.6
            audio_np = np.sign(audio_np) * (np.abs(audio_np) ** compression_ratio)
        
        # Convert back to torch tensor
        audio = torch.from_numpy(audio_np).unsqueeze(0)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        
        # Enhanced normalization that preserves dynamic information
        audio_mean = audio.mean()
        audio_std = audio.std()
        if audio_std > 1e-8:
            audio = (audio - audio_mean) / audio_std
        else:
            audio = audio - audio_mean
            
        # Scale to reasonable range
        audio = audio * 0.5  # Prevent saturation
        
        # Pad or truncate to max_length
        if audio.shape[1] < max_length:
            # Use reflection padding to avoid discontinuities
            pad_length = max_length - audio.shape[1]
            if audio.shape[1] > pad_length:
                audio = torch.nn.functional.pad(audio, (0, pad_length), mode='reflect')
            else:
                audio = torch.nn.functional.pad(audio, (0, pad_length))
        else:
            audio = audio[:, :max_length]
        
        return audio.squeeze().numpy()
        
    except Exception as e:
        print(f"Error processing audio {file_path}: {e}")
        # Return silence if processing fails
        return np.zeros(max_length, dtype=np.float32)
# Prepare dataset for Hugging Face Trainer
def prepare_dataset(df, feature_extractor):
    def process_example(example):
        file_path = normalize_file_path(example['file_path'])
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            # Return a dummy example instead of None to avoid dataset mapping issues
            return {
                'input_values': np.zeros(16000, dtype=np.float32),
                'label': 0,  # Default to 'none' class
                'is_valid': False
            }
            
        try:
            # Get audio length first
            metadata = torchaudio.info(file_path)
            audio_length = metadata.num_frames / metadata.sample_rate
                
            # Add padding around the profanity segments
            padding = 0.2  # Increase from 0.1 to 0.2 seconds
            start_time = max(0, example['start_time'] - padding)
            end_time = min(example['end_time'] + padding, audio_length)
            
            # Process audio with background noise removal
            audio = preprocess_audio(file_path, start_time, end_time)
            
            # Apply feature extractor
            inputs = feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=False  # Changed to False to avoid padding issues
            )
            
            # Get input_values
            input_values = inputs.input_values.squeeze().numpy()
            
            # Check if label is within the expected range
            label = label_map.get(example['label'])
            if label is None or label < 0 or label >= num_labels:
                print(f"Warning: Label {example['label']} is out of range or not found in label_map for file {file_path}")
                return {
                    'input_values': np.zeros(16000, dtype=np.float32),
                    'label': 0,  # Default to 'none' class
                    'is_valid': False
                }
            
            return {
                'input_values': input_values,
                'label': label,
                'is_valid': True
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return {
                'input_values': np.zeros(16000, dtype=np.float32),
                'label': 0,  # Default to 'none' class
                'is_valid': False
            }
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(process_example, remove_columns=dataset.column_names)
    # Filter out invalid examples after mapping
    dataset = dataset.filter(lambda example: example.get('is_valid', False))
    # Remove the is_valid column as it's no longer needed
    dataset = dataset.remove_columns(['is_valid'])
    
    return dataset
def augment_dataset(dataset, feature_extractor):
    """
    Heavily augment rare classes to balance the dataset better.
    Target: Balance all profanity classes to have similar counts.
    """
    from collections import Counter
    class_counts = Counter([ex['label'] for ex in dataset])
    
    # Calculate target count (use the median count of profanity classes, excluding 'none')
    profanity_counts = {k: v for k, v in class_counts.items() if k != 0}  # 0 = 'none'
    target_count = max(profanity_counts.values()) if profanity_counts else 100
    
    print(f"Class counts before augmentation: {class_counts}")
    print(f"Target count for minority classes: {target_count}")
    
    augmented_examples = []
    for example in dataset:
        label = example['label']
        current_count = class_counts[label]
        
        # Calculate number of augmentations needed
        if label == 0:  # 'none' class - minimal augmentation
            n_aug = 1
        elif current_count < target_count * 0.3:  # Very rare classes
            n_aug = 8
        elif current_count < target_count * 0.5:  # Moderately rare classes
            n_aug = 5
        elif current_count < target_count * 0.8:  # Less rare classes
            n_aug = 3
        else:  # Common profanity classes
            n_aug = 2
            
        for _ in range(n_aug):
            aug_input = np.array(example['input_values'])
            
            # Apply multiple augmentation techniques
            # 1. Volume variation (more aggressive for rare classes)
            volume_range = (0.7, 1.4) if label != 0 else (0.9, 1.1)
            aug_input = aug_input * np.random.uniform(*volume_range)
            
            # 2. Add noise (contextual for profanity vs clean)
            noise_level = np.random.uniform(0.001, 0.005) if label != 0 else np.random.uniform(0.0005, 0.002)
            aug_input = aug_input + np.random.normal(0, noise_level, aug_input.shape)
            
            # 3. Time stretching (more variation for profanity)
            stretch_range = (0.85, 1.2) if label != 0 else (0.95, 1.05)
            if np.random.rand() < 0.7:
                aug_input = librosa.effects.time_stretch(aug_input, rate=np.random.uniform(*stretch_range))
                
            # 4. Pitch shifting (preserve Thai tonal characteristics)
            if np.random.rand() < 0.6:
                pitch_range = (-3, 3) if label != 0 else (-1.5, 1.5)
                aug_input = librosa.effects.pitch_shift(
                    aug_input, sr=16000, 
                    n_steps=np.random.uniform(*pitch_range),
                    bins_per_octave=200
                )
                
            # 5. Frequency masking (simulate partial occlusion)
            if np.random.rand() < 0.4:
                mask_size = int(len(aug_input) * np.random.uniform(0.03, 0.12))
                mask_start = np.random.randint(0, max(1, len(aug_input) - mask_size))
                aug_input[mask_start:mask_start + mask_size] *= np.random.uniform(0.1, 0.4)
                
            # 6. Add reverb effect occasionally
            if np.random.rand() < 0.3 and label != 0:
                reverb_decay = np.random.uniform(0.1, 0.3)
                reverb_delay = np.random.randint(500, 1500)
                reverb = np.exp(-reverb_decay * np.linspace(0, 1, reverb_delay))
                aug_input = np.convolve(aug_input, reverb, mode='full')[:len(aug_input)]
            
            # 7. Normalize to prevent clipping
            max_val = np.max(np.abs(aug_input))
            if max_val > 0:
                aug_input = aug_input / (max_val + 1e-6)
                
            augmented_examples.append({
                'input_values': aug_input,
                'label': label
            })
    
    # Add all augmented examples to dataset
    for aug_example in augmented_examples:
        dataset = dataset.add_item(aug_example)
        
    # Print final distribution
    final_counts = Counter([ex['label'] for ex in dataset])
    print(f"Class counts after augmentation: {final_counts}")
    
    return dataset
def collate_fn(batch):
    # Filter out None or corrupted examples
    valid_batch = []
    for item in batch:
        if item is None:
            continue
        
        # Validate input_values
        if 'input_values' not in item or item['input_values'] is None:
            continue
        
        input_vals = item['input_values']
        if hasattr(input_vals, '__len__') and len(input_vals) == 0:
            continue
        
        # Validate label
        if 'label' not in item or item['label'] is None:
            continue
            
        label_val = item['label']
        if not isinstance(label_val, (int, float)) or label_val < 0 or label_val >= num_labels:
            continue
            
        valid_batch.append(item)
    
    if len(valid_batch) == 0:
        # Return a dummy batch if all examples are invalid
        dummy_input = torch.zeros(1, 16000)
        dummy_label = torch.tensor([0])
        return {
            'input_values': dummy_input,
            'labels': dummy_label
        }
    
    # Process valid examples
    input_values = []
    labels = []
    
    for item in valid_batch:
        try:
            # Convert and validate input_values
            input_tensor = torch.tensor(item['input_values']).squeeze()
            if input_tensor.dim() == 0:  # scalar
                input_tensor = input_tensor.unsqueeze(0)
            
            # Ensure tensor has correct length
            if len(input_tensor) == 0:
                print(f"Warning: Empty input tensor, using dummy data")
                input_tensor = torch.zeros(16000)
            
            input_values.append(input_tensor)
            labels.append(item['label'])
                
        except Exception as e:
            print(f"Warning: Skipping corrupted sample: {e}")
            continue
    
    if len(input_values) == 0:
        # Fallback to dummy batch
        dummy_input = torch.zeros(1, 16000)
        dummy_label = torch.tensor([0])
        return {
            'input_values': dummy_input,
            'labels': dummy_label
        }
    
    # Pad sequences to same length
    input_values = pad_sequence(input_values, batch_first=True)
    labels = torch.tensor(labels)
    
    return {
        'input_values': input_values,
        'labels': labels
    }
# Main training function using pre-prepared datasets
def train_wav2vec2_model_with_data(train_dataset, val_dataset, model_name, output_dir):
    """
    Train the model using pre-prepared train and validation datasets
    """
    # Calculate class weights from training dataset
    train_labels = [ex['label'] for ex in train_dataset]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.FloatTensor(class_weights)
    
    print("\nClass weights:")
    for label, weight in zip(label_map.keys(), class_weights):
        print(f"{label}: {weight:.4f}")
    
    # Check class distribution
    class_counts = Counter(train_labels)
    print("\nTraining set class distribution:")
    for label_idx, label_name in enumerate(label_map.keys()):
        count = class_counts.get(label_idx, 0)
        print(f"{label_name}: {count}")
    
    # Load pre-trained model and feature extractor
    config = Wav2Vec2Config.from_pretrained(
        model_name,
        num_labels=num_labels,
        finetuning_task="audio-classification",
        attention_dropout=0.3,
        hidden_dropout=0.5
    )
    
    # Initialize the model
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        config=config
    )
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        return_attention_mask=True,
        do_normalize=True,
    )
    
    # Training arguments optimized with best hyperparameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=150,            # Best: 150 epochs
        per_device_train_batch_size=16,   # Best: 4 batch size
        per_device_eval_batch_size=16,    # Match training batch size
        gradient_accumulation_steps=2,   # Best: 2 accumulation steps
        save_strategy="steps",
        save_steps=500,                  # Best: 500 save steps
        logging_dir=f"{output_dir}/logs",
        eval_steps=250,                  # Best: 250 eval steps
        logging_steps=30,
        learning_rate=3e-5,              # Best: 3e-05 learning rate
        save_total_limit=3,
        warmup_ratio=0.05,               # Best: 0.05 warmup ratio
        weight_decay=0.0,                # Best: 0.0 weight decay
        fp16=True,                       # Best: True
        bf16=False,
        gradient_checkpointing=True,    # Best: False
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",  # Focus on F1 instead of accuracy
        greater_is_better=True,
        eval_strategy="steps",
        push_to_hub=False,
        save_on_each_node=False,
        disable_tqdm=False,
        remove_unused_columns=True,
        # Additional parameters for better convergence
        adam_epsilon=1e-6,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",  # Fixed scheduler type
        dataloader_num_workers=0,    # ADDED: Disable multiprocessing to prevent corruption
        dataloader_drop_last=True,   # ADDED: Drop incomplete batches
    )
    
    # Initialize trainer with enhanced focal loss and best early stopping patience
    trainer = SafeSavingTrainer(
        class_weights=class_weights,
        use_focal_loss=True, 
        gamma=3.0,              # Increased gamma for harder examples
        alpha=0.75,             # Higher alpha to focus more on profanity classes
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],  # Best: 20 patience
        data_collator=collate_fn,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(output_dir)
    feature_extractor.save_pretrained(output_dir)

# Main training function
def train_wav2vec2_model(csv_file, model_name, output_dir):
    # Load dataset
    df = load_dataset(csv_file)
    
    # Calculate class weights
    class_weights = calculate_class_weights(df)
    print("\nClass weights:")
    for label, weight in zip(label_map.keys(), class_weights):
        print(f"{label}: {weight:.4f}")
    
    # Check class distribution
    class_counts = df['label'].value_counts()
    print("\nClass distribution:")
    print(class_counts)
    
    # For classes with only one sample, duplicate it
    for label, count in class_counts.items():
        if count < 2:
            # Find the row with this label
            row_to_duplicate = df[df['label'] == label].iloc[0]
            # Add it to the dataframe again
            df = pd.concat([df, pd.DataFrame([row_to_duplicate])], ignore_index=True)
    
    # Now perform the train-test split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Load pre-trained model and feature extractor
    config = Wav2Vec2Config.from_pretrained(
        model_name,
        num_labels=num_labels,
        finetuning_task="audio-classification",
        attention_dropout=0.3,
        hidden_dropout=0.5
    )
    
    # Initialize the model
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
    train_dataset = prepare_dataset(train_df, feature_extractor)
    val_dataset = prepare_dataset(val_df, feature_extractor)
    
    # Training arguments optimized for RTX 6000 and better recall
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=200,            # Best: 150 epochs
        per_device_train_batch_size=4,   # Best: 4 batch size
        per_device_eval_batch_size=4,    # Match training batch size
        gradient_accumulation_steps=2,   # Best: 2 accumulation steps
        save_strategy="steps",
        save_steps=500,                  # Best: 500 save steps
        logging_dir=f"{output_dir}/logs",
        eval_steps=250,                  # Best: 250 eval steps
        logging_steps=30,
        learning_rate=3e-5,              # Best: 3e-05 learning rate
        save_total_limit=3,
        warmup_ratio=0.05,               # Best: 0.05 warmup ratio
        weight_decay=0.0,                # Best: 0.0 weight decay
        fp16=True,                       # Best: True
        bf16=False,
        gradient_checkpointing=False,    # Best: False
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",  # Focus on F1 instead of accuracy
        # metric_for_best_model="accuracy",
        greater_is_better=True,
        eval_strategy="steps",
        push_to_hub=False,
        save_on_each_node=False,
        disable_tqdm=False,
        max_steps=5000,                  # Best: 5000 max steps
        remove_unused_columns=True,
        # Additional parameters for better convergence
        adam_epsilon=1e-6,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",  # Fixed scheduler type
    )
    
    # Initialize trainer with enhanced focal loss and best early stopping patience
    trainer = SafeSavingTrainer(
        class_weights=class_weights,
        use_focal_loss=True, 
        gamma=3.0,              # Increased gamma for harder examples
        alpha=0.75,             # Higher alpha to focus more on profanity classes
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],  # Best: 20 patience
        data_collator=collate_fn,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(output_dir)
    feature_extractor.save_pretrained(output_dir)
def augment_audio(audio):
    """
    Apply complex augmentation techniques to audio data
    """
    augmented = audio.copy()
    
    # Random combination of augmentations
    augmentation_types = np.random.choice([
        'noise', 'pitch', 'speed', 'reverb', 'time_mask'
    ], size=np.random.randint(1, 4), replace=False)
    
    for aug_type in augmentation_types:
        if aug_type == 'noise':
            # Add different types of noise
            noise_type = np.random.choice(['gaussian', 'pink', 'uniform'])
            if noise_type == 'gaussian':
                noise_level = np.random.uniform(0.001, 0.005)
                noise = np.random.normal(0, noise_level, len(augmented))
            elif noise_type == 'pink':
                noise = np.random.uniform(-0.003, 0.003, len(augmented))
                noise = librosa.core.pink_noise(len(augmented)) * noise_level
            else:  # uniform
                noise = np.random.uniform(-0.002, 0.002, len(augmented))
            augmented += noise
            
        elif aug_type == 'pitch':
            # More varied pitch shifting
            pitch_shift = np.random.uniform(-300, 300)
            augmented = librosa.effects.pitch_shift(
                augmented, 
                sr=16000, 
                n_steps=pitch_shift/100,
                bins_per_octave=200
            )
            
        elif aug_type == 'speed':
            # Time stretching with variable rates
            speed_factor = np.random.uniform(0.8, 1.2)
            augmented = librosa.effects.time_stretch(augmented, rate=speed_factor)
            
        elif aug_type == 'reverb':
            # Add simple reverb effect
            reverb_delay = np.random.randint(1000, 3000)
            decay = np.random.uniform(0.1, 0.5)
            reverb = np.exp(-decay * np.linspace(0, 1, reverb_delay))
            augmented = np.convolve(augmented, reverb, mode='full')[:len(augmented)]
            
        elif aug_type == 'time_mask':
            # Random time masking
            mask_size = int(len(augmented) * np.random.uniform(0.05, 0.15))
            mask_start = np.random.randint(0, len(augmented) - mask_size)
            augmented[mask_start:mask_start + mask_size] = 0
    
    # Normalize after augmentation
    augmented = augmented / (np.max(np.abs(augmented)) + 1e-6)
    return augmented

def augment_short_words(audio, label):
    """
    Enhanced augmentation specifically for short words
    """
    if label in ['กู', 'มึง']:
        augmented = []
        # Create multiple variations
        for _ in range(3):
            aug_audio = audio.copy()
            
            # Apply chain of augmentations
            # 1. Volume variation
            volume_factor = np.random.uniform(0.8, 1.2)
            aug_audio = aug_audio * volume_factor
            
            # 2. Time stretching with controlled range
            stretch_factor = np.random.uniform(0.85, 1.15)
            aug_audio = librosa.effects.time_stretch(aug_audio, rate=stretch_factor)
            
            # 3. Pitch shifting with finer control
            n_steps = np.random.uniform(-2, 2)
            aug_audio = librosa.effects.pitch_shift(
                aug_audio, 
                sr=16000, 
                n_steps=n_steps,
                bins_per_octave=200
            )
            
            # 4. Add subtle background noise
            noise_level = np.random.uniform(0.0005, 0.002)
            noise = np.random.normal(0, noise_level, len(aug_audio))
            aug_audio += noise
            
            # 5. Optional frequency masking
            if np.random.random() < 0.5:
                mask_size = int(len(aug_audio) * np.random.uniform(0.05, 0.1))
                mask_start = np.random.randint(0, len(aug_audio) - mask_size)
                aug_audio[mask_start:mask_start + mask_size] *= np.random.uniform(0.1, 0.3)
            
            # 6. Optional time reversal
            if np.random.random() < 0.2:
                aug_audio = np.flip(aug_audio)
            
            # Normalize
            aug_audio = aug_audio / (np.max(np.abs(aug_audio)) + 1e-6)
            augmented.append(aug_audio)
        
        return augmented
    return [audio]  # Return original if not a short word

def evaluate_model(model, feature_extractor, test_data):
    predictions = []
    labels = []
    
    for item in test_data:
        inputs = feature_extractor(item['audio'], sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1)
            predictions.append(pred.item())
            labels.append(item['label'])
    
    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)
    
    # Calculate per-class metrics
    class_names = ['none', 'เย็ด', 'กู', 'มึง', 'เหี้ย', 'ควย', "สวะ", "หี", 'แตด']
    per_class_metrics = {}
    for i, name in enumerate(class_names):
        class_preds = [p == i for p in predictions]
        class_labels = [l == i for l in labels]
        true_pos = sum(p and l for p, l in zip(class_preds, class_labels))
        total = sum(class_labels)
        if total > 0:
            per_class_metrics[name] = true_pos / total
    
    return {
        'accuracy': accuracy,
        'per_class_metrics': per_class_metrics
    }
def plot_fold_performances(fold_performances):
    """
    Plot accuracy metrics for each fold
    """
    # Prepare data
    folds = list(fold_performances.keys())
    accuracies = [data['accuracy'] for data in fold_performances.values()]
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    bars = plt.bar(folds, accuracies)
    
    # Customize plot
    plt.title('Model Accuracy Across K-Folds', fontsize=14)
    plt.xlabel('Fold Number', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.0)  # Set y-axis from 0 to 1
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.savefig('kfold_accuracy.png')
    plt.close()

def evaluate_all_folds(test_data, num_folds=5, base_dir='./models/fine_tuned_wav2vec2'):
    fold_performances = {}
    best_accuracy = 0
    best_model = None
    best_feature_extractor = None
    
    for fold in range(1, num_folds + 1):
        model_path = f'{base_dir}_fold_{fold}'
        
        # Load model and feature extractor for this fold
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        
        # Evaluate model
        model.eval()
        results = evaluate_model(model, feature_extractor, test_data)
        
        fold_performances[fold] = {
            'accuracy': results['accuracy'],
            'per_class_metrics': results['per_class_metrics'],
            'model_path': model_path
        }
        
        # Track best model
        if results['accuracy'] > best_accuracy:
            best_accuracy = results['accuracy']
            best_model = model
            best_feature_extractor = feature_extractor
            best_fold = fold
        
        print(f"\nFold {fold} Performance:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print("Per-class metrics:", results['per_class_metrics'])
    
    # Plot the fold performances
    plot_fold_performances(fold_performances)
    
    print(f"\nBest performing model is Fold {best_fold}")
    print(f"Path: {fold_performances[best_fold]['model_path']}")
    print(f"Accuracy: {best_accuracy:.4f}")
    
    # Save the best model to a dedicated "best_model" directory
    best_model_dir = f'{base_dir}_best_model'
    best_model.save_pretrained(best_model_dir)
    best_feature_extractor.save_pretrained(best_model_dir)
    print(f"Best model saved to: {best_model_dir}")
    
    return best_model_dir

def run_cross_validation(csv_file, model_name, output_dir, n_splits=5):
    """
    Proper cross-validation with separate test set
    """
    # Load the dataset
    df = load_dataset(csv_file)
    
    # First, split off a test set that won't be used during cross-validation
    train_val_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    print(f"Total dataset: {len(df)} samples")
    print(f"Train+Val set: {len(train_val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Prepare feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        return_attention_mask=True,
        do_normalize=True,
    )
    
    # Prepare the train+val dataset
    train_val_dataset = prepare_dataset(train_val_df, feature_extractor)
    test_dataset = prepare_dataset(test_df, feature_extractor)
    
    print(f"📊 CROSS-VALIDATION SETUP")
    print(f"Train+Val samples: {len(train_val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    train_val_class_counts = Counter([ex['label'] for ex in train_val_dataset])
    test_class_counts = Counter([ex['label'] for ex in test_dataset])
    print(f"Train+Val distribution: {train_val_class_counts}")
    print(f"Test distribution: {test_class_counts}")
    
    # Perform k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = {}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
        print(f"\n{'='*50}")
        print(f"TRAINING FOLD {fold + 1}/{n_splits}")
        print(f"{'='*50}")
        
        # Create train and validation subsets
        train_subset = train_val_dataset.select(train_idx)
        val_subset = train_val_dataset.select(val_idx)
        
        print(f"Fold {fold + 1} - Train: {len(train_subset)}, Val: {len(val_subset)}")
        
        # Train model for this fold
        output_dir_fold = f'{output_dir}_fold_{fold + 1}'
        
        try:
            train_wav2vec2_model_with_data(train_subset, val_subset, model_name, output_dir_fold)
            
            # Evaluate this fold on the validation set
            model = Wav2Vec2ForSequenceClassification.from_pretrained(output_dir_fold)
            model.eval()
            
            # Quick validation evaluation
            val_results = evaluate_dataset(model, feature_extractor, val_subset)
            fold_results[fold + 1] = {
                'val_accuracy': val_results['accuracy'],
                'val_f1_macro': val_results.get('f1_macro', 0),
                'model_path': output_dir_fold
            }
            
            print(f"Fold {fold + 1} Validation Results:")
            print(f"  Accuracy: {val_results['accuracy']:.4f}")
            print(f"  F1 Macro: {val_results.get('f1_macro', 0):.4f}")
            
        except Exception as e:
            print(f"Error training fold {fold + 1}: {e}")
            fold_results[fold + 1] = {
                'val_accuracy': 0.0,
                'val_f1_macro': 0.0,
                'model_path': output_dir_fold,
                'error': str(e)
            }
    
    # Find best fold
    best_fold = max(fold_results.keys(), key=lambda k: fold_results[k]['val_f1_macro'])
    best_model_path = fold_results[best_fold]['model_path']
    
    print(f"\n{'='*50}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*50}")
    
    for fold, results in fold_results.items():
        if 'error' not in results:
            print(f"Fold {fold}: Val Acc={results['val_accuracy']:.4f}, Val F1={results['val_f1_macro']:.4f}")
        else:
            print(f"Fold {fold}: FAILED - {results['error']}")
    
    print(f"\nBest fold: {best_fold}")
    print(f"Best model path: {best_model_path}")
    
    # Final evaluation on test set with best model
    print(f"\n{'='*50}")
    print("FINAL TEST SET EVALUATION")
    print(f"{'='*50}")
    
    try:
        best_model = Wav2Vec2ForSequenceClassification.from_pretrained(best_model_path)
        best_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(best_model_path)
        best_model.eval()
        
        test_results = evaluate_dataset(best_model, best_feature_extractor, test_dataset)
        
        print(f"Test Set Results (Best Model - Fold {best_fold}):")
        print(f"  Accuracy: {test_results['accuracy']:.4f}")
        print(f"  F1 Macro: {test_results.get('f1_macro', 0):.4f}")
        
        # Save best model to final location
        final_model_dir = f'{output_dir}_best_model'
        best_model.save_pretrained(final_model_dir)
        best_feature_extractor.save_pretrained(final_model_dir)
        print(f"Best model saved to: {final_model_dir}")
        
        return {
            'fold_results': fold_results,
            'best_fold': best_fold,
            'best_model_path': final_model_dir,
            'test_results': test_results
        }
        
    except Exception as e:
        print(f"Error evaluating best model: {e}")
        return {
            'fold_results': fold_results,
            'best_fold': best_fold,
            'best_model_path': best_model_path,
            'error': str(e)
        }

def evaluate_dataset(model, feature_extractor, dataset):
    """
    Evaluate model on a dataset
    """
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for example in dataset:
            try:
                # Prepare input
                input_values = torch.tensor(example['input_values']).unsqueeze(0).to(device)
                
                # Forward pass
                outputs = model(input_values=input_values)
                pred = torch.argmax(outputs.logits, dim=-1).cpu().item()
                
                predictions.append(pred)
                true_labels.append(example['label'])
                
            except Exception as e:
                print(f"Error evaluating example: {e}")
                continue
    
    if len(predictions) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0}
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'predictions': predictions,
        'true_labels': true_labels
    }

if __name__ == "__main__":
    csv_file = './csv/train_with_augmentation.csv'  # Use the new balanced dataset
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    output_dir = './models/audio_train_enhanced'
    
    # Run proper cross-validation
    cv_results = run_cross_validation(
        csv_file=csv_file,
        model_name=model_name,
        output_dir=output_dir,
        n_splits=5  # Using 3 folds for faster training
    )
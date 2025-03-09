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
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply class weights to CrossEntropyLoss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": (predictions == labels).astype(np.float32).mean().item()
    }

# Load and preprocess the dataset
def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    return df
# Load and preprocess audio
def preprocess_audio(file_path, start_time, end_time, max_length=16000):
    """Load and preprocess audio segment with Hamming window and noise reduction"""
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
    
    # Apply Hamming window
    window_length = len(audio_np)
    hamming_window = np.hamming(window_length)
    audio_np = audio_np * hamming_window
    
    # Apply pre-emphasis filter to reduce noise
    audio_np = librosa.effects.preemphasis(audio_np)
    
    # Simple noise reduction by removing low amplitude noise
    noise_threshold = 0.005  # Adjust this value based on your needs
    audio_np = np.where(np.abs(audio_np) < noise_threshold, 0, audio_np)
    
    # Convert back to torch tensor
    audio = torch.from_numpy(audio_np).unsqueeze(0)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    
    # Normalize audio
    audio = (audio - audio.mean()) / (audio.std() + 1e-8)
    
    # Pad or truncate to max_length
    if audio.shape[1] < max_length:
        audio = torch.nn.functional.pad(audio, (0, max_length - audio.shape[1]))
    else:
        audio = audio[:, :max_length]
    
    return audio.squeeze().numpy()
# Prepare dataset for Hugging Face Trainer
def prepare_dataset(df, feature_extractor):
    def process_example(example):
        file_path = example['file_path'].replace('\\', '/')
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
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
            padding=True
        )
        
        # Check if label is within the expected range
        label = label_map.get(example['label'])
        if label is None or label < 0 or label >= num_labels:
            print(f"Warning: Label {example['label']} is out of range or not found in label_map for file {file_path}")
            return None
        
        return {
            'input_values': inputs.input_values.squeeze().numpy(),
            'attention_mask': inputs.attention_mask.squeeze().numpy(),
            'label': label
        }
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(process_example, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda example: example is not None)
    
    return dataset
def augment_dataset(dataset, feature_extractor):
    """
    Create augmented versions of the dataset for short words
    """
    augmented_examples = []
    
    for example in dataset:
        if example['label'] in [2, 3]:  # Labels for กู, มึง
            # Create augmented versions
            for _ in range(2):  # Create 2 augmented versions
                # Convert to numpy array if not already
                aug_input = np.array(example['input_values'])
                
                # Apply augmentations
                # Volume variation
                volume_factor = np.random.uniform(0.8, 1.2)
                aug_input = aug_input * volume_factor
                
                # Add slight noise
                noise = np.random.normal(0, 0.001, aug_input.shape)
                aug_input = aug_input + noise
                
                # Time stretching
                stretch_factor = np.random.uniform(0.9, 1.1)
                aug_input = librosa.effects.time_stretch(aug_input, rate=stretch_factor)
                
                # Pitch shifting
                n_steps = np.random.uniform(-1, 1)
                aug_input = librosa.effects.pitch_shift(aug_input, sr=16000, n_steps=n_steps)
                
                # Create new example
                augmented_examples.append({
                    'input_values': aug_input,
                    'attention_mask': example['attention_mask'],
                    'label': example['label']
                })
    
    # Add augmented examples to dataset
    for aug_example in augmented_examples:
        dataset = dataset.add_item(aug_example)
    
    return dataset
def collate_fn(batch):
    input_values = [torch.tensor(item['input_values']).squeeze() for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']).squeeze() for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    input_values = pad_sequence(input_values, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    return {
        'input_values': input_values,
        'attention_mask': attention_mask,
        'labels': labels
    }
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
        finetuning_task="audio-classification"
    )
    
    # Initialize the model with default classifier
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        return_attention_mask=True,
        do_normalize=True,
    )
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_df, feature_extractor)
    val_dataset = prepare_dataset(val_df, feature_extractor)
    
    # Training arguments optimized for RTX 6000
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=100,
        per_device_train_batch_size=16,  # Increased from 4
        per_device_eval_batch_size=16,   # Increased from 4
        gradient_accumulation_steps=1,    # Reduced from 2 since we increased batch size
        save_strategy="steps",
        save_steps=50,                    # More frequent saving
        logging_dir=f"{output_dir}/logs",
        eval_steps=50,                    # More frequent evaluation
        logging_steps=50,
        learning_rate=3e-5,              # Slightly increased learning rate
        save_total_limit=2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,                       # Keep mixed precision training
        bf16=False,                      # RTX 6000 performs better with fp16 than bf16
        gradient_checkpointing=True,     # Enable gradient checkpointing
        dataloader_num_workers=4,        # Parallel data loading
        dataloader_pin_memory=True,      # Pin memory for faster data transfer to GPU
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        eval_strategy="steps",
        push_to_hub=False,
        save_on_each_node=False,
        max_steps=14700,
        disable_tqdm=False,
        remove_unused_columns=True,
    )
    
    # Initialize trainer with class weights
    trainer = CustomTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
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
class ProfanityClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

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

if __name__ == "__main__":
    csv_file = './csv/main.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    output_dir = './models/audio_train'
    
    # Load the dataset first
    df = load_dataset(csv_file)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        return_attention_mask=True,
        do_normalize=True,
    )
    
    # Prepare and augment dataset
    dataset = prepare_dataset(df, feature_extractor)
    dataset = augment_dataset(dataset, feature_extractor)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Training fold {fold + 1}")
        train_subset = dataset.select(train_idx)
        val_subset = dataset.select(val_idx)
        # Train model for this fold
        output_dir_fold = f'{output_dir}_fold_{fold + 1}'
        train_wav2vec2_model(csv_file, model_name, output_dir_fold)
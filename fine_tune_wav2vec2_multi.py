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
from background_removal import remove_background_noise, validate_for_wav2vec2, resample_if_needed
import tempfile

# Define label mapping
label_map = {
    'none': 0,
    'เย็ดแม่': 1,
    'กู': 2,
    'มึง': 3,
    'เหี้ย': 4
}

# Add this after your imports and before other functions
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply class weights
        class_weights = torch.tensor([1.0, 1.0, 1.5, 1.5, 1.0]).to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, 5), labels.view(-1))
        
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
    """Updated to include background noise removal with proper file handling and format conversion"""
    # First, get the sample rate
    metadata = torchaudio.info(file_path)
    sr = metadata.sample_rate
    
    try:
        # Create temporary files with unique names
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            
        cleaned_temp = temp_path.replace('.wav', '_cleaned.wav')
        
        # Load the specific segment
        audio, sr = torchaudio.load(file_path, 
                                  frame_offset=int(start_time * sr), 
                                  num_frames=int((end_time - start_time) * sr))
        
        # Convert to mono first if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample to 16kHz before noise removal
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
            sr = 16000
        
        # Save the preprocessed segment temporarily
        torchaudio.save(temp_path, audio, sr)
        
        # Remove background noise
        remove_background_noise(temp_path, cleaned_temp)
        
        # Load the cleaned audio
        audio, sr = torchaudio.load(cleaned_temp)
        
        # Pad or truncate to max_length
        if audio.shape[1] < max_length:
            audio = torch.nn.functional.pad(audio, (0, max_length - audio.shape[1]))
        else:
            audio = audio[:, :max_length]
        
        return audio.squeeze().numpy()
        
    finally:
        # Clean up temporary files in finally block to ensure cleanup
        try:
            if os.path.exists(temp_path):
                os.close(os.open(temp_path, os.O_RDONLY))  # Close any open handles
                os.unlink(temp_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file {temp_path}: {e}")
            
        try:
            if os.path.exists(cleaned_temp):
                os.close(os.open(cleaned_temp, os.O_RDONLY))  # Close any open handles
                os.unlink(cleaned_temp)
        except Exception as e:
            print(f"Warning: Could not delete temporary file {cleaned_temp}: {e}")
# Prepare dataset for Hugging Face Trainer
def prepare_dataset(df, feature_extractor):
    def process_example(example):
        file_path = example['File Name'].replace('\\', '/')
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        # Get audio length first
        metadata = torchaudio.info(file_path)
        audio_length = metadata.num_frames / metadata.sample_rate
            
        # Add padding around the profanity segments
        padding = 0.2  # Increase from 0.1 to 0.2 seconds
        if example['Label'] in ['กู', 'มึง']:  # Special handling for short words
            padding = 0.3  # Even more context for single-syllable words
        start_time = max(0, example['Start Time (s)'] - padding)
        end_time = min(example['End Time (s)'] + padding, audio_length)
        
        # Process audio with background noise removal
        audio = preprocess_audio(file_path, start_time, end_time)
        
        # Apply feature extractor
        inputs = feature_extractor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        return {
            'input_values': inputs.input_values.squeeze().numpy(),
            'attention_mask': inputs.attention_mask.squeeze().numpy(),
            'label': label_map[example['Label']]
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
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])
    
    # Load pre-trained model and feature extractor
    config = Wav2Vec2Config.from_pretrained(
        model_name,
        num_labels=5,
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
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=100,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        eval_steps=100,
        logging_steps=100,
        learning_rate=2e-5,
        warmup_ratio=0.15,
        weight_decay=0.02,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        evaluation_strategy="steps",
    )
    
    # Initialize trainer with custom trainer class
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=collate_fn,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(output_dir)
    feature_extractor.save_pretrained(output_dir)
def augment_audio(audio):
    augmented = audio.copy()
    # Add random noise
    noise_level = np.random.uniform(0.001, 0.005)
    noise = np.random.normal(0, noise_level, len(audio))
    augmented += noise
    
    # Random time shift
    shift = np.random.randint(-1000, 1000)
    augmented = np.roll(augmented, shift)
    
    # Random pitch shift
    pitch_shift = np.random.uniform(-100, 100)
    augmented = librosa.effects.pitch_shift(augmented, sr=16000, n_steps=pitch_shift)
    
    return augmented
def augment_short_words(audio, label):
    """
    Special augmentation for short words
    """
    if label in ['กู', 'มึง']:
        augmented = []
        # Create multiple variations
        for _ in range(3):  # Create 3 variations
            aug_audio = audio.copy()
            
            # Slight volume variations
            volume_factor = np.random.uniform(0.8, 1.2)
            aug_audio = aug_audio * volume_factor
            
            # Small time stretching
            stretch_factor = np.random.uniform(0.9, 1.1)
            aug_audio = librosa.effects.time_stretch(aug_audio, rate=stretch_factor)
            
            # Slight pitch variations
            n_steps = np.random.uniform(-1, 1)
            aug_audio = librosa.effects.pitch_shift(aug_audio, sr=16000, n_steps=n_steps)
            
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
def visualize_k_fold_results(fold_metrics):
    """
    Visualize results across different folds
    """
    # Extract metrics
    accuracies = [m['eval_accuracy'] for m in fold_metrics]
    precisions = [m['eval_precision'] for m in fold_metrics]
    recalls = [m['eval_recall'] for m in fold_metrics]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    folds = range(1, len(fold_metrics) + 1)
    
    plt.plot(folds, accuracies, 'o-', label='Accuracy')
    plt.plot(folds, precisions, 's-', label='Precision')
    plt.plot(folds, recalls, '^-', label='Recall')
    
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Model Performance Across Folds')
    plt.legend()
    plt.grid(True)
    
    plt.show()
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
    class_names = ['none', 'เย็ดม่', 'กู', 'มึง', 'เหี้ย']
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
def evaluate_all_folds(test_data, num_folds=5, base_dir='./models/fine_tuned_wav2vec2'):
    fold_performances = {}
    
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
        
        print(f"\nFold {fold} Performance:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print("Per-class metrics:", results['per_class_metrics'])
    
    # Find best performing fold
    best_fold = max(fold_performances.items(), 
                   key=lambda x: x[1]['accuracy'])
    
    print(f"\nBest performing model is Fold {best_fold[0]}")
    print(f"Path: {best_fold[1]['model_path']}")
    print(f"Accuracy: {best_fold[1]['accuracy']:.4f}")
    
    return best_fold[1]['model_path']

if __name__ == "__main__":
    csv_file = 'profanity_dataset_word.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    output_dir = './models/fine_tuned_wav2vec2'
    
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
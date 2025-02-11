import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Config
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import EarlyStoppingCallback
import torch.nn as nn
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

# Define label mapping with unified เหี้ย label
label_map = {
    'กู': 0,
    'ควย': 1,
    'มึง': 2,
    'สวะ': 3,
    'หี': 4,
    'เย็ด': 5,
    'เหี้ย': 6,
    'แตด': 7
}

# Define the number of labels
num_labels = len(label_map)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply class weights
        class_weights = torch.tensor([1.0] * num_labels).to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": (predictions == labels).astype(np.float32).mean().item()
    }

# Load and preprocess audio
def preprocess_audio(file_path, max_length=16000):
    """Load and preprocess audio with background noise removal"""
    # Load the audio file
    audio, sr = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    
    # Convert to numpy for noise reduction
    audio_np = audio.squeeze().numpy()
    
    # Noise reduction
    # Calculate noise profile from the first 1000 samples (assumed to be noise)
    noise_sample = audio_np[:1000]
    noise_profile = np.mean(np.abs(noise_sample))
    
    # Apply noise gate
    threshold = noise_profile * 2  # Adjust this multiplier as needed
    audio_denoised = audio_np.copy()
    audio_denoised[np.abs(audio_denoised) < threshold] = 0
    
    # Additional noise reduction using spectral gating
    # Convert to spectrogram
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(audio_denoised, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise spectrum
    noise_magnitude = np.mean(magnitude[:, :10], axis=1, keepdims=True)
    
    # Apply spectral subtraction
    gain = 1.0  # Adjust this value to control noise reduction strength
    magnitude_reduced = np.maximum(0, magnitude - gain * noise_magnitude)
    
    # Reconstruct signal
    stft_reduced = magnitude_reduced * np.exp(1j * phase)
    audio_denoised = librosa.istft(stft_reduced, hop_length=hop_length)
    
    # Convert back to torch tensor
    audio_denoised = torch.from_numpy(audio_denoised).float()
    
    # Pad or truncate to max_length
    if audio_denoised.shape[0] < max_length:
        audio_denoised = torch.nn.functional.pad(audio_denoised, (0, max_length - audio_denoised.shape[0]))
    else:
        audio_denoised = audio_denoised[:max_length]
    
    return audio_denoised.numpy()

def augment_dataset(dataset, feature_extractor):
    """
    Create augmented versions of the dataset for short words
    """
    augmented_examples = []
    
    for example in dataset:
        if example['label'] in [0, 2]:  # Labels for กู, มึง (short words)
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
    input_values = [torch.tensor(item['input_values']) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    
    # Ensure all tensors are on CPU and properly shaped
    input_values = pad_sequence(input_values, batch_first=True, padding_value=0.0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0.0)
    
    # Ensure proper dimensions
    if len(input_values.shape) == 1:
        input_values = input_values.unsqueeze(0)
    if len(attention_mask.shape) == 1:
        attention_mask = attention_mask.unsqueeze(0)
    
    # Ensure all tensors are contiguous
    input_values = input_values.contiguous()
    attention_mask = attention_mask.contiguous()
    
    return {
        'input_values': input_values,
        'attention_mask': attention_mask,
        'labels': labels
    }

def prepare_dataset(df, feature_extractor):
    """Prepare dataset with better error handling"""
    # Unify the เหี้ย labels first
    df['label'] = df['label'].replace('เหี้͏ย', 'เหี้ย')
    
    # Create a list to store valid examples
    valid_examples = []
    
    for _, row in df.iterrows():
        file_path = row['file_path']
        
        if not os.path.exists(file_path):
            print(f"Skipping missing file: {file_path}")
            continue
        
        try:
            # Process audio
            audio = preprocess_audio(file_path)
            
            # Ensure audio is 1D
            if len(audio.shape) > 1:
                audio = audio.squeeze()
            
            # Ensure audio is the correct type
            audio = audio.astype(np.float32)
            
            # Apply feature extractor
            inputs = feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            # Get label from label_map
            label = label_map.get(row['label'])
            if label is None:
                print(f"Warning: Label {row['label']} not found in label_map")
                continue
            
            # Add valid example to list
            valid_examples.append({
                'input_values': inputs.input_values.squeeze().numpy(),
                'attention_mask': inputs.attention_mask.squeeze().numpy(),
                'label': label
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Create dataset from valid examples
    if not valid_examples:
        raise ValueError("No valid examples found in the dataset!")
    
    dataset = Dataset.from_list(valid_examples)
    return dataset

def plot_fold_performances(fold_performances):
    """
    Plot accuracy metrics for each fold
    """
    plt.figure(figsize=(10, 6))
    
    # Create bar plot of accuracies
    folds = list(range(1, len(fold_performances) + 1))
    accuracies = [perf['best_accuracy'] for perf in fold_performances]
    
    plt.bar(folds, accuracies)
    plt.title('Model Performance Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    
    # Add value labels on top of bars
    for i, acc in enumerate(accuracies, 1):
        plt.text(i, acc, f'{acc:.4f}', ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, 'fold_performances.png'))
    plt.close()

def train_model(csv_file, model_name, output_dir):
    # Load dataset
    df = pd.read_csv(csv_file)
    
    # Print initial class distribution
    print("Overall class distribution:")
    print(df['label'].value_counts())
    
    # Initialize K-fold
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store performance metrics for each fold
    fold_performances = []
    
    # Create base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform K-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
        print(f"\nTraining Fold {fold}")
        
        # Split data for this fold
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Print fold-specific class distribution
        print(f"\nFold {fold} training class distribution:")
        print(train_df['label'].value_counts())
        
        # Create fold-specific output directory
        fold_output_dir = os.path.join(output_dir, f'fold_{fold}')
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Initialize model and feature extractor
        config = Wav2Vec2Config.from_pretrained(
            model_name,
            num_labels=num_labels,
            finetuning_task="audio-classification",
            mask_time_prob=0.0,
            mask_feature_prob=0.0,
            gradient_checkpointing=False
        )
        
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
        
        # Training arguments with more disk-efficient settings
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=100,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            evaluation_strategy="epoch",     # Changed from "steps" to "epoch"
            save_strategy="epoch",           # Changed from "steps" to "epoch"
            save_steps=500,
            eval_steps=100,
            logging_steps=100,
            learning_rate=1e-4,
            save_total_limit=1,             # Reduced to only keep the best model
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            fp16=True,
            warmup_ratio=0.1,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            no_cuda=False,
            seed=42,
            data_seed=42,
            max_grad_norm=1.0,
            save_safetensors=True,
            push_to_hub=False,
            overwrite_output_dir=True
        )
        
        try:
            # Initialize trainer
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                data_collator=collate_fn,
            )
            
            # Clean up previous fold's files before training
            if os.path.exists(fold_output_dir):
                for file in os.listdir(fold_output_dir):
                    file_path = os.path.join(fold_output_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
            
            # Train the model
            train_result = trainer.train()
            
            # Evaluate the model
            eval_result = trainer.evaluate()
            
            # Store fold performance
            fold_performances.append({
                'fold': fold,
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result['eval_loss'],
                'best_accuracy': eval_result['eval_accuracy']
            })
            
            # Save only the best model for this fold
            try:
                # Clean up checkpoint files first
                checkpoint_dirs = [d for d in os.listdir(fold_output_dir) if d.startswith('checkpoint-')]
                for d in checkpoint_dirs:
                    shutil.rmtree(os.path.join(fold_output_dir, d))
                
                # Save the final model
                trainer.save_model(fold_output_dir)
                feature_extractor.save_pretrained(fold_output_dir)
                
                # Save fold metrics
                with open(os.path.join(fold_output_dir, 'metrics.txt'), 'w') as f:
                    f.write(f"Training Loss: {train_result.training_loss}\n")
                    f.write(f"Evaluation Loss: {eval_result['eval_loss']}\n")
                    f.write(f"Accuracy: {eval_result['eval_accuracy']}\n")
            
            except Exception as e:
                print(f"Warning: Error saving model for fold {fold}: {str(e)}")
            
        except Exception as e:
            print(f"Error in fold {fold}: {str(e)}")
            fold_performances.append({
                'fold': fold,
                'train_loss': float('inf'),
                'eval_loss': float('inf'),
                'best_accuracy': 0.0
            })
            
            # Clean up failed fold's directory
            if os.path.exists(fold_output_dir):
                shutil.rmtree(fold_output_dir)
    
    # Plot and save fold performances
    plot_fold_performances(fold_performances)
    
    # Find best performing fold
    best_fold = max(fold_performances, key=lambda x: x['best_accuracy'])
    print(f"\nBest performing fold: {best_fold['fold']}")
    print(f"Best accuracy: {best_fold['best_accuracy']:.4f}")
    
    # Save overall results
    with open(os.path.join(output_dir, 'overall_results.txt'), 'w') as f:
        f.write("Fold Performance Summary:\n")
        for perf in fold_performances:
            f.write(f"\nFold {perf['fold']}:\n")
            f.write(f"Training Loss: {perf['train_loss']}\n")
            f.write(f"Evaluation Loss: {perf['eval_loss']}\n")
            f.write(f"Accuracy: {perf['best_accuracy']}\n")
        f.write(f"\nBest Fold: {best_fold['fold']}\n")
        f.write(f"Best Accuracy: {best_fold['best_accuracy']:.4f}\n")

if __name__ == "__main__":
    csv_file = './dataset_one.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    output_dir = './models/wav2vec2_one'
    
    train_model(csv_file, model_name, output_dir)

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

# Load and preprocess the dataset
def load_dataset(csv_file):
    df = pd.read_csv(csv_file)
    return df

# Load and preprocess audio
def preprocess_audio(file_path, start_time, end_time, max_length=16000):
    # First, get the sample rate
    metadata = torchaudio.info(file_path)
    sr = metadata.sample_rate

    # Now load the audio with the correct frame offset and number of frames
    audio, sr = torchaudio.load(file_path, 
                                frame_offset=int(start_time * sr), 
                                num_frames=int((end_time - start_time) * sr))
    
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Pad or truncate to max_length
    if audio.shape[1] < max_length:
        audio = torch.nn.functional.pad(audio, (0, max_length - audio.shape[1]))
    else:
        audio = audio[:, :max_length]
    
    return audio.squeeze().numpy()

# Prepare dataset for Hugging Face Trainer
def prepare_dataset(df, feature_extractor):
    # First, filter out rows with missing files
    valid_files = []
    for file_path in df['File Name'].unique():
        file_path = file_path.replace('\\', '/')
        if os.path.exists(file_path):
            valid_files.append(file_path)
        else:
            print(f"Skipping missing file: {file_path}")
    
    # Filter DataFrame to only include existing files
    df = df[df['File Name'].apply(lambda x: x.replace('\\', '/') in valid_files)].reset_index(drop=True)
    
    def process_example(example):
        try:
            file_path = example['File Name'].replace('\\', '/')
            audio = preprocess_audio(file_path, example['Start Time (s)'], example['End Time (s)'])
            print(f"Audio shape after preprocessing: {audio.shape}")
            
            # Apply feature extractor
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            print(f"Input values shape after feature extractor: {inputs.input_values.shape}")
            
            # Generate attention mask if it's not provided
            if 'attention_mask' not in inputs:
                attention_mask = torch.ones_like(inputs.input_values)
                print(f"Generated attention mask shape: {attention_mask.shape}")
            else:
                attention_mask = inputs.attention_mask
                print(f"Attention mask shape from feature extractor: {attention_mask.shape}")
            
            return {
                'input_values': inputs.input_values.squeeze().numpy(),
                'attention_mask': attention_mask.squeeze().numpy(),
                'label': 1 if example['Label'] == 'profanity' else 0
            }
        except Exception as e:
            print(f"Error processing example: {str(e)}")
            # Return a dummy example that will be filtered out
            return {
                'input_values': np.zeros(1),  # dummy array
                'attention_mask': np.zeros(1),  # dummy array
                'label': -1  # invalid label
            }

    # Convert DataFrame to Dataset
    dataset = Dataset.from_pandas(df)
    
    # Process the dataset
    processed_dataset = dataset.map(
        process_example,
        remove_columns=dataset.column_names,
        num_proc=1  # Set to 1 to avoid multiprocessing issues
    )
    
    # Filter out failed examples (those with label == -1)
    processed_dataset = processed_dataset.filter(lambda x: x['label'] != -1)
    
    if len(processed_dataset) == 0:
        raise ValueError("No valid examples found in the dataset after processing.")
        
    print(f"Successfully processed {len(processed_dataset)} examples")
    return processed_dataset

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
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Load pre-trained model and feature extractor
    config = Wav2Vec2Config.from_pretrained(
        model_name,
        num_labels=2,
        finetuning_task="audio-classification",
        pad_token_id=0,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        gradient_checkpointing=True,
        layerdrop=0.1,
    )

    # Initialize feature extractor first
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        return_attention_mask=True,
        do_normalize=True,
        padding=True,
        sampling_rate=16000,
    )

    # Initialize the model with the correct architecture
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Freeze feature encoder layers (optional, but can help with fine-tuning)
    if hasattr(model, 'wav2vec2'):
        for param in model.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False

    # Prepare datasets
    train_dataset = prepare_dataset(train_df, feature_extractor)
    val_dataset = prepare_dataset(val_df, feature_extractor)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_dir='./logs',
        learning_rate=2e-5,
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        gradient_accumulation_steps=2,
        logging_steps=10,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # Define Trainer
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=[early_stopping_callback],
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate accuracy
    accuracy = (predictions == labels).astype(np.float32).mean().item()
    
    # Calculate precision, recall, and F1 for the profanity class (assuming it's label 1)
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

if __name__ == "__main__":
    csv_file = 'profanity_dataset.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"  # Thai pre-trained model
    output_dir = './models/fine_tuned_wav2vec2'

    train_wav2vec2_model(csv_file, model_name, output_dir)

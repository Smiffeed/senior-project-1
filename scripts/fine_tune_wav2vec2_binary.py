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
    def process_example(example):
        file_path = example['File Name'].replace('\\', '/')
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
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
            'input_values': inputs.input_values.squeeze(),
            'attention_mask': attention_mask.squeeze(),
            'label': 1 if example['Label'] == 'เย็ดแม่' else 0
        }
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(process_example, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda example: example is not None)
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
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    # Load pre-trained model and feature extractor
    config = Wav2Vec2Config.from_pretrained(
        model_name,
        num_labels=2,
        finetuning_task="audio-classification"
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
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=100,  # Increase number of epochs
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_dir='./logs',
        learning_rate=1e-4,  # Adjust learning rate
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        warmup_steps=500,  # Add warmup steps
        weight_decay=0.01,  # Add weight decay
    )
    # Define Trainer
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,  # Add this function
        data_collator=collate_fn,
        callbacks=[early_stopping_callback],  # Add this line
    )
    # Train the model
    trainer.train()
    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}
if __name__ == "__main__":
    csv_file = 'profanity_dataset_p1.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"  # Thai pre-trained model
    output_dir = './models/fine_tuned_wav2vec2'
    train_wav2vec2_model(csv_file, model_name, output_dir)
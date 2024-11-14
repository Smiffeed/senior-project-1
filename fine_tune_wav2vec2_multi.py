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
        
        # Apply feature extractor
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Map labels to integers
        label_map = {
            'none': 0,
            'เย็ดแม่': 1,
            'กู': 2,
            'มึง': 3,
            'เหี้ย': 4
        }
        
        return {
            'input_values': inputs.input_values.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'label': label_map[example['Label']]
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
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])
    # Load pre-trained model and feature extractor
    config = Wav2Vec2Config.from_pretrained(
        model_name,
        num_labels=5,  # Update number of labels to 5
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
        num_train_epochs=100,  # Reduced epochs since we have more data
        per_device_train_batch_size=8,  # Increased batch size
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_dir='./logs',
        learning_rate=2e-5,  # Reduced learning rate
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        warmup_steps=200,
        weight_decay=0.01,
        gradient_accumulation_steps=2,  # Added gradient accumulation
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
    accuracy = (predictions == labels).astype(np.float32).mean().item()
    
    # Calculate per-class accuracy
    class_names = ['none', 'เย็ดแม่', 'กู', 'มึง', 'เหี้ย']
    per_class_accuracy = {}
    
    for i, class_name in enumerate(class_names):
        class_mask = labels == i
        if class_mask.sum() > 0:
            class_accuracy = ((predictions == labels) & class_mask).sum() / class_mask.sum()
            per_class_accuracy[f"{class_name}_accuracy"] = class_accuracy
    
    metrics = {"accuracy": accuracy, **per_class_accuracy}
    return metrics
if __name__ == "__main__":
    csv_file = 'profanity_dataset_word.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"  # Thai pre-trained model
    output_dir = './models/fine_tuned_wav2vec2'
    train_wav2vec2_model(csv_file, model_name, output_dir)
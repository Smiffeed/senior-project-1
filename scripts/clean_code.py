import pandas as pd
import numpy as np
import torch
import torchaudio
from torch import nn
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Config, EarlyStoppingCallback, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import os

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    label_map = {
        'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4,
        'ควย': 5, 'สวะ': 6, 'หี': 7, 'แตด': 8
    }
    num_labels = len(label_map)

    df = pd.read_csv('csv/main.csv')
    print(f"Loaded dataset with {len(df)} samples")

    # CREATE FEATURE EXTRACTOR FIRST!
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        return_attention_mask=True,
        do_normalize=True,
    )
    print("Feature extractor loaded!")

    # Simplified preprocessing - let feature extractor handle everything
    def preprocess_row(row, feature_extractor, target_sr=16000):
        # Load and extract segment
        waveform, sample_rate = torchaudio.load(row['file_path'])
        start_sample = int(row['start_time'] * sample_rate)
        end_sample = int(row['end_time'] * sample_rate)
        segment = waveform[:, start_sample:end_sample]
        
        # Convert to mono
        if segment.shape[0] > 1:
            segment = segment.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            segment = resampler(segment)
        
        # Convert to numpy
        audio_array = segment.squeeze().numpy()
        
        # Use feature extractor WITHOUT max_length (let it decide naturally)
        inputs = feature_extractor(
            audio_array,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True  # No truncation, no max_length
        )
        
        return {
            'input_values': inputs.input_values.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze() if 'attention_mask' in inputs else None,
            'labels': label_map[row['label']]
        }

    # Test preprocessing (CORRECT way)
    print("Testing preprocessing...")
    result = preprocess_row(df.iloc[0], feature_extractor)
    print(f"Input shape: {result['input_values'].shape}")
    print(f"Label: {result['labels']}")
    
    # Process all rows
    print("Processing all data...")
    processed = [preprocess_row(row, feature_extractor) for _, row in df.iterrows()]
    
    # Check length variation
    lengths = [len(p['input_values']) for p in processed]
    print(f"Audio lengths - Min: {min(lengths)}, Max: {max(lengths)}, Std: {np.std(lengths):.0f}")

    # Add this after preprocessing to filter extreme lengths
    def filter_extreme_lengths(processed, min_length=1600, max_length=160000):  # 0.1s to 10s
        filtered = []
        for item in processed:
            length = len(item['input_values'])
            if min_length <= length <= max_length:
                filtered.append(item)
            else:
                print(f"Filtered out sample with length {length}")
        return filtered

    # Apply filtering
    processed = filter_extreme_lengths(processed)
    print(f"After filtering: {len(processed)} samples remain")

    # Custom Dataset that handles variable lengths
    from torch.utils.data import Dataset
    
    class VariableLengthAudioDataset(Dataset):
        def __init__(self, processed_data):
            self.data = processed_data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Custom collator for variable lengths
    def collate_fn(batch):
        # Extract all input_values and find max length
        input_values = [item['input_values'] for item in batch]
        labels = torch.tensor([item['labels'] for item in batch])
        
        # Pad to max length in this batch
        max_len = max(len(iv) for iv in input_values)
        padded_inputs = []
        attention_masks = []
        
        for iv in input_values:
            # Pad with zeros
            padded = torch.zeros(max_len)
            padded[:len(iv)] = iv
            padded_inputs.append(padded)
            
            # Create attention mask (1 for real data, 0 for padding)
            mask = torch.zeros(max_len)
            mask[:len(iv)] = 1
            attention_masks.append(mask)
        
        return {
            'input_values': torch.stack(padded_inputs),
            'attention_mask': torch.stack(attention_masks),
            'labels': labels
        }

    dataset = VariableLengthAudioDataset(processed)
    print(f"Dataset created with {len(dataset)} samples")

    # Calculate class weights
    labels = [p['labels'] for p in processed]
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_weights = 1.0 - (counts / len(labels))
    class_weights = torch.from_numpy(class_weights).float().to('cuda')
    
    print("Class distribution:", dict(zip(unique_labels, counts)))

    class weightedLoss(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs)
            logits = outputs.get("logits")
            labels = inputs.get("labels")
            loss_func = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_func(logits.view(-1, num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='weighted'),
            'precision': precision_score(labels, preds, average='weighted'),
            'recall': recall_score(labels, preds, average='weighted')
        }

    # Training setup - REMOVE dataloader_collate_fn
    batch_size = 8  # Smaller batch for variable lengths
    training_args = TrainingArguments(
        output_dir='./output',
        learning_rate=2e-5,
        num_train_epochs=100,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        weight_decay=0.01,
        logging_steps=len(dataset) // batch_size,
        save_total_limit=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        fp16=True,
        remove_unused_columns=False,
        report_to=None,
        disable_tqdm=False,
        log_level="info",
        logging_first_step=True,
    )

    # Split dataset
    indices = list(range(len(dataset)))
    train_idx, eval_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    eval_dataset = torch.utils.data.Subset(dataset, eval_idx)

    # Load model
    config = Wav2Vec2Config.from_pretrained(model_name, num_labels=num_labels)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, config=config)

    # Create trainer - ADD data_collator here
    trainer = weightedLoss(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,  # ← ADD THIS LINE
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)]
    )

    # Memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    print("Starting training...")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Total epochs: 100 (with early stopping)")
    print(f"Steps per epoch: {len(train_dataset) // batch_size}")

    # Train
    trainer.train()

    # Save
    trainer.save_model("./output/model")
    feature_extractor.save_pretrained("./output/model")
    print("Training completed!")
    print("Model and feature extractor saved to: ./output/model")
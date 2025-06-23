import pandas as pd
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor, EarlyStoppingCallback, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb  # For experiment tracking
from improved_model import create_improved_model
from data_augmentation import balance_dataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Enable W&B logging - comment out if not using W&B
# wandb.init(project="thai-profanity-detection", name="improved_model_v1")

# Define the dataset class outside the main block
class VariableLengthAudioDataset(Dataset):
    def __init__(self, processed_data):
        self.data = processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Define collate function outside the main block
def collate_fn(batch):
    # Extract all input_values and find max length
    input_values = [item['input_values'] for item in batch]
    labels = torch.tensor([item['labels'] for item in batch])
    
    # Pad to max length in this batch (but with an upper limit)
    max_len = min(max(len(iv) for iv in input_values), 160000)  # Upper limit to prevent memory issues
    padded_inputs = []
    attention_masks = []
    
    for iv in input_values:
        # Pad with zeros
        padded = torch.zeros(max_len)
        seq_len = min(len(iv), max_len)  # Truncate if needed
        padded[:seq_len] = iv[:seq_len]
        padded_inputs.append(padded)
        
        # Create attention mask (1 for real data, 0 for padding)
        mask = torch.zeros(max_len)
        mask[:seq_len] = 1
        attention_masks.append(mask)
    
    return {
        'input_values': torch.stack(padded_inputs),
        'attention_mask': torch.stack(attention_masks),
        'labels': labels
    }

# Define compute_metrics function outside the main block
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Create class-wise metrics
    class_metrics = {}
    for i in range(len(np.unique(labels))):
        class_name = str(i)  # We'll map these later
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': cm[i, :].sum()
        }
    
    # Aggregate metrics
    all_metrics = {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted'),
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(labels, preds, average='weighted'),
        'class_metrics': class_metrics
    }
    
    return all_metrics

# Update the WeightedTrainer class to match the newer transformers API
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs[0]
        
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    # 1. Configuration
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    num_epochs = 100
    batch_size = 8
    learning_rate = 2e-5
    early_stopping_patience = 5
    gradient_accumulation_steps = 2  # To simulate larger batch sizes
    weight_decay = 0.01
    warmup_ratio = 0.1
    audio_max_length_seconds = 5
    target_sr = 16000
    use_data_augmentation = True
    use_staged_training = True
    use_fp16 = True
    
    label_map = {
        'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4,
        'ควย': 5, 'สวะ': 6, 'หี': 7, 'แตด': 8
    }
    inv_label_map = {v: k for k, v in label_map.items()}
    num_labels = len(label_map)
    
    print("Loading dataset...")
    df = pd.read_csv('csv/main.csv')
    
    # 2. Data Augmentation 
    if use_data_augmentation:
        # You can run data_augmentation.py separately or use it here
        print("Balancing dataset with augmentation...")
        df = balance_dataset(df, label_column='label')
    
    print(f"Dataset loaded with {len(df)} samples")
    
    # 3. Setup feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        return_attention_mask=True,
        do_normalize=True,
    )
    print("Feature extractor loaded!")
    
    # 4. Improved preprocessing function
    def preprocess_row(row, feature_extractor, target_sr=16000, max_length_samples=None):
        # Load and extract segment
        try:
            waveform, sample_rate = torchaudio.load(row['file_path'])
            
            # Check if waveform is empty or too short
            if waveform.numel() == 0 or waveform.shape[1] < 100:  # Minimum reasonable length
                print(f"Skipping empty or too short audio: {row['file_path']}")
                return None
                
            start_sample = int(row['start_time'] * sample_rate) if 'start_time' in row else 0
            end_sample = int(row['end_time'] * sample_rate) if 'end_time' in row else waveform.shape[1]
            
            # Validate segment boundaries
            if start_sample >= waveform.shape[1] or end_sample <= start_sample:
                print(f"Invalid segment boundaries for {row['file_path']}")
                return None
                
            # Ensure end_sample doesn't exceed waveform length
            end_sample = min(end_sample, waveform.shape[1])
            segment = waveform[:, start_sample:end_sample]
            
            # Convert to mono
            if segment.shape[0] > 1:
                segment = segment.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
                segment = resampler(segment)
            
            # Apply peak normalization for better volume consistency
            if segment.numel() > 0:  # Check again after processing
                segment = segment / (torch.max(torch.abs(segment)) + 1e-8)
            else:
                print(f"Empty segment after processing: {row['file_path']}")
                return None
                
            # Convert to numpy
            audio_array = segment.squeeze().numpy()
            
            # Use feature extractor
            inputs = feature_extractor(
                audio_array,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True
            )
            
            return {
                'input_values': inputs.input_values.squeeze(),
                'attention_mask': inputs.attention_mask.squeeze() if 'attention_mask' in inputs else None,
                'labels': label_map[row['label']]
            }
        except Exception as e:
            print(f"Error processing {row['file_path']}: {e}")
            return None
    
    # 5. Process all rows with progress bar
    print("Processing all data...")
    processed = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        result = preprocess_row(row, feature_extractor)
        if result is not None:
            processed.append(result)
    
    # Check length variation
    lengths = [len(p['input_values']) for p in processed]
    print(f"Audio lengths - Min: {min(lengths)}, Max: {max(lengths)}, Std: {np.std(lengths):.0f}")
    
    # 6. Add improved filtering
    # Improved filtering function
    def filter_extreme_lengths(processed, min_length=1600, max_length=160000):
        """Filter out samples with extreme lengths and ensure all have valid shapes"""
        filtered = []
        for item in processed:
            # Skip None items
            if item is None:
                continue
                
            # Check if input_values exists and has reasonable length
            if 'input_values' not in item or not isinstance(item['input_values'], torch.Tensor):
                print(f"Skipping item with missing or invalid input_values")
                continue
                
            length = len(item['input_values'])
            if min_length <= length <= max_length:
                filtered.append(item)
            else:
                print(f"Filtered out sample with length {length}")
                
        return filtered
    
    # Apply filtering
    processed = filter_extreme_lengths(processed)
    print(f"After filtering: {len(processed)} samples remain")
    
    # 7. Calculate class weights for balanced loss
    labels = [p['labels'] for p in processed]
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=labels)
    class_weights = torch.from_numpy(class_weights).float().to('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Class weights:", dict(zip(unique_labels, class_weights.cpu().numpy())))
    
    # 8. Dataset implementation
    
    # 9. Improved collator with better padding
    # 10. Create dataset and splits
    dataset = VariableLengthAudioDataset(processed)
    print(f"Dataset created with {len(dataset)} samples")
    
    # Better stratified split
    indices = list(range(len(dataset)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, 
                                          stratify=[processed[i]['labels'] for i in indices])
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42,
                                         stratify=[processed[i]['labels'] for i in temp_idx])
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
    # 11. Improved weighted loss trainer
    # 12. Enhanced metrics calculation
    # 13. Improved training configuration
    # Update these parameters to improve stability
    training_args = TrainingArguments(
        output_dir='./output/improved_model',
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_checkpointing=False,  # Keep this disabled
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        dataloader_pin_memory=True,
        weight_decay=weight_decay,
        logging_steps=100,
        eval_steps=100,
        save_total_limit=3,
        eval_strategy='steps',
        save_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        warmup_ratio=warmup_ratio,
        fp16=use_fp16,
        remove_unused_columns=False,
        report_to="none",
        logging_first_step=True,
    )
    
    # 14. Create improved model
    print("Creating model...")
    model = create_improved_model(model_name, num_labels, use_staged_training)

    # 15. Create trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience, early_stopping_threshold=0.001)]
    )
    
    # 16. Memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    
    # 17. Training details
    print("\n=== Training Configuration ===")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Batch size: {batch_size} (effective: {batch_size * gradient_accumulation_steps})")
    print(f"Learning rate: {learning_rate}")
    print(f"Total epochs: {num_epochs} (with early stopping patience: {early_stopping_patience})")
    print(f"Steps per epoch: {len(train_dataset) // batch_size}")
    print(f"Model: {model_name}")
    print(f"Using FP16: {use_fp16}")
    print(f"Using staged training: {use_staged_training}")
    print("============================\n")
    
    # 18. First stage training
    print("Starting first training stage...")
    trainer.train()
    
    # 19. Second stage training (if using staged training)
    if use_staged_training:
        # Switch to second stage
        model.start_second_stage()
        
        # Update learning rate for fine-tuning
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = learning_rate / 10
        
        print("Starting second training stage with full model fine-tuning...")
        trainer.train()
    
    # 20. Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_dataset)
    print("Test metrics:", test_metrics)
    
    # 21. Save model and feature extractor
    print("Saving model...")
    trainer.save_model("./output/improved_model/final")
    feature_extractor.save_pretrained("./output/improved_model/final")
    
    # 22. Save test metrics to file
    with open("./output/improved_model/test_metrics.txt", "w") as f:
        for k, v in test_metrics.items():
            if k != 'class_metrics':
                f.write(f"{k}: {v}\n")
        
        f.write("\n=== Class-wise Metrics ===\n")
        for class_name, metrics in test_metrics.get('class_metrics', {}).items():
            f.write(f"{class_name}:\n")
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name}: {value:.4f}\n")
    
    print("Training completed!")
    print("Model and feature extractor saved to: ./output/improved_model/final")

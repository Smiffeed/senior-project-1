import pandas as pd
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import os
import optuna
from optuna.trial import TrialState
import time
import json
from improved_model import create_improved_model
from improved_training import WeightedTrainer, compute_metrics, VariableLengthAudioDataset, collate_fn

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.001, 0.1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    
    # Load model and config (using global variables to avoid reloading for each trial)
    global model_name, num_labels, processed, train_idx, eval_idx
    
    # Create dataset
    dataset = VariableLengthAudioDataset(processed)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    eval_dataset = torch.utils.data.Subset(dataset, eval_idx)
    
    # Calculate class weights
    labels = [processed[i]['labels'] for i in train_idx]
    unique_labels = np.unique(labels)
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=labels)
    class_weights = torch.from_numpy(class_weights).float().to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with mask_time_prob=0 to disable masking for hyperparameter search
    # This avoids the sequence length error
    model = create_improved_model(
        model_name, 
        num_labels, 
        use_staged_training=True, 
        dropout_rate=dropout_rate,
        mask_time_prob=0.0  # Disable masking during hyperparameter optimization
    )
    
    # Update the training arguments
    training_args = TrainingArguments(
        output_dir=f'./output/hparam_trial_{trial.number}',
        learning_rate=learning_rate,
        num_train_epochs=15,  # Use fewer epochs for hyperparameter search
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_checkpointing=False,  # Start with this disabled to avoid initial errors
        gradient_accumulation_steps=2,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        weight_decay=weight_decay,
        logging_steps=100,
        eval_steps=100,
        save_total_limit=1,
        eval_strategy='steps',
        save_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        warmup_ratio=warmup_ratio,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        logging_first_step=False
    )
    
    # Create trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        class_weights=class_weights
    )
    
    # Train with error handling
    try:
        trainer.train()
        # Evaluate on validation set
        eval_result = trainer.evaluate()
        f1_score = eval_result.get("eval_f1", 0)
    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        # Return a very low score so Optuna will avoid these parameters
        f1_score = 0.0
    finally:
        # Clean up to save memory
        del trainer
        del model
        torch.cuda.empty_cache()
    
    return f1_score

def preprocess_data():
    """Load and preprocess data"""
    global model_name, num_labels
    
    # Define label map
    label_map = {
        'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4,
        'ควย': 5, 'สวะ': 6, 'หี': 7, 'แตด': 8
    }
    num_labels = len(label_map)
    
    # Load dataset
    df = pd.read_csv('csv/main.csv')
    print(f"Loaded dataset with {len(df)} samples")
    
    # Setup feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        return_attention_mask=True,
        do_normalize=True,
    )
    
    # Define preprocessing function
    def preprocess_row(row, feature_extractor, target_sr=16000):
        try:
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
            
            # Peak normalization
            segment = segment / (torch.max(torch.abs(segment)) + 1e-8)
            
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
    
    # Process all samples
    from tqdm import tqdm
    processed = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        result = preprocess_row(row, feature_extractor)
        if result is not None:
            processed.append(result)
    
    # Filter extreme lengths
    def filter_extreme_lengths(processed, min_length=1600, max_length=160000):
        """
        Filter out samples with extreme lengths and ensure all samples are long enough for the model
        """
        filtered = []
        for item in processed:
            if item is None:
                continue
                
            # The Wav2Vec2 model needs a minimum sequence length to work properly
            # For masking to work, sequence_length must be greater than mask_length (typically 10)
            # Add some buffer to be safe (e.g., 20)
            length = len(item['input_values'])
            
            if length < 20:
                print(f"Filtered out sample with length {length} (too short for masking)")
            elif length > max_length:
                print(f"Filtered out sample with length {length} (too long)")
            else:
                filtered.append(item)
                
        return filtered
    
    processed = filter_extreme_lengths(processed)
    print(f"After filtering: {len(processed)} samples remain")
    
    # Create train/validation/test indices
    indices = list(range(len(processed)))
    labels = [p['labels'] for p in processed]
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=labels, random_state=42)
    eval_idx, test_idx = train_test_split(temp_idx, test_size=0.5, 
                                         stratify=[processed[i]['labels'] for i in temp_idx], random_state=42)
    
    print(f"Train set: {len(train_idx)} samples")
    print(f"Validation set: {len(eval_idx)} samples")
    print(f"Test set: {len(test_idx)} samples")
    
    return processed, train_idx, eval_idx, test_idx

if __name__ == "__main__":
    # Global variables
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    
    # Preprocess data
    print("Preprocessing data...")
    processed, train_idx, eval_idx, test_idx = preprocess_data()
    
    # Create study
    print("Starting hyperparameter optimization...")
    study = optuna.create_study(direction="maximize", study_name="wav2vec2_hyperopt")
    study.optimize(objective, n_trials=30, timeout=24*3600)  # Run for 24 hours or 30 trials
    
    # Print best parameters
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters
    with open("./output/best_hyperparams.json", "w") as f:
        json.dump(trial.params, f, indent=2)
    
    # Plot optimization history
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig("./output/hyperopt_history.png")
        
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig("./output/hyperopt_importances.png")
        
        plt.figure(figsize=(12, 10))
        optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        plt.savefig("./output/hyperopt_parallel.png")
    except:
        print("Could not generate optimization plots")

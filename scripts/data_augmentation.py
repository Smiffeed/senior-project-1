import os
import numpy as np
import pandas as pd
import torchaudio
import torch
import random
from audiomentations import (
    Compose, AddGaussianNoise, TimeStretch, PitchShift,
    Shift, Gain, RoomSimulator
)

def augment_audio_sample(audio, sample_rate):
    """Apply a random augmentation to an audio sample"""
    augmenter = Compose([
        # Time domain augmentations
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_shift=-0.2, max_shift=0.2, p=0.5),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),  # Fixed parameter names
        RoomSimulator(p=0.3)
    ])
    
    # Convert from tensor to numpy for audiomentations
    if isinstance(audio, torch.Tensor):
        audio_np = audio.numpy()
    else:
        audio_np = audio
        
    # Apply augmentation
    augmented = augmenter(samples=audio_np, sample_rate=sample_rate)
    
    # Return as tensor if input was tensor
    if isinstance(audio, torch.Tensor):
        return torch.from_numpy(augmented)
    return augmented

def balance_dataset(df, label_column='label', target_count=None, aug_factor=2, max_samples_per_class=500):
    """
    Balance a dataset by augmenting underrepresented classes
    
    Parameters:
    - df: DataFrame with 'file_path', 'start_time', 'end_time', and label columns
    - label_column: Column name containing the class labels
    - target_count: Target count per class (if None, use the max count * aug_factor)
    - aug_factor: Factor to determine how many samples to create per original
    - max_samples_per_class: Safety limit to prevent too many augmentations
    
    Returns:
    - Balanced DataFrame with original and augmented samples
    """
    # Count samples per class
    class_counts = df[label_column].value_counts()
    print(f"Original class distribution:\n{class_counts}")
    
    if target_count is None:
        target_count = int(class_counts.max() * aug_factor)
    
    # Safety check to prevent too many augmentations
    target_count = min(target_count, max_samples_per_class)
    print(f"Target count per class: {target_count}")
    
    balanced_rows = []
    balanced_rows.extend(df.to_dict('records'))  # Add original data
    
    # Import tqdm for progress bar
    from tqdm import tqdm
    
    # For each underrepresented class
    for label, count in class_counts.items():
        if count < target_count:
            # Select rows for this class
            class_df = df[df[label_column] == label]
            
            # Calculate how many augmented samples we need
            needed = min(target_count - count, max_samples_per_class - count)
            
            # Check if we have enough source samples
            if len(class_df) == 0:
                print(f"Warning: No samples found for class {label}. Skipping.")
                continue
                
            print(f"Augmenting class '{label}': {count} samples → {count + needed} samples")
            
            # Create augmented samples (with repetition if needed)
            for i in tqdm(range(needed), desc=f"Augmenting '{label}'"):
                # Select a random row to augment
                row_idx = i % len(class_df)
                row = class_df.iloc[row_idx].copy()
                
                try:
                    # Load audio
                    waveform, sample_rate = torchaudio.load(row['file_path'])
                    start_sample = int(row['start_time'] * sample_rate)
                    end_sample = int(row['end_time'] * sample_rate)
                    
                    # Safety check for valid segments
                    if start_sample >= end_sample or end_sample > waveform.shape[1]:
                        print(f"Warning: Invalid segment indices for {row['file_path']}. Skipping.")
                        continue
                        
                    segment = waveform[:, start_sample:end_sample]
                    
                    # Augment the audio
                    augmented = augment_audio_sample(segment, sample_rate)
                    
                    # Create new file path for augmented audio
                    base_dir = os.path.dirname(row['file_path'])
                    file_name = os.path.basename(row['file_path']).split('.')
                    aug_file_name = f"{file_name[0]}_aug_{i}.{file_name[1]}"
                    aug_file_path = os.path.join(base_dir, "augmented", aug_file_name)
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(aug_file_path), exist_ok=True)
                    
                    # Save augmented audio
                    if isinstance(augmented, torch.Tensor):
                        torchaudio.save(aug_file_path, augmented, sample_rate)
                    else:
                        torchaudio.save(aug_file_path, torch.from_numpy(augmented).unsqueeze(0), sample_rate)
                    
                    # Update row with new file path
                    row['file_path'] = aug_file_path
                    row['augmented'] = True
                    
                    # Add to balanced rows
                    balanced_rows.append(row)
                except Exception as e:
                    print(f"Error processing file {row['file_path']}: {e}")
                    continue
    
    # Create new dataframe with original and augmented data
    balanced_df = pd.DataFrame(balanced_rows)
    
    # Print new class distribution
    new_class_counts = balanced_df[label_column].value_counts()
    print(f"New class distribution after augmentation:\n{new_class_counts}")
    
    return balanced_df

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    # Set torch to use CPU for reliability
    torch.set_num_threads(4)  # Limit CPU threads for stability
    
    try:
        # Load your dataset
        print("Loading dataset...")
        df = pd.read_csv('csv/main.csv')
        print(f"Dataset loaded with {len(df)} samples")
        
        # Check if required columns exist
        required_cols = ['file_path', 'start_time', 'end_time', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check if files exist
        print("Validating file paths...")
        df = df[df.apply(lambda row: os.path.exists(row['file_path']), axis=1)]
        print(f"Found {len(df)} valid files")
        
        # Balance the dataset with reasonable limits
        balanced_df = balance_dataset(
            df, 
            label_column='label',
            aug_factor=1.5,  # Reduced from 2 to create fewer samples
            max_samples_per_class=300  # Safety limit
        )
        
        # Save the balanced dataset
        print("Saving balanced dataset...")
        balanced_df.to_csv('csv/balanced_main.csv', index=False)
        
        print(f"Original dataset size: {len(df)}")
        print(f"Balanced dataset size: {len(balanced_df)}")
        print(f"Dataset saved to: csv/balanced_main.csv")
        print(f"Process completed in {(time.time() - start_time) / 60:.2f} minutes")
    
    except Exception as e:
        print(f"Error in data augmentation process: {e}")
        import traceback
        traceback.print_exc()

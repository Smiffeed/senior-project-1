#!/usr/bin/env python3
"""
Create a more balanced dataset for training by duplicating rare profanity samples
and reducing the 'none' class dominance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import os

def create_balanced_dataset(input_csv, output_csv, target_ratio=0.3):
    """
    Create a more balanced dataset where profanity classes make up target_ratio of the data
    
    Args:
        input_csv: Path to the original training CSV
        output_csv: Path to save the balanced dataset
        target_ratio: Target ratio for profanity samples (default 0.3 = 30%)
    """
    
    df = pd.read_csv(input_csv)
    print(f"Original dataset size: {len(df)}")
    
    # Separate profanity and non-profanity samples
    none_samples = df[df['label'] == 'none'].copy()
    profanity_samples = df[df['label'] != 'none'].copy()
    
    print(f"Original none samples: {len(none_samples)}")
    print(f"Original profanity samples: {len(profanity_samples)}")
    
    # Count profanity classes
    profanity_counts = profanity_samples['label'].value_counts()
    print("\nOriginal profanity distribution:")
    print(profanity_counts)
    
    # Calculate target counts
    total_profanity_target = int(len(none_samples) * target_ratio / (1 - target_ratio))
    per_class_target = max(50, total_profanity_target // len(profanity_counts))  # Minimum 50 per class
    
    print(f"\nTarget profanity samples: {total_profanity_target}")
    print(f"Target per class: {per_class_target}")
    
    # Balance profanity classes by duplication
    balanced_profanity = []
    
    for label in profanity_counts.index:
        class_samples = profanity_samples[profanity_samples['label'] == label].copy()
        current_count = len(class_samples)
        
        if current_count < per_class_target:
            # Calculate how many duplications we need
            multiplier = per_class_target // current_count
            remainder = per_class_target % current_count
            
            # Add the full multiplications
            for _ in range(multiplier):
                balanced_profanity.append(class_samples.copy())
            
            # Add the remainder
            if remainder > 0:
                balanced_profanity.append(class_samples.head(remainder).copy())
        else:
            # If we already have enough, just take what we need
            balanced_profanity.append(class_samples.head(per_class_target).copy())
    
    # Combine balanced profanity samples
    balanced_profanity_df = pd.concat(balanced_profanity, ignore_index=True)
    
    # Shuffle the profanity samples
    balanced_profanity_df = balanced_profanity_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Reduce none samples to balance the dataset
    target_none_count = int(len(balanced_profanity_df) * (1 - target_ratio) / target_ratio)
    target_none_count = min(target_none_count, len(none_samples))  # Don't exceed available samples
    
    balanced_none_df = none_samples.sample(n=target_none_count, random_state=42)
    
    # Combine everything
    final_df = pd.concat([balanced_none_df, balanced_profanity_df], ignore_index=True)
    
    # Shuffle the final dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the balanced dataset
    final_df.to_csv(output_csv, index=False)
    
    print(f"\nFinal dataset size: {len(final_df)}")
    print("Final class distribution:")
    final_counts = final_df['label'].value_counts()
    print(final_counts)
    print("\nFinal class percentages:")
    print((final_counts / len(final_df) * 100).round(2))
    
    return final_df

if __name__ == "__main__":
    input_file = "csv/train.csv"
    output_file = "csv/balanced_train_enhanced.csv"
    
    print("Creating enhanced balanced dataset...")
    balanced_df = create_balanced_dataset(input_file, output_file, target_ratio=0.4)
    
    print(f"\nBalanced dataset saved to: {output_file}")
    
    # Create train/val split from the balanced dataset
    train_df, val_df = train_test_split(
        balanced_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=balanced_df['label']
    )
    
    # Save train/val splits
    train_df.to_csv("csv/balanced_train_enhanced_train.csv", index=False)
    val_df.to_csv("csv/balanced_train_enhanced_val.csv", index=False)
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

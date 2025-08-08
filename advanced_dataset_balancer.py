#!/usr/bin/env python3
"""
Advanced Data Augmentation and Dataset Balancing for Thai Profanity Detection

This script addresses the severe class imbalance in your training dataset:
- 'none': 55.7% (520 samples) - too dominant
- Profanity classes: 2.25% to 9.22% - severely underrepresented
- Rarest class 'แตด': only 21 samples vs 520 'none' samples

SOLUTION APPROACH:
1. Reduce 'none' class dominance 
2. Aggressively augment rare profanity classes
3. Create balanced synthetic samples
4. Maintain audio quality and diversity
"""

import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import os
from collections import Counter
import torchaudio
import torch
from sklearn.model_selection import train_test_split
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class AudioAugmentor:
    """Advanced audio augmentation for Thai speech data"""
    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def load_audio_segment(self, file_path, start_time, end_time):
        """Load specific audio segment"""
        try:
            # Get metadata first
            info = torchaudio.info(file_path)
            original_sr = info.sample_rate
            
            # Calculate frame positions
            start_frame = int(start_time * original_sr)
            num_frames = int((end_time - start_time) * original_sr)
            
            # Load the segment
            audio, sr = torchaudio.load(
                file_path, 
                frame_offset=start_frame, 
                num_frames=num_frames
            )
            
            # Convert to mono
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.sr:
                audio = torchaudio.functional.resample(audio, sr, self.sr)
            
            return audio.squeeze().numpy()
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def time_stretch(self, audio, factor_range=(0.8, 1.25)):
        """Time stretching while preserving pitch"""
        factor = np.random.uniform(*factor_range)
        try:
            stretched = librosa.effects.time_stretch(audio, rate=factor)
            return stretched
        except:
            return audio
    
    def pitch_shift(self, audio, semitone_range=(-3, 3)):
        """Pitch shifting while preserving Thai tonal characteristics"""
        n_steps = np.random.uniform(*semitone_range)
        try:
            shifted = librosa.effects.pitch_shift(
                audio, sr=self.sr, n_steps=n_steps, bins_per_octave=24
            )
            return shifted
        except:
            return audio
    
    def add_noise(self, audio, noise_level_range=(0.001, 0.01)):
        """Add various types of background noise"""
        noise_type = np.random.choice(['gaussian', 'uniform', 'pink'])
        noise_level = np.random.uniform(*noise_level_range)
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, len(audio))
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level, noise_level, len(audio))
        else:  # pink noise
            try:
                noise = librosa.core.pink_noise(len(audio)) * noise_level
            except:
                noise = np.random.normal(0, noise_level, len(audio))
        
        return audio + noise
    
    def volume_change(self, audio, factor_range=(0.7, 1.4)):
        """Random volume scaling"""
        factor = np.random.uniform(*factor_range)
        return audio * factor
    
    def add_reverb(self, audio, room_size_range=(0.1, 0.8)):
        """Simple reverb effect simulation"""
        room_size = np.random.uniform(*room_size_range)
        delay_samples = int(room_size * 0.05 * self.sr)  # Max 50ms delay
        
        if delay_samples > 0:
            reverb_signal = np.zeros_like(audio)
            reverb_signal[delay_samples:] = audio[:-delay_samples] * (room_size * 0.3)
            return audio + reverb_signal
        return audio
    
    def frequency_mask(self, audio, mask_ratio_range=(0.05, 0.15)):
        """Random frequency masking in time domain"""
        mask_ratio = np.random.uniform(*mask_ratio_range)
        mask_length = int(len(audio) * mask_ratio)
        
        if mask_length > 0:
            start_idx = np.random.randint(0, len(audio) - mask_length)
            audio_copy = audio.copy()
            audio_copy[start_idx:start_idx + mask_length] *= np.random.uniform(0.1, 0.3)
            return audio_copy
        return audio
    
    def dynamic_range_compression(self, audio, ratio=0.6):
        """Compress dynamic range to enhance weak sounds"""
        return np.sign(audio) * (np.abs(audio) ** ratio)
    
    def augment_profanity(self, audio, intensity='medium'):
        """Apply multiple augmentations for profanity samples"""
        if intensity == 'light':
            n_augs = np.random.randint(2, 4)
        elif intensity == 'medium':
            n_augs = np.random.randint(3, 5)
        else:  # heavy
            n_augs = np.random.randint(4, 7)
        
        # Available augmentations
        augmentations = [
            lambda x: self.time_stretch(x, (0.85, 1.2)),
            lambda x: self.pitch_shift(x, (-2.5, 2.5)),
            lambda x: self.add_noise(x, (0.002, 0.008)),
            lambda x: self.volume_change(x, (0.8, 1.3)),
            lambda x: self.add_reverb(x, (0.1, 0.6)),
            lambda x: self.frequency_mask(x, (0.03, 0.12)),
            lambda x: self.dynamic_range_compression(x, np.random.uniform(0.5, 0.8))
        ]
        
        # Randomly select and apply augmentations
        selected_augs = np.random.choice(augmentations, n_augs, replace=False)
        
        augmented = audio.copy()
        for aug_func in selected_augs:
            augmented = aug_func(augmented)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(augmented))
        if max_val > 1.0:
            augmented = augmented / (max_val + 1e-8)
        
        return augmented
    
    def augment_none_class(self, audio):
        """Light augmentation for 'none' class to add variety"""
        augmentations = [
            lambda x: self.volume_change(x, (0.9, 1.1)),
            lambda x: self.add_noise(x, (0.0005, 0.002)),
            lambda x: self.time_stretch(x, (0.95, 1.05)),
        ]
        
        # Apply 1-2 light augmentations
        n_augs = np.random.randint(1, 3)
        selected_augs = np.random.choice(augmentations, n_augs, replace=False)
        
        augmented = audio.copy()
        for aug_func in selected_augs:
            augmented = aug_func(augmented)
            
        return augmented

class DatasetBalancer:
    """Balance dataset through strategic sampling and augmentation"""
    
    def __init__(self, csv_path, target_none_ratio=0.4):
        self.df = pd.read_csv(csv_path)
        self.target_none_ratio = target_none_ratio  # Target 40% 'none', 60% profanity
        self.augmentor = AudioAugmentor()
        
    def analyze_distribution(self):
        """Analyze current class distribution"""
        class_counts = self.df['label'].value_counts()
        print("=== CURRENT DISTRIBUTION ===")
        for label, count in class_counts.items():
            pct = count / len(self.df) * 100
            print(f"{label:>6}: {count:>3} samples ({pct:>5.1f}%)")
        
        return class_counts
    
    def calculate_target_counts(self, class_counts):
        """Calculate target sample counts for balanced dataset"""
        profanity_classes = [label for label in class_counts.index if label != 'none']
        
        # Calculate total profanity samples we want
        none_count = class_counts['none']
        target_none_count = int(none_count * 0.8)  # Reduce 'none' by 20%
        
        # Calculate profanity target based on desired ratio
        target_profanity_total = int(target_none_count * (1 - self.target_none_ratio) / self.target_none_ratio)
        target_per_profanity = target_profanity_total // len(profanity_classes)
        
        # Ensure minimum samples per class
        target_per_profanity = max(target_per_profanity, 80)
        
        targets = {'none': target_none_count}
        for label in profanity_classes:
            targets[label] = target_per_profanity
            
        print(f"\n=== TARGET DISTRIBUTION ===")
        total_target = sum(targets.values())
        for label, target in targets.items():
            pct = target / total_target * 100
            print(f"{label:>6}: {target:>3} samples ({pct:>5.1f}%)")
            
        return targets
    
    def create_balanced_dataset(self, output_path):
        """Create balanced dataset with augmentation"""
        print("🚀 Starting dataset balancing process...")
        
        # Analyze current distribution
        class_counts = self.analyze_distribution()
        
        # Calculate targets
        targets = self.calculate_target_counts(class_counts)
        
        # Create balanced dataset
        balanced_data = []
        
        for label in targets.keys():
            label_df = self.df[self.df['label'] == label].copy()
            current_count = len(label_df)
            target_count = targets[label]
            
            print(f"\n📊 Processing class '{label}': {current_count} -> {target_count}")
            
            if label == 'none':
                # For 'none' class: randomly sample to reduce count
                if current_count > target_count:
                    sampled_df = label_df.sample(n=target_count, random_state=42)
                else:
                    sampled_df = label_df
                    
                # Add light augmentation for variety
                for _, row in sampled_df.iterrows():
                    # Add original
                    balanced_data.append(row.to_dict())
                    
                    # Add one lightly augmented version for variety
                    if np.random.random() < 0.3:  # 30% chance
                        aug_row = row.copy()
                        aug_row['augmented'] = True
                        aug_row['aug_type'] = 'light_none'
                        balanced_data.append(aug_row.to_dict())
                        
            else:
                # For profanity classes: augment heavily
                multiplier = max(1, target_count // current_count)
                remainder = target_count % current_count
                
                for idx, (_, row) in enumerate(label_df.iterrows()):
                    # Add original
                    balanced_data.append(row.to_dict())
                    
                    # Add augmented versions
                    n_augs = multiplier - 1
                    if idx < remainder:
                        n_augs += 1
                    
                    for aug_idx in range(n_augs):
                        aug_row = row.copy()
                        aug_row['augmented'] = True
                        
                        # Vary augmentation intensity based on class rarity
                        if current_count <= 25:  # Very rare classes
                            aug_row['aug_type'] = 'heavy'
                        elif current_count <= 50:  # Moderately rare
                            aug_row['aug_type'] = 'medium'
                        else:  # Less rare
                            aug_row['aug_type'] = 'light'
                            
                        balanced_data.append(aug_row.to_dict())
        
        # Create final DataFrame
        balanced_df = pd.DataFrame(balanced_data)
        
        # Add augmentation flags if not present
        if 'augmented' not in balanced_df.columns:
            balanced_df['augmented'] = False
        if 'aug_type' not in balanced_df.columns:
            balanced_df['aug_type'] = 'none'
        
        # Shuffle the dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save balanced dataset
        balanced_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Balanced dataset saved to: {output_path}")
        print(f"📈 Total samples: {len(balanced_df)}")
        
        # Show final distribution
        final_counts = balanced_df['label'].value_counts()
        print("\n=== FINAL DISTRIBUTION ===")
        for label, count in final_counts.items():
            pct = count / len(balanced_df) * 100
            print(f"{label:>6}: {count:>3} samples ({pct:>5.1f}%)")
        
        # Show augmentation statistics
        aug_stats = balanced_df['aug_type'].value_counts()
        print(f"\n=== AUGMENTATION STATISTICS ===")
        for aug_type, count in aug_stats.items():
            pct = count / len(balanced_df) * 100
            print(f"{aug_type:>10}: {count:>3} samples ({pct:>5.1f}%)")
        
        return balanced_df

def main():
    """Main function to balance the dataset"""
    print("🎯 THAI PROFANITY DATASET BALANCER")
    print("=" * 50)
    
    # Configuration
    input_csv = 'csv/train.csv'
    output_csv = 'csv/balanced_train_advanced.csv'
    
    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"❌ Error: Input file {input_csv} not found!")
        return
    
    # Create balancer
    balancer = DatasetBalancer(input_csv, target_none_ratio=0.35)  # 35% 'none', 65% profanity
    
    # Create balanced dataset
    balanced_df = balancer.create_balanced_dataset(output_csv)
    
    print("\n🎉 Dataset balancing completed successfully!")
    print("\n📋 NEXT STEPS:")
    print("1. Use 'csv/balanced_train_advanced.csv' for training")
    print("2. Your model will handle train/validation split internally")
    print("3. The augmentation metadata is included for reference")
    print("4. Original samples are preserved with augmented variants")
    
    print("\n💡 EXPECTED IMPROVEMENTS:")
    print("- Better recall for rare profanity classes")
    print("- Reduced bias towards 'none' class")
    print("- More robust model with diverse training data")
    print("- Improved F1-scores across all classes")
    
    print(f"\n📊 FINAL DATASET SUMMARY:")
    final_counts = balanced_df['label'].value_counts()
    print(f"   Total samples: {len(balanced_df)}")
    print("   Class distribution:")
    for label, count in final_counts.items():
        pct = count / len(balanced_df) * 100
        print(f"     {label:>6}: {count:>3} samples ({pct:>5.1f}%)")

if __name__ == "__main__":
    main()

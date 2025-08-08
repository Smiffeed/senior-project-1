#!/usr/bin/env python3
"""
Real Audio Augmentation for Thai Profanity Detection Dataset

This script performs ACTUAL audio augmentation by:
1. Loading audio segments from the balanced dataset
2. Applying various audio transformations (pitch, speed, noise, etc.)
3. Saving augmented audio files to disk
4. Creating a new CSV with paths to augmented files

Dependencies: pip install librosa soundfile torchaudio
"""

import pandas as pd
import numpy as np
import os
import torchaudio
import torch
import soundfile as sf
from pathlib import Path
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class RealAudioAugmentor:
    """Performs actual audio augmentation and saves files"""
    
    def __init__(self, target_sr=16000, output_dir="dataset/augmented"):
        self.target_sr = target_sr
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Audio augmentor initialized. Output dir: {self.output_dir}")
    
    def load_audio_segment(self, file_path, start_time, end_time):
        """Load and preprocess audio segment"""
        try:
            # Get file info
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
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Ensure float32 dtype
            audio = audio.float()
            
            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                audio = resampler(audio)
            
            # Convert to numpy with proper dtype
            audio_np = audio.squeeze().numpy().astype(np.float32)
            
            return audio_np
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def pitch_shift(self, audio, shift_steps):
        """Pitch shift using phase vocoder approximation"""
        # Simple pitch shifting using interpolation (basic method)
        if abs(shift_steps) < 0.1:
            return audio
        
        # Convert semitones to ratio
        ratio = 2 ** (shift_steps / 12.0)
        
        # Ensure audio is float32 and convert to tensor
        audio_np = audio.astype(np.float32)
        audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        original_length = len(audio)
        
        try:
            # Simple resampling approach
            new_sr = int(self.target_sr * ratio)
            if new_sr <= 0 or new_sr > 96000:  # Reasonable bounds
                return audio
            
            resampler = torchaudio.transforms.Resample(self.target_sr, new_sr)
            shifted = resampler(audio_tensor)
            
            # Resample back to original sample rate
            resampler_back = torchaudio.transforms.Resample(new_sr, self.target_sr)
            result = resampler_back(shifted)
            
            # Trim or pad to original length
            result_np = result.squeeze().numpy().astype(np.float32)
            if len(result_np) > original_length:
                result_np = result_np[:original_length]
            elif len(result_np) < original_length:
                pad_length = original_length - len(result_np)
                result_np = np.pad(result_np, (0, pad_length), mode='edge')
            
            return result_np
            
        except Exception as e:
            # If pitch shifting fails, return original
            print(f"Pitch shift failed: {e}, returning original")
            return audio
    
    def time_stretch(self, audio, stretch_factor):
        """Time stretching using simple interpolation"""
        if abs(stretch_factor - 1.0) < 0.01:
            return audio
        
        try:
            # Ensure audio is float32
            audio = audio.astype(np.float32)
            
            # Simple time stretching using interpolation
            original_length = len(audio)
            new_length = int(original_length / stretch_factor)
            
            if new_length <= 0:
                return audio
            
            # Create new time indices
            old_indices = np.linspace(0, original_length - 1, new_length)
            
            # Interpolate
            stretched = np.interp(old_indices, np.arange(original_length), audio)
            
            # Pad or trim to original length
            if len(stretched) > original_length:
                stretched = stretched[:original_length]
            elif len(stretched) < original_length:
                pad_length = original_length - len(stretched)
                stretched = np.pad(stretched, (0, pad_length), mode='edge')
            
            return stretched.astype(np.float32)
            
        except Exception as e:
            print(f"Time stretch failed: {e}, returning original")
            return audio
    
    def add_noise(self, audio, noise_level):
        """Add gaussian noise"""
        noise = np.random.normal(0, noise_level, len(audio)).astype(np.float32)
        return (audio + noise).astype(np.float32)
    
    def volume_change(self, audio, volume_factor):
        """Change volume"""
        return (audio * volume_factor).astype(np.float32)
    
    def add_reverb(self, audio, room_size=0.3):
        """Simple reverb simulation"""
        delay_samples = int(room_size * 0.05 * self.target_sr)  # Max 50ms delay
        
        if delay_samples > 0:
            reverb_signal = np.zeros_like(audio)
            if len(audio) > delay_samples:
                reverb_signal[delay_samples:] = audio[:-delay_samples] * (room_size * 0.3)
                return (audio + reverb_signal).astype(np.float32)
        return audio.astype(np.float32)
    
    def frequency_mask(self, audio, mask_ratio=0.1):
        """Mask part of the signal"""
        mask_length = int(len(audio) * mask_ratio)
        if mask_length > 0:
            start_idx = np.random.randint(0, len(audio) - mask_length)
            audio_copy = audio.copy().astype(np.float32)
            audio_copy[start_idx:start_idx + mask_length] *= 0.2
            return audio_copy
        return audio.astype(np.float32)
    
    def augment_audio(self, audio, aug_type):
        """Apply augmentation based on type"""
        augmented = audio.copy()
        
        if aug_type == 'light':
            # Light augmentation
            augmentations = []
            if np.random.random() < 0.5:
                augmentations.append(lambda x: self.volume_change(x, np.random.uniform(0.8, 1.2)))
            if np.random.random() < 0.3:
                augmentations.append(lambda x: self.add_noise(x, np.random.uniform(0.001, 0.005)))
            if np.random.random() < 0.3:
                augmentations.append(lambda x: self.time_stretch(x, np.random.uniform(0.95, 1.05)))
        
        elif aug_type == 'medium':
            # Medium augmentation
            augmentations = []
            if np.random.random() < 0.7:
                augmentations.append(lambda x: self.volume_change(x, np.random.uniform(0.7, 1.3)))
            if np.random.random() < 0.5:
                augmentations.append(lambda x: self.add_noise(x, np.random.uniform(0.002, 0.008)))
            if np.random.random() < 0.6:
                augmentations.append(lambda x: self.time_stretch(x, np.random.uniform(0.9, 1.1)))
            if np.random.random() < 0.4:
                augmentations.append(lambda x: self.pitch_shift(x, np.random.uniform(-2, 2)))
            if np.random.random() < 0.3:
                augmentations.append(lambda x: self.add_reverb(x, np.random.uniform(0.1, 0.4)))
        
        elif aug_type == 'heavy':
            # Heavy augmentation
            augmentations = []
            if np.random.random() < 0.8:
                augmentations.append(lambda x: self.volume_change(x, np.random.uniform(0.6, 1.4)))
            if np.random.random() < 0.7:
                augmentations.append(lambda x: self.add_noise(x, np.random.uniform(0.003, 0.012)))
            if np.random.random() < 0.8:
                augmentations.append(lambda x: self.time_stretch(x, np.random.uniform(0.85, 1.15)))
            if np.random.random() < 0.7:
                augmentations.append(lambda x: self.pitch_shift(x, np.random.uniform(-3, 3)))
            if np.random.random() < 0.5:
                augmentations.append(lambda x: self.add_reverb(x, np.random.uniform(0.2, 0.6)))
            if np.random.random() < 0.4:
                augmentations.append(lambda x: self.frequency_mask(x, np.random.uniform(0.05, 0.15)))
        
        else:  # light_none or original
            augmentations = []
            if np.random.random() < 0.3:
                augmentations.append(lambda x: self.volume_change(x, np.random.uniform(0.9, 1.1)))
        
        # Apply selected augmentations
        for aug_func in augmentations:
            augmented = aug_func(augmented)
        
        # Ensure float32 dtype
        augmented = augmented.astype(np.float32)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(augmented))
        if max_val > 1.0:
            augmented = augmented / (max_val + 1e-8)
        elif max_val > 0:
            # Ensure reasonable amplitude
            augmented = augmented / (max_val + 1e-8) * 0.8
        
        return augmented.astype(np.float32)
    
    def save_audio(self, audio, output_path):
        """Save audio to file"""
        try:
            sf.write(output_path, audio, self.target_sr)
            return True
        except Exception as e:
            print(f"Error saving {output_path}: {e}")
            return False

def process_train_csv_with_augmentation(input_csv, output_csv_path):
    """Create a complete training CSV with original + augmented samples in train.csv format"""
    
    print("🎵 CREATING COMPLETE TRAINING DATASET WITH AUGMENTATION")
    print("=" * 60)
    
    # Load the original train.csv
    df = pd.read_csv(input_csv)
    print(f"✅ Loaded original dataset: {len(df)} samples")
    
    # Analyze class distribution
    class_counts = df['label'].value_counts()
    print(f"\n📊 Original class distribution:")
    for label, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"   {label:>6}: {count:>3} samples ({pct:>5.1f}%)")
    
    # Initialize augmentor
    augmentor = RealAudioAugmentor()
    
    # Prepare output data - start with all original samples
    output_data = []
    
    # Add all original samples first
    for _, row in df.iterrows():
        output_data.append({
            'file_path': row['file_path'],
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'label': row['label']
        })
    
    print(f"\n🔄 Creating augmented samples for rare classes...")
    
    # Define augmentation strategy based on class rarity
    augmentation_targets = {
        'แตด': 60,  # Very rare - need many augmentations
        'สวะ': 50,  # Rare
        'หี': 45,   # Rare
        'เย็ด': 35, # Moderately rare
        'ควย': 30,  # Moderately rare
        'เหี้ย': 25, # Less rare
        'กู': 20,   # Less rare
        'มึง': 20,  # Less rare
        'none': 0   # Don't augment 'none' class
    }
    
    for label, target_augmentations in augmentation_targets.items():
        if target_augmentations == 0:
            continue
            
        label_samples = df[df['label'] == label]
        current_count = len(label_samples)
        
        if current_count == 0:
            continue
            
        print(f"\n📈 Augmenting '{label}': {current_count} -> +{target_augmentations} augmented samples")
        
        # Determine augmentation intensity
        if current_count <= 25:
            aug_intensity = 'heavy'
        elif current_count <= 45:
            aug_intensity = 'medium'
        else:
            aug_intensity = 'light'
        
        # Create augmented samples
        augmented_created = 0
        sample_idx = 0
        
        with tqdm(total=target_augmentations, desc=f"Augmenting {label}", leave=False) as pbar:
            while augmented_created < target_augmentations:
                # Cycle through available samples
                row = label_samples.iloc[sample_idx % len(label_samples)]
                sample_idx += 1
                
                # Load original audio
                audio = augmentor.load_audio_segment(
                    row['file_path'], 
                    row['start_time'], 
                    row['end_time']
                )
                
                if audio is None:
                    continue
                
                # Create augmented version
                augmented_audio = augmentor.augment_audio(audio, aug_intensity)
                
                # Generate output filename
                original_filename = Path(row['file_path']).stem
                aug_filename = f"{original_filename}_{label}_{augmented_created}_{aug_intensity}.wav"
                output_path = augmentor.output_dir / aug_filename
                
                # Save augmented audio
                if augmentor.save_audio(augmented_audio, output_path):
                    # Add to output data in same format as train.csv
                    duration = len(augmented_audio) / augmentor.target_sr
                    output_data.append({
                        'file_path': str(output_path),
                        'start_time': 0.0,  # Full augmented file
                        'end_time': duration,
                        'label': label
                    })
                    
                    augmented_created += 1
                    pbar.update(1)
    
    # Create output DataFrame
    output_df = pd.DataFrame(output_data)
    
    # Shuffle the dataset
    output_df = output_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the complete training dataset
    output_df.to_csv(output_csv_path, index=False)
    
    print(f"\n✅ Complete training dataset created!")
    print(f"📁 Output CSV: {output_csv_path}")
    print(f"📁 Augmented audio files: {augmentor.output_dir}")
    print(f"📊 Total samples: {len(output_df)}")
    
    # Show final statistics
    final_counts = output_df['label'].value_counts()
    print(f"\n=== FINAL TRAINING DATASET STATISTICS ===")
    total_samples = len(output_df)
    for label, count in final_counts.items():
        pct = count / total_samples * 100
        original_count = class_counts.get(label, 0)
        augmented_count = count - original_count
        print(f"   {label:>6}: {count:>3} total ({original_count:>2} orig + {augmented_count:>2} aug) ({pct:>5.1f}%)")
    
    print(f"\n📈 DATASET IMPROVEMENT:")
    print(f"   Original dataset: {len(df)} samples")
    print(f"   Final dataset: {len(output_df)} samples")
    print(f"   Increase: {len(output_df) - len(df)} augmented samples")
    print(f"   Growth: {len(output_df)/len(df):.1f}x")
    
    return output_df

def main():
    """Main function"""
    print("🎯 REAL AUDIO AUGMENTATION FOR THAI PROFANITY DATASET")
    print("=" * 60)
    
    # Configuration - Use original train.csv instead of balanced version
    input_csv = 'csv/train.csv'
    output_csv = 'csv/train_with_augmentation.csv'
    
    # Check if input exists
    if not os.path.exists(input_csv):
        print(f"❌ Error: {input_csv} not found!")
        print("Please make sure csv/train.csv exists.")
        return
    
    # Process dataset
    try:
        output_df = process_train_csv_with_augmentation(input_csv, output_csv)
        
        print(f"\n🎉 SUCCESS!")
        print(f"📋 NEXT STEPS:")
        print(f"1. Use '{output_csv}' for training instead of csv/train.csv")
        print(f"2. Augmented audio files are in 'dataset/augmented/'")
        print(f"3. Your model now has real audio variations")
        print(f"4. Expected much better performance on rare classes")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure you have required packages: pip install soundfile torchaudio")

if __name__ == "__main__":
    main()

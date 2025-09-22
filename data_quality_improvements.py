#!/usr/bin/env python3
"""
Data quality improvements and collection strategy
"""

import pandas as pd
import numpy as np
from collections import Counter
import librosa

class DataQualityAnalyzer:
    """
    Analyze and improve data quality
    """
    
    def __init__(self, train_csv, eval_csv):
        self.train_df = pd.read_csv(train_csv)
        self.eval_df = pd.read_csv(eval_csv)
    
    def analyze_current_issues(self):
        """Identify current data quality issues"""
        issues = {}
        
        # 1. Class distribution analysis
        train_labels = Counter(self.train_df['label'])
        eval_labels = Counter(self.eval_df['label'])
        
        issues['class_imbalance'] = {
            'train': train_labels,
            'eval': eval_labels,
            'imbalance_ratio': max(train_labels.values()) / min(train_labels.values())
        }
        
        # 2. Temporal duration analysis
        self.train_df['duration'] = self.train_df['end_time'] - self.train_df['start_time']
        self.eval_df['duration'] = self.eval_df['end_time'] - self.eval_df['start_time']
        
        issues['duration_stats'] = {
            'train_mean': self.train_df['duration'].mean(),
            'train_std': self.train_df['duration'].std(),
            'eval_mean': self.eval_df['duration'].mean(),
            'eval_std': self.eval_df['duration'].std(),
            'very_short_segments': len(self.train_df[self.train_df['duration'] < 0.2]),
            'very_long_segments': len(self.train_df[self.train_df['duration'] > 2.0])
        }
        
        # 3. File path analysis (check for missing profanity labels)
        file_issues = []
        for _, row in self.train_df.iterrows():
            filename = row['file_path'].split('/')[-1]
            # Check if filename contains profanity not in labels
            thai_profanity_in_filename = ['หี', 'ควย', 'สัส', 'ระยำ', 'แม่ง']
            for word in thai_profanity_in_filename:
                if word in filename and row['label'] not in ['เย็ด', 'กู', 'มึง', 'เหี้ย']:
                    file_issues.append({
                        'file': filename,
                        'found_word': word,
                        'current_label': row['label']
                    })
        
        issues['labeling_inconsistencies'] = file_issues
        
        return issues
    
    def suggest_improvements(self, issues):
        """Suggest specific improvements based on analysis"""
        suggestions = {}
        
        # 1. Address class imbalance
        target_samples_per_class = max(issues['class_imbalance']['train'].values())
        augmentation_needed = {}
        
        for label, count in issues['class_imbalance']['train'].items():
            if count < target_samples_per_class * 0.5:  # If less than 50% of max class
                augmentation_needed[label] = target_samples_per_class - count
        
        suggestions['data_augmentation'] = {
            'strategy': 'aggressive_minority_augmentation',
            'needed_samples': augmentation_needed,
            'methods': [
                'pitch_shift_preserve_formants',
                'time_stretch_multiple_rates',
                'background_noise_mixing',
                'speed_perturbation',
                'vocal_tract_length_perturbation'
            ]
        }
        
        # 2. Address temporal issues
        if issues['duration_stats']['very_short_segments'] > 0:
            suggestions['short_segments'] = {
                'strategy': 'context_expansion',
                'action': 'Expand segments shorter than 0.2s to include 0.1s context on each side',
                'affected_samples': issues['duration_stats']['very_short_segments']
            }
        
        # 3. Address labeling inconsistencies
        if issues['labeling_inconsistencies']:
            suggestions['relabeling'] = {
                'strategy': 'expand_label_set',
                'action': 'Consider adding more profanity classes or improve annotation',
                'inconsistent_files': len(issues['labeling_inconsistencies'])
            }
        
        # 4. Data collection recommendations
        suggestions['new_data_collection'] = {
            'priority_classes': list(augmentation_needed.keys()),
            'collection_strategy': [
                'Record natural conversations with Thai speakers',
                'Collect from diverse audio sources (podcasts, social media, streams)',
                'Ensure multiple speakers per profanity word',
                'Include various emotional contexts (angry, casual, joking)',
                'Collect from different audio qualities (clear, noisy, compressed)'
            ],
            'target_hours_per_class': 2.0  # 2 hours of audio per profanity class
        }
        
        return suggestions

class ImprovedDataProcessor:
    """
    Process data with improvements based on evaluation insights
    """
    
    def __init__(self, optimal_configs):
        self.optimal_configs = optimal_configs
    
    def create_multi_resolution_dataset(self, df):
        """
        Create dataset with multiple window sizes optimized for different tasks
        """
        datasets = {}
        
        for task_type, config in self.optimal_configs.items():
            window_size = config['window_size']
            stride_size = config['stride_size']
            
            task_dataset = []
            
            for _, row in df.iterrows():
                file_path = row['file_path']
                start_time = row['start_time']
                end_time = row['end_time']
                label = row['label']
                
                # Load audio segment
                try:
                    audio, sr = librosa.load(
                        file_path, 
                        sr=16000, 
                        offset=start_time, 
                        duration=end_time - start_time
                    )
                    
                    # Create overlapping windows
                    windows = self.create_overlapping_windows(
                        audio, window_size, stride_size, sr
                    )
                    
                    for window_audio in windows:
                        task_dataset.append({
                            'audio': window_audio,
                            'label': label,
                            'task_type': task_type,
                            'window_size': window_size,
                            'original_file': file_path
                        })
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
            
            datasets[task_type] = task_dataset
        
        return datasets
    
    def create_overlapping_windows(self, audio, window_size, stride_size, sr=16000):
        """Create overlapping windows from audio"""
        window_samples = int(window_size * sr)
        stride_samples = int(stride_size * sr)
        
        windows = []
        start = 0
        
        while start + window_samples <= len(audio):
            window = audio[start:start + window_samples]
            
            # Ensure minimum length
            if len(window) >= window_samples * 0.8:  # At least 80% of target length
                # Pad if necessary
                if len(window) < window_samples:
                    padding = window_samples - len(window)
                    window = np.pad(window, (0, padding), mode='constant')
                
                windows.append(window)
            
            start += stride_samples
        
        return windows
    
    def balance_dataset_advanced(self, dataset, target_distribution=None):
        """
        Advanced dataset balancing with quality preservation
        """
        if target_distribution is None:
            # Aim for balanced distribution
            label_counts = Counter([item['label'] for item in dataset])
            target_count = max(label_counts.values())
            target_distribution = {label: target_count for label in label_counts.keys()}
        
        balanced_dataset = []
        
        for label, target_count in target_distribution.items():
            label_data = [item for item in dataset if item['label'] == label]
            current_count = len(label_data)
            
            # Add all original samples
            balanced_dataset.extend(label_data)
            
            # Augment if needed
            if current_count < target_count:
                needed_samples = target_count - current_count
                augmented_samples = self.augment_samples_intelligently(
                    label_data, needed_samples
                )
                balanced_dataset.extend(augmented_samples)
        
        return balanced_dataset
    
    def augment_samples_intelligently(self, samples, needed_count):
        """
        Intelligent augmentation that maintains linguistic properties
        """
        augmented = []
        augmentation_methods = [
            self.pitch_shift_preserve_formants,
            self.time_stretch_preserve_pitch,
            self.add_environmental_noise,
            self.volume_normalization_variation,
            self.spectral_mask_augmentation
        ]
        
        while len(augmented) < needed_count and samples:
            # Select random sample
            base_sample = np.random.choice(samples)
            
            # Select random augmentation method
            aug_method = np.random.choice(augmentation_methods)
            
            try:
                augmented_audio = aug_method(base_sample['audio'])
                
                # Quality check
                if self.passes_quality_check(augmented_audio, base_sample['audio']):
                    augmented_sample = base_sample.copy()
                    augmented_sample['audio'] = augmented_audio
                    augmented_sample['augmented'] = True
                    augmented.append(augmented_sample)
                    
            except Exception as e:
                print(f"Augmentation failed: {e}")
                continue
        
        return augmented[:needed_count]
    
    def pitch_shift_preserve_formants(self, audio, n_steps_range=(-3, 3)):
        """Pitch shift while preserving formant structure"""
        n_steps = np.random.uniform(*n_steps_range)
        return librosa.effects.pitch_shift(audio, sr=16000, n_steps=n_steps)
    
    def time_stretch_preserve_pitch(self, audio, rate_range=(0.9, 1.1)):
        """Time stretch while preserving pitch"""
        rate = np.random.uniform(*rate_range)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def add_environmental_noise(self, audio, snr_range=(15, 30)):
        """Add realistic environmental noise"""
        snr_db = np.random.uniform(*snr_range)
        
        # Generate colored noise
        noise_color = np.random.choice(['white', 'pink', 'brown'])
        
        if noise_color == 'white':
            noise = np.random.normal(0, 1, len(audio))
        elif noise_color == 'pink':
            # Pink noise (1/f)
            freqs = np.fft.fftfreq(len(audio))
            freqs[0] = 1
            noise_fft = np.random.normal(0, 1, len(audio)) / np.sqrt(np.abs(freqs))
            noise = np.fft.ifft(noise_fft).real
        else:  # brown
            # Brown noise (1/f^2)
            freqs = np.fft.fftfreq(len(audio))
            freqs[0] = 1
            noise_fft = np.random.normal(0, 1, len(audio)) / np.abs(freqs)
            noise = np.fft.ifft(noise_fft).real
        
        # Scale noise to achieve target SNR
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            noise_scale = np.sqrt(signal_power / (noise_power * 10**(snr_db/10)))
            noise = noise * noise_scale
        
        return audio + noise
    
    def volume_normalization_variation(self, audio, gain_range=(-6, 6)):
        """Apply volume variation in dB"""
        gain_db = np.random.uniform(*gain_range)
        gain_linear = 10**(gain_db/20)
        return audio * gain_linear
    
    def spectral_mask_augmentation(self, audio, mask_prob=0.1, mask_size=0.1):
        """Apply spectral masking (SpecAugment-style)"""
        # Convert to spectrogram
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Apply frequency masking
        if np.random.random() < mask_prob:
            freq_mask_size = int(magnitude.shape[0] * mask_size)
            freq_mask_start = np.random.randint(0, magnitude.shape[0] - freq_mask_size)
            magnitude[freq_mask_start:freq_mask_start + freq_mask_size, :] = 0
        
        # Apply time masking
        if np.random.random() < mask_prob:
            time_mask_size = int(magnitude.shape[1] * mask_size)
            time_mask_start = np.random.randint(0, magnitude.shape[1] - time_mask_size)
            magnitude[:, time_mask_start:time_mask_start + time_mask_size] = 0
        
        # Convert back to audio
        masked_stft = magnitude * np.exp(1j * phase)
        return librosa.istft(masked_stft)
    
    def passes_quality_check(self, augmented_audio, original_audio):
        """Check if augmented audio maintains quality"""
        # Basic quality checks
        
        # 1. No clipping
        if np.max(np.abs(augmented_audio)) > 0.99:
            return False
        
        # 2. Not too quiet
        if np.max(np.abs(augmented_audio)) < 0.01:
            return False
        
        # 3. Length preservation (within 10%)
        length_ratio = len(augmented_audio) / len(original_audio)
        if length_ratio < 0.9 or length_ratio > 1.1:
            return False
        
        # 4. Spectral similarity (simplified)
        orig_energy = np.sum(original_audio ** 2)
        aug_energy = np.sum(augmented_audio ** 2)
        
        if orig_energy > 0:
            energy_ratio = aug_energy / orig_energy
            if energy_ratio < 0.1 or energy_ratio > 10.0:
                return False
        
        return True
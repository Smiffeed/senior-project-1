#!/usr/bin/env python3
"""
Convert variable-length annotations to fixed-length training windows
Solves the mismatch between exact profanity timestamps and training requirements
"""

import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from collections import defaultdict
import json

class VariableToFixedWindowConverter:
    """
    Convert variable-length profanity annotations to fixed-length training windows
    """
    
    def __init__(self, target_window_sizes=[0.3, 0.5, 1.0, 2.0], overlap_ratio=0.5):
        """
        Initialize converter
        
        Args:
            target_window_sizes: List of target window sizes in seconds
            overlap_ratio: Overlap ratio for sliding windows (0.5 = 50% overlap)
        """
        self.target_window_sizes = target_window_sizes
        self.overlap_ratio = overlap_ratio
        
        # Label mapping
        self.label_map = {
            'none': 0,
            'เย็ด': 1,
            'กู': 2,
            'มึง': 3,
            'เหี้ย': 4
        }
    
    def analyze_annotation_statistics(self, csv_file):
        """Analyze the current annotation statistics"""
        df = pd.read_csv(csv_file)
        df['duration'] = df['end_time'] - df['start_time']
        
        stats = {
            'total_segments': len(df),
            'duration_stats': {
                'min': df['duration'].min(),
                'max': df['duration'].max(),
                'mean': df['duration'].mean(),
                'median': df['duration'].median(),
                'std': df['duration'].std()
            },
            'label_distribution': df['label'].value_counts().to_dict(),
            'duration_by_label': {}
        }
        
        # Duration statistics by label
        for label in df['label'].unique():
            label_durations = df[df['label'] == label]['duration']
            stats['duration_by_label'][label] = {
                'count': len(label_durations),
                'mean_duration': label_durations.mean(),
                'median_duration': label_durations.median(),
                'min_duration': label_durations.min(),
                'max_duration': label_durations.max()
            }
        
        print("=== ANNOTATION STATISTICS ===")
        print(f"Total segments: {stats['total_segments']}")
        print(f"Duration range: {stats['duration_stats']['min']:.3f}s - {stats['duration_stats']['max']:.3f}s")
        print(f"Mean duration: {stats['duration_stats']['mean']:.3f}s")
        print(f"Median duration: {stats['duration_stats']['median']:.3f}s")
        
        print("\nLabel distribution:")
        for label, count in stats['label_distribution'].items():
            print(f"  {label}: {count} segments")
        
        print("\nDuration by label:")
        for label, label_stats in stats['duration_by_label'].items():
            print(f"  {label}: {label_stats['mean_duration']:.3f}s avg, range: {label_stats['min_duration']:.3f}s - {label_stats['max_duration']:.3f}s")
        
        return stats
    
    def convert_to_fixed_windows(self, csv_file, output_dir, strategy='adaptive'):
        """
        Convert variable annotations to fixed windows
        
        Args:
            csv_file: Path to annotation CSV
            output_dir: Output directory for generated datasets
            strategy: 'adaptive', 'multi_scale', or 'context_aware'
        """
        df = pd.read_csv(csv_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"Converting {len(df)} annotations using {strategy} strategy...")
        
        if strategy == 'adaptive':
            return self._adaptive_conversion(df, output_dir)
        elif strategy == 'multi_scale':
            return self._multi_scale_conversion(df, output_dir)
        elif strategy == 'context_aware':
            return self._context_aware_conversion(df, output_dir)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _adaptive_conversion(self, df, output_dir):
        """
        Adaptive conversion: Choose window size based on profanity duration
        """
        training_data = []
        
        for window_size in self.target_window_sizes:
            print(f"\nProcessing window size: {window_size}s")
            window_data = []
            
            # Group by file to process efficiently
            for file_path, file_group in df.groupby('file_path'):
                try:
                    file_windows = self._process_file_adaptive(
                        file_path, file_group, window_size
                    )
                    window_data.extend(file_windows)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
            
            # Save window-specific dataset
            window_df = pd.DataFrame(window_data)
            output_file = output_dir / f"train_window_{window_size}s.csv"
            window_df.to_csv(output_file, index=False)
            
            print(f"Generated {len(window_data)} windows for {window_size}s")
            print(f"Label distribution: {window_df['label'].value_counts().to_dict()}")
            
            training_data.append({
                'window_size': window_size,
                'data': window_data,
                'file_path': str(output_file)
            })
        
        return training_data
    
    def _process_file_adaptive(self, file_path, annotations, window_size):
        """Process single file with adaptive windowing"""
        windows = []
        
        try:
            # Load full audio file
            audio, sr = librosa.load(file_path, sr=16000)
            audio_duration = len(audio) / sr
            
            # Sort annotations by start time
            annotations = annotations.sort_values('start_time')
            
            # Create annotation timeline
            timeline = self._create_annotation_timeline(annotations, audio_duration)
            
            # Generate sliding windows
            stride = window_size * (1 - self.overlap_ratio)
            current_time = 0
            
            while current_time + window_size <= audio_duration:
                window_start = current_time
                window_end = current_time + window_size
                
                # Determine window label based on profanity content
                window_label = self._determine_window_label(
                    timeline, window_start, window_end, window_size
                )
                
                # Extract audio window
                start_sample = int(window_start * sr)
                end_sample = int(window_end * sr)
                window_audio = audio[start_sample:end_sample]
                
                # Ensure exact window size
                target_samples = int(window_size * sr)
                if len(window_audio) < target_samples:
                    # Pad if too short
                    padding = target_samples - len(window_audio)
                    window_audio = np.pad(window_audio, (0, padding), mode='constant')
                elif len(window_audio) > target_samples:
                    # Trim if too long
                    window_audio = window_audio[:target_samples]
                
                windows.append({
                    'file_path': file_path,
                    'start_time': window_start,
                    'end_time': window_end,
                    'label': window_label,
                    'window_size': window_size,
                    'audio_samples': len(window_audio),
                    'profanity_coverage': self._calculate_profanity_coverage(
                        timeline, window_start, window_end
                    )
                })
                
                current_time += stride
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []
        
        return windows
    
    def _create_annotation_timeline(self, annotations, audio_duration):
        """Create timeline with profanity labels"""
        timeline = {}
        
        for _, row in annotations.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']
            label = row['label']
            
            # Add annotation to timeline
            timeline[(start_time, end_time)] = label
        
        return timeline
    
    def _determine_window_label(self, timeline, window_start, window_end, window_size):
        """
        Determine label for a fixed window based on profanity content
        
        Strategy:
        1. If >50% of window contains profanity → profanity label
        2. If multiple profanities → most prominent one
        3. Otherwise → 'none'
        """
        profanity_coverage = {}
        total_profanity_duration = 0
        
        for (ann_start, ann_end), label in timeline.items():
            if label == 'none':
                continue
                
            # Calculate overlap with window
            overlap_start = max(window_start, ann_start)
            overlap_end = min(window_end, ann_end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                total_profanity_duration += overlap_duration
                
                if label not in profanity_coverage:
                    profanity_coverage[label] = 0
                profanity_coverage[label] += overlap_duration
        
        # Decision logic
        profanity_ratio = total_profanity_duration / window_size
        
        if profanity_ratio > 0.5:  # >50% profanity
            # Return most prominent profanity
            return max(profanity_coverage, key=profanity_coverage.get)
        elif profanity_ratio > 0.1:  # 10-50% profanity
            # Return most prominent if clear winner
            if profanity_coverage:
                max_coverage = max(profanity_coverage.values())
                if max_coverage / total_profanity_duration > 0.6:  # >60% of profanity is one type
                    return max(profanity_coverage, key=profanity_coverage.get)
        
        # Default to 'none'
        return 'none'
    
    def _calculate_profanity_coverage(self, timeline, window_start, window_end):
        """Calculate what percentage of window contains profanity"""
        total_profanity = 0
        window_duration = window_end - window_start
        
        for (ann_start, ann_end), label in timeline.items():
            if label == 'none':
                continue
                
            overlap_start = max(window_start, ann_start)
            overlap_end = min(window_end, ann_end)
            
            if overlap_start < overlap_end:
                total_profanity += overlap_end - overlap_start
        
        return total_profanity / window_duration if window_duration > 0 else 0
    
    def _context_aware_conversion(self, df, output_dir):
        """
        Context-aware conversion: Expand short profanity with context
        """
        training_data = []
        
        for window_size in self.target_window_sizes:
            print(f"\nProcessing context-aware windows: {window_size}s")
            window_data = []
            
            for file_path, file_group in df.groupby('file_path'):
                try:
                    file_windows = self._process_file_context_aware(
                        file_path, file_group, window_size
                    )
                    window_data.extend(file_windows)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
            
            # Save dataset
            window_df = pd.DataFrame(window_data)
            output_file = output_dir / f"train_context_aware_{window_size}s.csv"
            window_df.to_csv(output_file, index=False)
            
            print(f"Generated {len(window_data)} context-aware windows")
            
            training_data.append({
                'window_size': window_size,
                'data': window_data,
                'file_path': str(output_file)
            })
        
        return training_data
    
    def _process_file_context_aware(self, file_path, annotations, window_size):
        """Process file with context-aware windowing"""
        windows = []
        
        try:
            audio, sr = librosa.load(file_path, sr=16000)
            audio_duration = len(audio) / sr
            
            # Process each profanity annotation
            for _, row in annotations.iterrows():
                if row['label'] == 'none':
                    continue  # Skip none annotations for profanity-focused training
                
                profanity_start = row['start_time']
                profanity_end = row['end_time']
                profanity_duration = profanity_end - profanity_start
                label = row['label']
                
                # Strategy based on profanity duration
                if profanity_duration >= window_size * 0.8:
                    # Long profanity: use as-is with minimal context
                    window_start = max(0, profanity_start - 0.1)
                    window_end = min(audio_duration, profanity_end + 0.1)
                    
                    # Adjust to exact window size
                    window_center = (window_start + window_end) / 2
                    window_start = max(0, window_center - window_size / 2)
                    window_end = min(audio_duration, window_start + window_size)
                    
                elif profanity_duration >= window_size * 0.3:
                    # Medium profanity: center with context
                    profanity_center = (profanity_start + profanity_end) / 2
                    window_start = max(0, profanity_center - window_size / 2)
                    window_end = min(audio_duration, window_start + window_size)
                    
                else:
                    # Short profanity: multiple context variations
                    context_variations = [0.2, 0.4, 0.6, 0.8]  # Different context ratios
                    
                    for context_ratio in context_variations:
                        context_size = window_size * context_ratio
                        profanity_offset_ratio = (1 - context_ratio) / 2
                        
                        # Place profanity at different positions within window
                        window_start = max(0, profanity_start - window_size * profanity_offset_ratio)
                        window_end = min(audio_duration, window_start + window_size)
                        
                        # Ensure profanity is within window
                        if window_end > profanity_end and window_start < profanity_start:
                            windows.append(self._create_window_sample(
                                file_path, audio, sr, window_start, window_end, 
                                label, window_size, f"context_{context_ratio}"
                            ))
                    
                    continue  # Skip the main window creation for short profanity
                
                # Create main window sample
                windows.append(self._create_window_sample(
                    file_path, audio, sr, window_start, window_end, 
                    label, window_size, "main"
                ))
            
            # Add some 'none' samples for balance
            windows.extend(self._generate_none_samples(
                file_path, audio, sr, annotations, window_size, 
                target_count=len(windows) // 3  # 1:3 ratio none:profanity
            ))
            
        except Exception as e:
            print(f"Error in context-aware processing: {e}")
            return []
        
        return windows
    
    def _create_window_sample(self, file_path, audio, sr, window_start, window_end, 
                             label, window_size, variant_type):
        """Create a single window sample"""
        # Extract audio
        start_sample = int(window_start * sr)
        end_sample = int(window_end * sr)
        window_audio = audio[start_sample:end_sample]
        
        # Ensure exact size
        target_samples = int(window_size * sr)
        if len(window_audio) != target_samples:
            if len(window_audio) < target_samples:
                padding = target_samples - len(window_audio)
                window_audio = np.pad(window_audio, (0, padding), mode='constant')
            else:
                window_audio = window_audio[:target_samples]
        
        return {
            'file_path': file_path,
            'start_time': window_start,
            'end_time': window_end,
            'label': label,
            'window_size': window_size,
            'variant_type': variant_type,
            'audio_samples': len(window_audio)
        }
    
    def _generate_none_samples(self, file_path, audio, sr, annotations, window_size, target_count):
        """Generate balanced 'none' samples"""
        none_windows = []
        audio_duration = len(audio) / sr
        
        # Find regions without profanity
        profanity_regions = []
        for _, row in annotations.iterrows():
            if row['label'] != 'none':
                profanity_regions.append((row['start_time'], row['end_time']))
        
        # Sort profanity regions
        profanity_regions.sort()
        
        # Generate none samples from gaps
        generated_count = 0
        current_time = 0
        
        while generated_count < target_count and current_time + window_size <= audio_duration:
            window_start = current_time
            window_end = current_time + window_size
            
            # Check if window overlaps with profanity
            has_profanity = False
            for prof_start, prof_end in profanity_regions:
                if not (window_end <= prof_start or window_start >= prof_end):
                    has_profanity = True
                    break
            
            if not has_profanity:
                none_windows.append(self._create_window_sample(
                    file_path, audio, sr, window_start, window_end,
                    'none', window_size, 'none_sample'
                ))
                generated_count += 1
                current_time += window_size  # Non-overlapping for none samples
            else:
                current_time += window_size * 0.1  # Small step if profanity found
        
        return none_windows
    
    def create_balanced_training_sets(self, training_data_list, balance_strategy='oversample'):
        """Create balanced training sets from generated windows"""
        balanced_sets = {}
        
        for data_info in training_data_list:
            window_size = data_info['window_size']
            data = data_info['data']
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Balance dataset
            if balance_strategy == 'oversample':
                balanced_df = self._oversample_minorities(df)
            elif balance_strategy == 'undersample':
                balanced_df = self._undersample_majorities(df)
            else:  # 'weighted'
                balanced_df = df  # Keep original, use weighted loss
            
            balanced_sets[window_size] = {
                'data': balanced_df,
                'original_count': len(df),
                'balanced_count': len(balanced_df),
                'label_distribution': balanced_df['label'].value_counts().to_dict()
            }
            
            print(f"\nWindow {window_size}s:")
            print(f"  Original: {len(df)} samples")
            print(f"  Balanced: {len(balanced_df)} samples")
            print(f"  Distribution: {balanced_df['label'].value_counts().to_dict()}")
        
        return balanced_sets
    
    def _oversample_minorities(self, df):
        """Oversample minority classes to match majority"""
        from sklearn.utils import resample
        
        # Find target count (max class count)
        class_counts = df['label'].value_counts()
        target_count = class_counts.max()
        
        balanced_dfs = []
        
        for label in class_counts.index:
            class_df = df[df['label'] == label]
            current_count = len(class_df)
            
            if current_count < target_count:
                # Oversample
                oversampled_df = resample(
                    class_df,
                    replace=True,
                    n_samples=target_count,
                    random_state=42
                )
                balanced_dfs.append(oversampled_df)
            else:
                balanced_dfs.append(class_df)
        
        return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)

def main():
    """Main function to convert eval_5labels.csv to training-ready datasets"""
    
    # Initialize converter
    converter = VariableToFixedWindowConverter(
        target_window_sizes=[0.3, 0.5, 1.0, 2.0],  # Based on your evaluation insights
        overlap_ratio=0.5
    )
    
    # Analyze current annotations
    input_file = "csv/eval_5labels.csv"
    stats = converter.analyze_annotation_statistics(input_file)
    
    # Convert to fixed windows with different strategies
    print("\n" + "="*50)
    print("CONVERTING TO FIXED WINDOWS")
    print("="*50)
    
    strategies = ['adaptive', 'context_aware']
    all_training_data = {}
    
    for strategy in strategies:
        print(f"\n--- Processing with {strategy} strategy ---")
        
        output_dir = f"csv/fixed_windows_{strategy}"
        training_data = converter.convert_to_fixed_windows(
            input_file, output_dir, strategy=strategy
        )
        
        all_training_data[strategy] = training_data
    
    # Create balanced datasets
    print("\n" + "="*50)
    print("CREATING BALANCED DATASETS")
    print("="*50)
    
    for strategy, training_data in all_training_data.items():
        print(f"\n--- Balancing {strategy} datasets ---")
        
        balanced_sets = converter.create_balanced_training_sets(
            training_data, balance_strategy='oversample'
        )
        
        # Save balanced datasets
        for window_size, balanced_info in balanced_sets.items():
            output_file = f"csv/fixed_windows_{strategy}/balanced_train_{window_size}s.csv"
            balanced_info['data'].to_csv(output_file, index=False)
            print(f"Saved balanced dataset: {output_file}")
    
    # Generate recommendations
    print("\n" + "="*50)
    print("TRAINING RECOMMENDATIONS")
    print("="*50)
    
    print("\nBased on your evaluation results and data analysis:")
    print("\n1. OPTIMAL WINDOW CONFIGURATIONS:")
    print("   - Binary classification: Use 2.0s windows (best F1 performance)")
    print("   - Multiclass classification: Use 0.3s windows (best class separation)")
    print("   - IoU evaluation: Use 0.5s windows (balanced performance)")
    
    print("\n2. RECOMMENDED TRAINING STRATEGY:")
    print("   - Stage 1: Train binary classifier with 2.0s windows")
    print("   - Stage 2: Fine-tune multiclass with 0.3s windows")
    print("   - Use context_aware strategy for better profanity boundary learning")
    
    print("\n3. DATA USAGE:")
    print("   - Use adaptive strategy for general robustness")
    print("   - Use context_aware strategy for precise profanity detection")
    print("   - Apply oversampling to balance minority profanity classes")
    
    print(f"\n4. NEXT STEPS:")
    print("   - Replace your current train.csv with generated balanced datasets")
    print("   - Modify your training script to use appropriate window sizes")
    print("   - Implement multi-stage training (binary → multiclass)")

if __name__ == "__main__":
    main()
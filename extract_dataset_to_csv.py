#!/usr/bin/env python3
"""
📊 DATASET CSV EXTRACTOR
Extract audio dataset annotations into CSV format like train_backup.csv

Features:
- Extracts single-word recordings from dataset/single/
- Extracts complex annotations from label files
- Supports multiple audio conditions (clear, noise, music, effect)
- Creates comprehensive CSV with all timing information
- Validates audio file existence
- Progress tracking and statistics
"""

import os
import csv
import librosa
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict

class DatasetExtractor:
    """Extract dataset annotations into CSV format"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.extracted_data = []
        self.stats = defaultdict(int)
        
        # Supported profanity classes
        self.profanity_classes = {
            'กู', 'ควย', 'มึง', 'สวะ', 'หี', 'เย็ด', 'เหี้ย', 'แตด'
        }
        
        print(f"🎯 Dataset Extractor initialized")
        print(f"   Dataset path: {self.dataset_path}")
        print(f"   Supported classes: {', '.join(sorted(self.profanity_classes))}")
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            # Load audio and get duration
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            return duration
        except Exception as e:
            print(f"⚠️  Warning: Could not get duration for {audio_path}: {e}")
            return 0.0
    
    def _extract_single_recordings(self):
        """Extract single-word recordings from dataset/single/"""
        print("\n🎵 Extracting single-word recordings...")
        
        single_path = self.dataset_path / "single"
        if not single_path.exists():
            print(f"⚠️  Single recordings path not found: {single_path}")
            return
        
        for class_dir in single_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            if class_name not in self.profanity_classes:
                print(f"⚠️  Skipping unknown class: {class_name}")
                continue
            
            print(f"  📂 Processing class: {class_name}")
            
            for audio_file in tqdm(class_dir.glob("*.wav"), desc=f"  {class_name}"):
                # Get audio duration
                duration = self._get_audio_duration(str(audio_file))
                
                if duration > 0:
                    # Single recordings: entire file is one class
                    self.extracted_data.append({
                        'file_path': str(audio_file.resolve()),
                        'start_time': 0.0,
                        'end_time': duration,
                        'label': class_name
                    })
                    
                    self.stats[f'single_{class_name}'] += 1
                    self.stats['total_single_recordings'] += 1
    
    def _extract_complex_annotations(self):
        """Extract complex annotations from label files"""
        print("\n🎵 Extracting complex annotations...")
        
        # Check different source directories
        source_dirs = [
            self.dataset_path / "us",
            self.dataset_path / "internet"
        ]
        
        for source_dir in source_dirs:
            if not source_dir.exists():
                continue
                
            print(f"  📂 Processing source: {source_dir.name}")
            self._process_source_directory(source_dir)
    
    def _process_source_directory(self, source_dir: Path):
        """Process a source directory (us, internet)"""
        
        # Check different conditions (clear, noise, music, effect)
        condition_dirs = [
            source_dir / "clear",
            source_dir / "noise", 
            source_dir / "music",
            source_dir / "effect"
        ]
        
        for condition_dir in condition_dirs:
            if not condition_dir.exists():
                continue
                
            condition_name = condition_dir.name
            print(f"    🔊 Processing condition: {condition_name}")
            
            label_dir = condition_dir / "label"
            if not label_dir.exists():
                print(f"      ⚠️  No label directory found: {label_dir}")
                continue
            
            # Process each label file
            label_files = list(label_dir.glob("*.txt"))
            
            for label_file in tqdm(label_files, desc=f"    {condition_name}"):
                audio_name = label_file.stem + ".wav"
                audio_path = condition_dir / audio_name
                
                if not audio_path.exists():
                    print(f"      ⚠️  Audio file not found: {audio_path}")
                    continue
                
                # Parse label file
                self._parse_label_file(label_file, audio_path, condition_name)
    
    def _parse_label_file(self, label_file: Path, audio_path: Path, condition: str):
        """Parse individual label file"""
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            segments_count = 0
            profanity_segments = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) != 3:
                    print(f"      ⚠️  Invalid line format in {label_file}: {line}")
                    continue
                
                try:
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    label = parts[2].strip()
                    
                    # Add to extracted data
                    self.extracted_data.append({
                        'file_path': str(audio_path.resolve()),
                        'start_time': start_time,
                        'end_time': end_time,
                        'label': label
                    })
                    
                    segments_count += 1
                    
                    # Update statistics
                    if label != 'none':
                        profanity_segments += 1
                        self.stats[f'complex_{label}'] += 1
                    else:
                        self.stats['complex_none'] += 1
                    
                    self.stats[f'condition_{condition}'] += 1
                    
                except ValueError as e:
                    print(f"      ⚠️  Error parsing line in {label_file}: {line} - {e}")
                    continue
            
            self.stats['total_complex_files'] += 1
            self.stats['total_segments'] += segments_count
            
        except Exception as e:
            print(f"      ❌ Error processing {label_file}: {e}")
    
    def extract_all(self) -> list:
        """Extract all data from dataset"""
        print("🚀 Starting dataset extraction...")
        
        # Extract single recordings
        self._extract_single_recordings()
        
        # Extract complex annotations  
        self._extract_complex_annotations()
        
        print(f"\n✅ Extraction completed!")
        self._print_statistics()
        
        return self.extracted_data
    
    def _print_statistics(self):
        """Print extraction statistics"""
        print("\n📊 EXTRACTION STATISTICS:")
        print("=" * 50)
        
        # Single recordings
        print("\n🎵 Single Recordings:")
        single_total = 0
        for class_name in sorted(self.profanity_classes):
            count = self.stats.get(f'single_{class_name}', 0)
            if count > 0:
                print(f"  {class_name}: {count}")
                single_total += count
        print(f"  Total single recordings: {single_total}")
        
        # Complex annotations
        print("\n🎵 Complex Annotations:")
        complex_profanity = 0
        for class_name in sorted(self.profanity_classes):
            count = self.stats.get(f'complex_{class_name}', 0)
            if count > 0:
                print(f"  {class_name}: {count}")
                complex_profanity += count
        
        none_count = self.stats.get('complex_none', 0)
        print(f"  none: {none_count}")
        print(f"  Total complex profanity segments: {complex_profanity}")
        print(f"  Total complex segments: {complex_profanity + none_count}")
        
        # By condition
        print("\n🔊 By Audio Condition:")
        conditions = ['clear', 'noise', 'music', 'effect']
        for condition in conditions:
            count = self.stats.get(f'condition_{condition}', 0)
            if count > 0:
                print(f"  {condition}: {count}")
        
        # Overall totals
        print("\n📈 Overall Totals:")
        print(f"  Total files processed: {self.stats.get('total_complex_files', 0) + len([k for k in self.stats.keys() if k.startswith('single_')])}")
        print(f"  Total data entries: {len(self.extracted_data)}")
        print(f"  Total profanity instances: {complex_profanity + single_total}")
        print(f"  Total 'none' instances: {none_count}")
    
    def save_to_csv(self, output_path: str, sort_by_file: bool = True):
        """Save extracted data to CSV file"""
        print(f"\n💾 Saving data to CSV: {output_path}")
        
        if not self.extracted_data:
            print("❌ No data to save!")
            return
        
        # Sort data if requested
        if sort_by_file:
            self.extracted_data.sort(key=lambda x: (x['file_path'], x['start_time']))
        
        # Write CSV
        fieldnames = ['file_path', 'start_time', 'end_time', 'label']
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in tqdm(self.extracted_data, desc="Writing CSV"):
                writer.writerow(entry)
        
        print(f"✅ CSV saved successfully!")
        print(f"   Total entries: {len(self.extracted_data)}")
        print(f"   Output file: {output_path}")
    
    def generate_dataset_splits(self, output_dir: str, train_ratio: float = 0.8, 
                              val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Generate train/validation/test splits"""
        print(f"\n🔄 Generating dataset splits...")
        
        if not self.extracted_data:
            print("❌ No data to split!")
            return
        
        # Group by file to ensure same file doesn't appear in different splits
        file_groups = defaultdict(list)
        for entry in self.extracted_data:
            file_groups[entry['file_path']].append(entry)
        
        files = list(file_groups.keys())
        import random
        random.shuffle(files)
        
        # Calculate split indices
        total_files = len(files)
        train_end = int(total_files * train_ratio)
        val_end = int(total_files * (train_ratio + val_ratio))
        
        # Create splits
        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]
        
        # Generate datasets
        splits = {
            'train': train_files,
            'val': val_files, 
            'test': test_files
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, split_files in splits.items():
            split_data = []
            for file_path in split_files:
                split_data.extend(file_groups[file_path])
            
            # Sort by file path and start time
            split_data.sort(key=lambda x: (x['file_path'], x['start_time']))
            
            # Save split CSV
            output_path = os.path.join(output_dir, f'{split_name}.csv')
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['file_path', 'start_time', 'end_time', 'label'])
                writer.writeheader()
                writer.writerows(split_data)
            
            print(f"  📄 {split_name}: {len(split_data)} entries ({len(split_files)} files)")
        
        print(f"✅ Dataset splits saved to: {output_dir}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Extract dataset annotations into CSV format")
    parser.add_argument("-d", "--dataset", type=str, default="dataset", 
                       help="Dataset directory path")
    parser.add_argument("-o", "--output", type=str, default="csv/extracted_dataset.csv",
                       help="Output CSV file path")
    parser.add_argument("--splits", type=str,
                       help="Generate train/val/test splits in specified directory")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Training set ratio (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                       help="Validation set ratio (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                       help="Test set ratio (default: 0.1)")
    parser.add_argument("--no-sort", action="store_true",
                       help="Don't sort output by file path")
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        print("❌ Error: Train/validation/test ratios must sum to 1.0")
        return
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = DatasetExtractor(args.dataset)
    
    # Extract data
    data = extractor.extract_all()
    
    if not data:
        print("❌ No data extracted!")
        return
    
    # Save main CSV
    extractor.save_to_csv(args.output, sort_by_file=not args.no_sort)
    
    # Generate splits if requested
    if args.splits:
        extractor.generate_dataset_splits(
            args.splits,
            args.train_ratio,
            args.val_ratio, 
            args.test_ratio
        )
    
    print("\n🎉 Dataset extraction completed successfully!")
    print(f"   Main CSV: {args.output}")
    if args.splits:
        print(f"   Splits directory: {args.splits}")

if __name__ == "__main__":
    main()
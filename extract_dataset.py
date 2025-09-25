#!/usr/bin/env python3
"""
Dataset Extractor - Extract all data from dataset folder to train.csv
Processes both single word recordings and complex annotated recordings
"""

import os
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def get_audio_duration(file_path):
    """Get audio file duration"""
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except:
        return 0.0

def extract_single_recordings(dataset_path):
    """Extract single word recordings from dataset/single/"""
    data = []
    single_path = os.path.join(dataset_path, 'single')
    
    if not os.path.exists(single_path):
        return data
    
    print("🎯 Extracting single word recordings...")
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(single_path) 
                  if os.path.isdir(os.path.join(single_path, d))]
    
    for class_name in tqdm(class_dirs, desc="Processing classes"):
        class_path = os.path.join(single_path, class_name)
        
        # Get all wav files in this class
        wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        
        for wav_file in wav_files:
            file_path = os.path.join(class_path, wav_file)
            
            # Get duration
            duration = get_audio_duration(file_path)
            
            # Single recordings are entirely one class
            data.append({
                'file_path': file_path,
                'start_time': 0.0,
                'end_time': duration,
                'label': class_name
            })
    
    print(f"✅ Extracted {len(data)} single word recordings")
    return data

def extract_annotated_recordings(dataset_path):
    """Extract annotated recordings from internet/ and us/ folders"""
    data = []
    
    # Process both internet and us folders
    for main_folder in ['internet', 'us']:
        main_path = os.path.join(dataset_path, main_folder)
        
        if not os.path.exists(main_path):
            continue
            
        print(f"🎯 Extracting {main_folder} recordings...")
        
        # Get all subfolders (clear, music, noise, effect)
        subfolders = [d for d in os.listdir(main_path) 
                     if os.path.isdir(os.path.join(main_path, d))]
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(main_path, subfolder)
            label_path = os.path.join(subfolder_path, 'label')
            
            if not os.path.exists(label_path):
                continue
            
            print(f"  Processing {subfolder}...")
            
            # Get all label files
            label_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
            
            for label_file in tqdm(label_files, desc=f"  {subfolder} files"):
                # Get corresponding audio file
                audio_name = label_file.replace('.txt', '.wav')
                audio_path = os.path.join(subfolder_path, audio_name)
                
                if not os.path.exists(audio_path):
                    continue
                
                # Read label file
                label_file_path = os.path.join(label_path, label_file)
                
                try:
                    # Read annotations
                    annotations = []
                    with open(label_file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split('\t')
                                if len(parts) >= 3:
                                    start_time = float(parts[0])
                                    end_time = float(parts[1])
                                    label = parts[2]
                                    annotations.append((start_time, end_time, label))
                    
                    # Get audio duration
                    total_duration = get_audio_duration(audio_path)
                    
                    if total_duration == 0:
                        continue
                    
                    # Sort annotations by start time
                    annotations.sort()
                    
                    # Fill gaps with 'none' labels
                    current_time = 0.0
                    
                    for start_time, end_time, label in annotations:
                        # Add 'none' segment before this annotation if there's a gap
                        if start_time > current_time:
                            data.append({
                                'file_path': audio_path,
                                'start_time': current_time,
                                'end_time': start_time,
                                'label': 'none'
                            })
                        
                        # Add the annotated segment
                        data.append({
                            'file_path': audio_path,
                            'start_time': start_time,
                            'end_time': end_time,
                            'label': label
                        })
                        
                        current_time = end_time
                    
                    # Add final 'none' segment if needed
                    if current_time < total_duration:
                        data.append({
                            'file_path': audio_path,
                            'start_time': current_time,
                            'end_time': total_duration,
                            'label': 'none'
                        })
                
                except Exception as e:
                    print(f"⚠️ Error processing {label_file}: {e}")
                    continue
    
    print(f"✅ Extracted {len(data)} annotated segments")
    return data

def main():
    dataset_path = "dataset"
    output_file = "csv/train.csv"
    
    # Ensure output directory exists
    os.makedirs("csv", exist_ok=True)
    
    print("🚀 Dataset Extraction Started")
    print("=" * 50)
    
    all_data = []
    
    # Extract single recordings
    single_data = extract_single_recordings(dataset_path)
    all_data.extend(single_data)
    
    # Extract annotated recordings
    annotated_data = extract_annotated_recordings(dataset_path)
    all_data.extend(annotated_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    if len(df) == 0:
        print("❌ No data extracted!")
        return
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print("\n📊 EXTRACTION SUMMARY:")
    print(f"   Total segments: {len(df)}")
    print(f"   Output file: {output_file}")
    
    # Show label distribution
    print("\n🏷️ Label distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"   {label}: {count}")
    
    # Show some statistics
    total_duration = (df['end_time'] - df['start_time']).sum()
    print(f"\n⏱️ Total audio duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    # Show sample data
    print(f"\n🔍 Sample data (first 5 rows):")
    print(df.head().to_string())
    
    print(f"\n✅ Dataset extraction completed! Saved to {output_file}")

if __name__ == "__main__":
    main()
import os
import pandas as pd
import librosa
from pathlib import Path
import re

def create_combined_dataset():
    # Process human dataset (with note labels)
    human_dataset = []
    human_dir = "./eval"
    labels_dir = os.path.join(human_dir, 'labels')

    # Process all label files from human dataset
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            audio_name = os.path.splitext(label_file)[0]
            audio_path = os.path.join(human_dir, f"{audio_name}.wav")
            
            with open(os.path.join(labels_dir, label_file), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        start_time, end_time, word = line.strip().split('\t')
                        human_dataset.append({
                            'file_path': audio_path,
                            'start_time': float(start_time),
                            'end_time': float(end_time),
                            'label': word
                        })

    # Process synthetic dataset
    syn_dataset = []
    syn_dir = "./dataset_syn"
    
    for root, _, files in os.walk(syn_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3')):
                file_path = os.path.join(root, file)
                
                # Get audio duration using librosa
                duration = librosa.get_duration(path=file_path)
                
                # Extract word from filename (before space or number)
                word = re.split(r'[ 0-9]', file)[0]
                
                syn_dataset.append({
                    'file_path': file_path,
                    'start_time': 0.0,
                    'end_time': duration,
                    'label': word
                })

    # Combine both datasets into DataFrames
    df_human = pd.DataFrame(human_dataset)
    df_syn = pd.DataFrame(syn_dataset)
    
    # Concatenate the DataFrames
    df_combined = pd.concat([df_human, df_syn], ignore_index=True)
    
    # Save to CSV
    output_path = 'eval.csv'
    df_combined.to_csv(output_path, index=False)
    print(f"Combined dataset created successfully: {output_path}")
    print(f"Total samples: {len(df_combined)}")
    print(f"Human samples: {len(df_human)}")
    print(f"Synthetic samples: {len(df_syn)}")

if __name__ == "__main__":
    create_combined_dataset()

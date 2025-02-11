import os
import pandas as pd
from pathlib import Path
import re

def create_dataset():
    # Define the base directory for audio files
    base_dir = "./dataset_syn"
    
    # Lists to store data
    file_paths = []
    labels = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.wav', '.mp3')):  # Add more audio extensions if needed
                # Get full file path
                file_path = os.path.join(root, file)
                
                # Extract label from filename
                # For files like "กู (เอไอ1).wav" or "กู1.wav", get the word before space or number
                label = re.split(r'[ 0-9]', file)[0]
                
                # Add to lists
                file_paths.append(file_path)
                labels.append(label)
    
    # Create DataFrame
    df = pd.DataFrame({
        'file_path': file_paths,
        'label': labels
    })
    
    # Save to CSV
    output_path = 'dataset_one.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset created successfully: {output_path}")
    print(f"Total samples: {len(df)}")

if __name__ == "__main__":
    create_dataset()

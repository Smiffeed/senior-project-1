import os
import csv
import pandas as pd

def create_test_dataset(audio_dir='./test', output_csv='test_dataset.csv'):
    """
    Create test dataset CSV from audio files and their label files
    
    Args:
        audio_dir: Directory containing audio files and labels subdirectory
        output_csv: Output CSV file path
    """
    dataset = []
    labels_dir = os.path.join(audio_dir, 'labels')

    # Check if directories exist
    if not os.path.exists(audio_dir):
        raise ValueError(f"Audio directory {audio_dir} does not exist")
    if not os.path.exists(labels_dir):
        raise ValueError(f"Labels directory {labels_dir} does not exist")

    # Process all label files
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            # Get corresponding audio file path
            audio_name = os.path.splitext(label_file)[0]
            audio_path = os.path.join(audio_dir, f"{audio_name}.wav")
            
            # Skip if audio file doesn't exist
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found for {label_file}")
                continue
            
            # Read and process label file
            with open(os.path.join(labels_dir, label_file), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                for line in lines:
                    if line.strip():  # Skip empty lines
                        try:
                            # Parse the line (assuming tab-separated format)
                            start_time, end_time, word = line.strip().split('\t')
                            
                            # Create dataset entry
                            dataset.append({
                                'File Name': audio_path,
                                'Start Time (s)': float(start_time),
                                'End Time (s)': float(end_time),
                                'Label': word,
                            })
                        except ValueError as e:
                            print(f"Error processing line in {label_file}: {line.strip()}")
                            print(f"Error: {e}")

    # Create DataFrame and save to CSV
    if dataset:
        df = pd.DataFrame(dataset)
        df.to_csv(output_csv, index=False)
        print(f"Dataset created with {len(df)} entries")
        
        # Print class distribution
        print("\nClass distribution:")
        print(df['Label'].value_counts())
    else:
        print("No data found to create dataset")

def verify_dataset(csv_file):
    """
    Verify the created dataset
    """
    df = pd.read_csv(csv_file)
    print("\nDataset verification:")
    print(f"Total entries: {len(df)}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")
    print("\nSample entries:")
    print(df.head())
    
    # Verify all audio files exist
    missing_files = []
    for file_path in df['File Name'].unique():
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("\nWarning: Missing audio files:")
        for file in missing_files:
            print(f"- {file}")
    else:
        print("\nAll audio files exist")

if __name__ == "__main__":
    # Create test dataset
    create_test_dataset()
    
    # Verify the created dataset
    verify_dataset('test_dataset.csv')

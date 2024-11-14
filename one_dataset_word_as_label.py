import os
import csv

def create_dataset(audio_dir, output_csv):
    dataset = []
    labels_dir = os.path.join(audio_dir, 'labels')

    # Process all label files
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            # Get corresponding audio file path
            audio_name = os.path.splitext(label_file)[0]
            audio_path = os.path.join(audio_dir, f"{audio_name}.wav")  # Assuming WAV files
            
            with open(os.path.join(labels_dir, label_file), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                for line in lines:
                    if line.strip():  # Skip empty lines
                        start_time, end_time, word = line.strip().split('\t')
                        dataset.append({
                            'File Name': audio_path,
                            'Start Time (s)': float(start_time),
                            'End Time (s)': float(end_time),
                            'Label': word,
                        })

    # Write dataset to CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['File Name', 'Start Time (s)', 'End Time (s)', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in dataset:
            writer.writerow(row)

    print(f"Dataset created and saved to {output_csv}")

# Usage
audio_dir = './dataset'
output_csv = 'profanity_dataset_word.csv'

create_dataset(audio_dir, output_csv)

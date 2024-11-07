import os
import csv
import random
from pydub import AudioSegment
import librosa

def create_dataset(non_profanity_dir, profanity_dir, output_csv):
    dataset = []

    # Process non-profanity audio files
    for audio_file in os.listdir(non_profanity_dir):
        if audio_file.endswith(('.wav', '.mp3', '.flac')):
            file_path = os.path.join(non_profanity_dir, audio_file)
            duration = librosa.get_duration(filename=file_path)
            
            # Add a random segment as non-profanity
            start_time = random.uniform(0, max(0, duration - 5))
            end_time = min(start_time + random.uniform(3, 5), duration)
            
            dataset.append({
                'File Name': file_path,  # Use full file path
                'Start Time (s)': round(start_time, 2),
                'End Time (s)': round(end_time, 2),
                'Label': 'None'
            })

    # Process profanity audio files
    for audio_file in os.listdir(profanity_dir):
        if audio_file.endswith(('.wav', '.mp3', '.flac')):
            audio_path = os.path.join(profanity_dir, audio_file)
            label_file = os.path.join(profanity_dir, 'labels', f"{os.path.splitext(audio_file)[0]}.txt")
            
            if os.path.exists(label_file):
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    start_time, end_time, label = line.strip().split('\t')
                    if label == "เย็ดแม่":
                        label = "profanity"
                    dataset.append({
                        'File Name': audio_path,
                        'Start Time (s)': float(start_time),
                        'End Time (s)': float(end_time),
                        'Label': label
                    })
            else:
                print(f"Warning: Label file not found for {audio_file}")

    # Shuffle the dataset
    random.shuffle(dataset)

    # Write dataset to CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['File Name', 'Start Time (s)', 'End Time (s)', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in dataset:
            writer.writerow(row)

    print(f"Dataset created and saved to {output_csv}")

# Usage
non_profanity_dir = './one_audio/non-profanity'
profanity_dir = './one_audio/Profanity'
output_csv = 'profanity_dataset.csv'

create_dataset(non_profanity_dir, profanity_dir, output_csv)

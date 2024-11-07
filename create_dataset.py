import os
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment
from pydub.utils import make_chunks
import random

def convert_to_wav(input_file, output_file):
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        return True
    except Exception as e:
        print(f"Error converting {input_file} to WAV: {str(e)}")
        return False

def convert_and_process_audio(input_file, target_sr=16000):
    try:
        # Load audio file
        audio = AudioSegment.from_file(input_file)
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Resample if necessary
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        
        # Normalize audio
        samples = samples / np.max(np.abs(samples))
        
        return samples, target_sr
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return None, None

def create_dataset(root_dir, output_csv):
    data = []
    
    for category in ['Profanity', 'non-profanity']:
        category_dir = os.path.join(root_dir, 'Audio', category)
        print(f"Checking directory: {category_dir}")
        if not os.path.exists(category_dir):
            print(f"Directory does not exist: {category_dir}")
            continue
        
        for filename in tqdm(os.listdir(category_dir), desc=f"Processing {category}"):
            file_path = os.path.join(category_dir, filename)
            
            # Process audio
            audio, sr = convert_and_process_audio(file_path)
            if audio is None:
                continue
            
            # Create new filename
            new_filename = f"{category}_{os.path.splitext(filename)[0]}.wav"
            new_file_path = os.path.join(root_dir, 'processed_audio', new_filename)
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
            
            # Save processed audio
            sf.write(new_file_path, audio, sr)
            
            # Get transcription from filename (without extension)
            transcription = os.path.splitext(filename)[0]
            
            # Assign label: 0 for non-profanity, 1 for profanity
            label = 1 if category == 'Profanity' else 0
            
            # Add to dataset
            data.append({
                'file_path': new_file_path,
                'transcription': transcription,
                'label': label
            })
    
    if not data:
        print("No data was processed. Check your directory structure and file extensions.")
        return

    # Shuffle the data randomly
    random.shuffle(data)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved to {output_csv}")
    print(f"Total processed files: {len(data)}")

if __name__ == "__main__":
    root_dir = "."  # Adjust this to your root directory if needed
    output_csv = "audio_dataset.csv"
    create_dataset(root_dir, output_csv)

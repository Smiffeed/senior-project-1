import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment

def process_audio(input_file, output_file, target_sr=16000):
    print(f"Processing {input_file}...")
    
    try:
        # Load the audio file using pydub
        audio = AudioSegment.from_file(input_file, format="m4a")
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Resample if necessary
        if audio.frame_rate != target_sr:
            print(f"Resampling from {audio.frame_rate} Hz to {target_sr} Hz")
            audio = audio.set_frame_rate(target_sr)
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        
        # Save as WAV using soundfile
        sf.write(output_file, samples, target_sr, subtype='PCM_16')
        print(f"Saved {output_file}")
        return True
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return False

def prepare_dataset(input_dir, output_dir, target_sr=16000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    success_count = 0
    total_files = 0
    
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc="Processing files"):
            if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                total_files += 1
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.wav')
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                if process_audio(input_path, output_path, target_sr):
                    success_count += 1
    
    print(f"Processed {success_count} out of {total_files} files successfully.")

if __name__ == "__main__":
    input_file = "2.m4a"
    output_file = "2.wav"
    target_sample_rate = 16000  # wav2vec2 typically uses 16kHz

    success = process_audio(input_file, output_file, target_sample_rate)
    
    if success:
        print("Audio processing completed successfully.")
    else:
        print("Audio processing failed.")

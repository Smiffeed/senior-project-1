import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

def remove_background_noise(input_path, output_path, noise_clip_duration=2):
    """
    Remove background noise from an audio file
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to save cleaned audio
        noise_clip_duration (int): Duration in seconds to sample noise from beginning
    """
    # Load the audio file
    data, rate = sf.read(input_path)
    
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    # Get noise profile from the first few seconds
    noise_sample = data[:int(rate * noise_clip_duration)]
    
    # Perform noise reduction
    reduced_noise = nr.reduce_noise(
        y=data,
        sr=rate,
        y_noise=noise_sample,
        prop_decrease=1.0,
        stationary=True
    )
    
    # Save the cleaned audio
    sf.write(output_path, reduced_noise, rate)

def batch_process_audio(input_files, output_dir):
    """
    Process multiple audio files
    
    Args:
        input_files (list): List of input audio file paths
        output_dir (str): Directory to save cleaned audio files
    """
    import os
    
    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_path = os.path.join(output_dir, f"cleaned_{filename}")
        remove_background_noise(input_file, output_path)
        print(f"Processed: {filename}")

def enhance_speech(input_path, output_path, noise_clip_duration=2):
    """
    Enhance speech clarity in an audio file with minimal quality loss:
    1. Removing background noise with gentler settings
    2. Light normalization
    3. Subtle frequency enhancement for speech clarity
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to save enhanced audio
        noise_clip_duration (int): Duration in seconds to sample noise
    """
    # Load the audio file
    data, rate = sf.read(input_path)
    
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    # Step 1: Gentler noise reduction
    noise_sample = data[:int(rate * noise_clip_duration)]
    reduced_noise = nr.reduce_noise(
        y=data,
        sr=rate,
        y_noise=noise_sample,
        prop_decrease=0.75,  # Reduced from 1.0 for more natural sound
        stationary=True,
        n_std_thresh_stationary=1.5,  # More conservative threshold
    )
    
    # Step 2: Light normalization (preserve some dynamics)
    normalized = reduced_noise * 0.9 / np.max(np.abs(reduced_noise))
    
    # Step 3: Very subtle frequency enhancement
    nyquist = rate / 2
    low = 80 / nyquist   # Wider frequency range
    high = 4000 / nyquist
    b, a = butter(2, [low, high], btype='band')  # Reduced order from 4 to 2
    
    # Apply gentle filter
    filtered = filtfilt(b, a, normalized)
    
    # Step 4: Blend with emphasis on original
    enhanced = 0.85 * normalized + 0.15 * filtered  # More weight to original
    
    # Final normalization
    enhanced = enhanced * 0.95 / np.max(np.abs(enhanced))  # Slight headroom
    
    # Save the enhanced audio
    sf.write(output_path, enhanced, rate)

def batch_enhance_audio(input_files, output_dir):
    """
    Process multiple audio files with speech enhancement
    
    Args:
        input_files (list): List of input audio file paths
        output_dir (str): Directory to save enhanced audio files
    """
    import os
    
    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_path = os.path.join(output_dir, f"enhanced_{filename}")
        enhance_speech(input_file, output_path)
        print(f"Enhanced: {filename}")

def validate_for_wav2vec2(audio_path):
    """
    Validate audio file for wav2vec2 compatibility
    """
    data, rate = sf.read(audio_path)
    
    # Check sample rate (wav2vec2 typically expects 16kHz)
    if rate != 16000:
        print(f"Warning: Sample rate is {rate}Hz. wav2vec2 expects 16kHz")
    
    # Check if mono
    if len(data.shape) > 1:
        print("Warning: Audio is not mono")
    
    # Check amplitude range
    if np.max(np.abs(data)) > 1.0:
        print("Warning: Audio contains clipping")

def resample_if_needed(data, orig_rate, target_rate=16000):
    """
    Resample audio to 16kHz if needed
    """
    if orig_rate != target_rate:
        from scipy import signal
        num_samples = int(len(data) * target_rate / orig_rate)
        data = signal.resample(data, num_samples)
    return data

# Example usage
if __name__ == "__main__":
    # Single file processing
    input_path = "test1.wav"
    output_path = "test1_cleaned.wav"
    remove_background_noise(input_path, output_path)
    
    # Batch processing
    # input_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
    # output_dir = "cleaned_audio/"
    # batch_process_audio(input_files, output_dir)
    
    # Single file enhancement
    input_path = "test1_cleaned.wav"
    output_path = "test1_enhanced.wav"
    enhance_speech(input_path, output_path)
    
    # Or batch processing
    # input_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
    # output_dir = "enhanced_audio/"
    # batch_enhance_audio(input_files, output_dir)

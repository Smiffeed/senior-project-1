import os
import numpy as np
import torchaudio
import torch
import librosa
from scipy import signal
from tqdm import tqdm

class AudioNoiseReducer:
    def __init__(
        self,
        target_sr=16000,
        noise_reduce_threshold=0.05,
        noise_reduce_win_size=512,
        noise_reduce_strength=1.5
    ):
        self.target_sr = target_sr
        self.noise_reduce_threshold = noise_reduce_threshold
        self.noise_reduce_win_size = noise_reduce_win_size
        self.noise_reduce_strength = noise_reduce_strength
    
    def spectral_gating(self, audio, sr):
        """Apply spectral gating noise reduction"""
        # Convert to mono if needed
        if len(audio.shape) > 1 and audio.shape[0] > 1:
            audio = audio.mean(axis=0)
            
        # Short-time Fourier transform
        S = librosa.stft(audio, n_fft=self.noise_reduce_win_size, hop_length=self.noise_reduce_win_size//4)
        
        # Magnitude and phase
        mag, phase = librosa.magphase(S)
        
        # Estimate noise profile from lowest energy frames
        energy = np.sum(mag**2, axis=0)
        thresh = np.percentile(energy, 10)  # Use lowest 10% of frames for noise estimate
        noise_indices = np.where(energy < thresh)[0]
        
        if len(noise_indices) > 0:
            noise_profile = np.mean(mag[:, noise_indices], axis=1, keepdims=True)
            
            # Apply spectral gating
            gain = (1 - np.minimum(noise_profile * self.noise_reduce_strength / (mag + 1e-10), 1))
            
            # Apply gain
            mag_filtered = mag * gain
            
            # Reconstruct signal
            S_filtered = mag_filtered * phase
            audio_filtered = librosa.istft(S_filtered, hop_length=self.noise_reduce_win_size//4)
            
            return audio_filtered
        else:
            return audio
            
    def adaptive_noise_reduction(self, waveform, sr):
        """Apply adaptive noise reduction based on signal statistics"""
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
            
        # Ensure mono
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(axis=0)
        elif len(waveform.shape) > 1:
            waveform = waveform[0]
        
        # Apply spectral gating
        filtered = self.spectral_gating(waveform, sr)
        
        # Return as tensor if input was tensor
        if isinstance(waveform, torch.Tensor):
            return torch.from_numpy(filtered).float()
        return filtered
            
    def process_file(self, input_path, output_path=None):
        """Process a single audio file to remove noise"""
        try:
            # Load audio
            waveform, sr = torchaudio.load(input_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)
                sr = self.target_sr
            
            # Apply noise reduction
            filtered = self.adaptive_noise_reduction(waveform.numpy()[0], sr)
            filtered_tensor = torch.from_numpy(filtered).unsqueeze(0).float()
            
            # Normalize
            filtered_tensor = filtered_tensor / (torch.max(torch.abs(filtered_tensor)) + 1e-8)
            
            # Save if output path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torchaudio.save(output_path, filtered_tensor, sr)
                
            return filtered_tensor, sr
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return None, None
    
    def process_directory(self, input_dir, output_dir):
        """Process all audio files in a directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all audio files
        audio_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    audio_files.append(os.path.join(root, file))
        
        print(f"Found {len(audio_files)} audio files")
        
        # Process each file
        for input_path in tqdm(audio_files):
            # Create output path
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Process file
            self.process_file(input_path, output_path)
            
        print(f"Processed {len(audio_files)} files")

if __name__ == "__main__":
    # Create noise reducer
    reducer = AudioNoiseReducer()
    
    # Process directory
    input_dir = "./main"
    output_dir = "./main_clean"
    
    print(f"Processing files from {input_dir} to {output_dir}")
    reducer.process_directory(input_dir, output_dir)
    print("Done!")

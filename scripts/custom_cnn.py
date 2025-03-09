import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import torchaudio
import librosa
from sklearn.metrics import accuracy_score, classification_report
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Define label mapping
label_map = {
    'none': 0,
    'เย็ด': 1,
    'กู': 2,
    'มึง': 3,
    'เหี้ย': 4,
    'ควย': 5,
    'สวะ': 6,
    'หี': 7,
    'แตด': 8
}

class LightweightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Image input layer: 40 x 101 x 1
        
        # 1st convolution layer
        self.conv1 = nn.Conv2d(1, 14, kernel_size=3, stride=1, padding=1)  # Output: 40 x 101 x 14
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 20 x 51 x 14
        
        # 2nd convolution layer
        self.conv2 = nn.Conv2d(14, 28, kernel_size=3, stride=1, padding=1)  # Output: 20 x 51 x 28
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 10 x 26 x 28
        
        # 3rd convolution layer
        self.conv3 = nn.Conv2d(28, 56, kernel_size=3, stride=1, padding=1)  # Output: 10 x 26 x 56
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 5 x 13 x 56
        
        # 4th convolution layer
        self.conv4 = nn.Conv2d(56, 56, kernel_size=3, stride=1, padding=1)  # Output: 5 x 13 x 56
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1, 13), stride=1)  # Output: 5 x 1 x 56
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Softmax layer
        self.classifier = nn.Linear(56 * 5, 9)  # 9 classes as per your label_map
        
    def forward(self, x):
        # First convolution block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        # Second convolution block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        # Third convolution block
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        # Fourth convolution block
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Dropout
        x = self.dropout(x)
        
        # Classification layer
        x = self.classifier(x)
        
        return x

class AudioFeatureExtractor:
    def __init__(self):
        self.target_size = (40, 101)  # Input size as per the architecture
    
    def extract_features(self, audio_input):
        """
        Convert audio input to the required format for the CNN
        Expected input shape: (batch_size, time_steps)
        Output shape: (batch_size, 1, 40, 101)
        """
        # Convert to spectrogram
        spec = torch.stft(
            audio_input,
            n_fft=80,  # Adjust to get 40 frequency bins
            hop_length=40,
            win_length=80,
            window=torch.hann_window(80),
            return_complex=True
        )
        
        # Convert to magnitude spectrogram
        spec = torch.abs(spec)
        
        # Resize to target dimensions
        spec = F.interpolate(
            spec.unsqueeze(1),
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        
        return spec

class AudioDataset(Dataset):
    def __init__(self, df, feature_extractor, augment=False):
        self.df = df
        self.feature_extractor = feature_extractor
        self.augment = augment
        
        # Clean file paths when initializing
        self.df['file_path'] = self.df['file_path'].apply(lambda x: os.path.normpath(x).replace('\\', '/'))
    
    def __len__(self):
        """Return the number of items in the dataset"""
        return len(self.df)
    
    def preprocess_audio(self, file_path, start_time, end_time, max_length=16000):
        """Load and preprocess audio segment with Hamming window, FFT-based noise reduction, and spectral gating"""
        try:
            # Verify file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return torch.zeros(1, max_length)
            
            # Get the sample rate
            metadata = torchaudio.info(file_path)
            sr = metadata.sample_rate
            
            # Calculate frame positions
            start_frame = int(start_time * sr)
            num_frames = int((end_time - start_time) * sr)
            
            try:
                # Load the specific segment
                audio, sr = torchaudio.load(
                    file_path, 
                    frame_offset=start_frame,
                    num_frames=num_frames
                )
            except Exception as e:
                print(f"Error loading audio file {file_path}: {str(e)}")
                return torch.zeros(1, max_length)
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Convert to numpy array for processing
            audio_np = audio.squeeze().numpy()
            
            # Apply Hamming window
            window_length = len(audio_np)
            hamming_window = np.hamming(window_length)
            audio_np = audio_np * hamming_window
            
            # FFT-based processing
            fft_size = 2048
            hop_length = fft_size // 4
            
            # Compute FFT
            stft = librosa.stft(audio_np, n_fft=fft_size, hop_length=hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Spectral gating for noise reduction
            noise_threshold = np.mean(magnitude) * 1.5
            magnitude_gated = np.where(magnitude > noise_threshold, magnitude, magnitude * 0.1)
            
            # Apply simple spectral subtraction
            noise_estimate = np.mean(magnitude[:, :10], axis=1, keepdims=True)  # Use first few frames as noise estimate
            magnitude_cleaned = np.maximum(magnitude_gated - noise_estimate, 0)
            
            # Reconstruct signal
            stft_cleaned = magnitude_cleaned * np.exp(1j * phase)
            audio_np = librosa.istft(stft_cleaned, hop_length=hop_length)
            
            # Apply pre-emphasis filter
            audio_np = librosa.effects.preemphasis(audio_np)
            
            # Apply augmentation if enabled
            if self.augment:
                audio_np = self.apply_augmentation(audio_np)
            
            # Convert back to torch tensor
            audio = torch.from_numpy(audio_np).float().unsqueeze(0)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
            
            # Normalize audio
            audio = (audio - audio.mean()) / (audio.std() + 1e-8)
            
            # Pad or truncate to max_length
            if audio.shape[1] < max_length:
                audio = torch.nn.functional.pad(audio, (0, max_length - audio.shape[1]))
            else:
                audio = audio[:, :max_length]
            
            return audio
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return torch.zeros(1, max_length)
    
    def apply_augmentation(self, audio):
        """Apply complex augmentation techniques to audio data"""
        # Random combination of augmentations
        augmentation_types = np.random.choice([
            'noise', 'pitch', 'speed', 'reverb', 'time_mask'
        ], size=np.random.randint(1, 4), replace=False)
        
        for aug_type in augmentation_types:
            if aug_type == 'noise':
                # Add different types of noise
                noise_type = np.random.choice(['gaussian', 'pink', 'uniform'])
                if noise_type == 'gaussian':
                    noise_level = np.random.uniform(0.001, 0.005)
                    noise = np.random.normal(0, noise_level, len(audio))
                elif noise_type == 'pink':
                    noise = np.random.uniform(-0.003, 0.003, len(audio))
                    noise = librosa.core.pink_noise(len(audio)) * noise_level
                else:  # uniform
                    noise = np.random.uniform(-0.002, 0.002, len(audio))
                audio += noise
                
            elif aug_type == 'pitch':
                # More varied pitch shifting
                pitch_shift = np.random.uniform(-300, 300)
                audio = librosa.effects.pitch_shift(
                    audio, 
                    sr=16000, 
                    n_steps=pitch_shift/100,
                    bins_per_octave=200
                )
                
            elif aug_type == 'speed':
                # Time stretching with variable rates
                speed_factor = np.random.uniform(0.8, 1.2)
                audio = librosa.effects.time_stretch(audio, rate=speed_factor)
                
            elif aug_type == 'reverb':
                # Add simple reverb effect
                reverb_delay = np.random.randint(1000, 3000)
                decay = np.random.uniform(0.1, 0.5)
                reverb = np.exp(-decay * np.linspace(0, 1, reverb_delay))
                audio = np.convolve(audio, reverb, mode='full')[:len(audio)]
                
            elif aug_type == 'time_mask':
                # Random time masking
                mask_size = int(len(audio) * np.random.uniform(0.05, 0.15))
                mask_start = np.random.randint(0, len(audio) - mask_size)
                audio[mask_start:mask_start + mask_size] = 0
        
        # Normalize after augmentation
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        return audio
    
    def augment_short_words(self, audio, label):
        """Enhanced augmentation specifically for short words"""
        if label in ['กู', 'มึง']:  # Labels for short words
            augmented = []
            # Create multiple variations
            for _ in range(3):
                aug_audio = audio.copy()
                
                # Apply chain of augmentations
                # 1. Volume variation
                volume_factor = np.random.uniform(0.8, 1.2)
                aug_audio = aug_audio * volume_factor
                
                # 2. Time stretching with controlled range
                stretch_factor = np.random.uniform(0.85, 1.15)
                aug_audio = librosa.effects.time_stretch(aug_audio, rate=stretch_factor)
                
                # 3. Pitch shifting with finer control
                n_steps = np.random.uniform(-2, 2)
                aug_audio = librosa.effects.pitch_shift(
                    aug_audio, 
                    sr=16000, 
                    n_steps=n_steps,
                    bins_per_octave=200
                )
                
                # 4. Add subtle background noise
                noise_level = np.random.uniform(0.0005, 0.002)
                noise = np.random.normal(0, noise_level, len(aug_audio))
                aug_audio += noise
                
                # 5. Optional frequency masking
                if np.random.random() < 0.5:
                    mask_size = int(len(aug_audio) * np.random.uniform(0.05, 0.1))
                    mask_start = np.random.randint(0, len(aug_audio) - mask_size)
                    aug_audio[mask_start:mask_start + mask_size] *= np.random.uniform(0.1, 0.3)
                
                # 6. Optional time reversal
                if np.random.random() < 0.2:
                    aug_audio = np.flip(aug_audio)
                
                # Normalize
                aug_audio = aug_audio / (np.max(np.abs(aug_audio)) + 1e-6)
                augmented.append(aug_audio)
            
            return augmented
        return [audio]  # Return original if not a short word

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            file_path = row['file_path']
            
            # Add padding around segments
            padding = 0.2  # Increase from 0.1 to 0.2 seconds
            start_time = max(0, float(row['start_time']) - padding)
            end_time = float(row['end_time']) + padding
            
            # Load and preprocess audio
            audio = self.preprocess_audio(file_path, start_time, end_time)
            
            # Extract features
            features = self.feature_extractor.extract_features(audio)
            
            # Get label
            label = label_map[row['label']]
            
            return {
                'features': features,
                'label': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Error in __getitem__ for index {idx}: {str(e)}")
            return {
                'features': torch.zeros(1, 40, 101),
                'label': torch.tensor(0, dtype=torch.long)
            }

class CNNTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.feature_extractor = AudioFeatureExtractor()
        
    def train(self, train_df, val_df, config):
        """Train the model"""
        # Create datasets
        train_dataset = AudioDataset(train_df, self.feature_extractor, augment=True)
        val_dataset = AudioDataset(val_df, self.feature_extractor, augment=False)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training loop
        best_accuracy = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(config['epochs']):
            # Training phase
            self.model.train()
            epoch_loss = 0
            
            for batch in train_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Validation phase
            accuracy = self.evaluate(val_loader)
            val_accuracies.append(accuracy)
            
            # Update scheduler
            scheduler.step(accuracy)
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), config['model_path'])
            
            print(f"Epoch {epoch+1}/{config['epochs']}")
            print(f"Training Loss: {avg_loss:.4f}")
            print(f"Validation Accuracy: {accuracy:.4f}")
            
        # Plot training progress
        self.plot_training_progress(train_losses, val_accuracies)
        
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(features)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy
    
    def plot_training_progress(self, train_losses, val_accuracies):
        """Plot training progress"""
        plt.figure(figsize=(12, 4))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies)
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

def train_custom_cnn(csv_file, output_dir):
    """Main training function"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(csv_file)
    
    # Clean file paths in the dataframe
    df['file_path'] = df['file_path'].apply(lambda x: os.path.normpath(x).replace('\\', '/'))
    
    # Verify files exist
    existing_files = df[df['file_path'].apply(os.path.exists)]
    if len(existing_files) < len(df):
        print(f"Warning: {len(df) - len(existing_files)} files not found")
        df = existing_files
    
    if len(df) == 0:
        raise ValueError("No valid audio files found in the dataset")
    
    # Split dataset
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    # Initialize model
    model = LightweightCNN()
    
    # Training configuration
    config = {
        'batch_size': 32,
        'num_workers': 4,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'epochs': 50,
        'model_path': os.path.join(output_dir, 'best_model.pth')
    }
    
    # Initialize trainer
    trainer = CNNTrainer(model, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train model
    trainer.train(train_df, val_df, config)
    
    print("Training completed!")
    print(f"Best model saved to: {config['model_path']}")

if __name__ == "__main__":
    csv_file = './csv/main.csv'
    output_dir = './models/custom_cnn'
    train_custom_cnn(csv_file, output_dir)

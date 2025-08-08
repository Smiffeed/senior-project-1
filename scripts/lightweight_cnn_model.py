#!/usr/bin/env python3
"""
🎯 LIGHTWEIGHT CNN MODEL FOR THAI PROFANITY DETECTION
Implementation based on research paper architecture for audio classification.

This CNN model is specifically designed for:
1. Fast inference on short audio segments
2. Better performance on small datasets
3. Lower computational requirements than Wav2Vec2
4. Optimized for profanity word classification

Architecture based on research table:
- 4 Convolutional layers with increasing filters (14→28→56→56)
- Max pooling for dimension reduction
- Dropout for regularization
- Dense classification layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pandas as pd
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import os
import warnings
warnings.filterwarnings('ignore')

# Enhanced label mapping
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

num_labels = len(label_map)
rev_label_map = {v: k for k, v in label_map.items()}

class LightweightCNN(nn.Module):
    """
    Lightweight CNN for Thai profanity detection based on research architecture.
    
    Input: Audio spectrograms (40 x 101 x 1)
    Output: 9-class classification (8 profanity + 1 none)
    """
    
    def __init__(self, num_classes=9, dropout_rate=0.5):
        super(LightweightCNN, self).__init__()
        
        # Based on research table architecture
        # Input: 40 x 101 x 1 (mel spectrogram)
        
        # 1st Convolutional Layer: 14 filters, 3x3, [1,1] padding
        self.conv1 = nn.Conv2d(1, 14, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # [2,2] stride
        
        # 2nd Convolutional Layer: 28 filters, 3x3, [1,1] padding  
        self.conv2 = nn.Conv2d(14, 28, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # [2,2] stride
        
        # 3rd Convolutional Layer: 56 filters, 3x3, [1,1] padding
        self.conv3 = nn.Conv2d(28, 56, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # [2,2] stride
        
        # 4th Convolutional Layer: 56 filters, 3x3, [1,1] padding
        self.conv4 = nn.Conv2d(56, 56, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        # Use adaptive pooling to reduce to target size instead of fixed kernel
        self.pool4 = nn.AdaptiveAvgPool2d((5, 1))  # Adaptive pooling to (5, 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dense layer (fully connected)
        # Calculate the size after convolutions and pooling
        self.fc_input_size = self._get_conv_output_size()
        self.fc = nn.Linear(self.fc_input_size, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_conv_output_size(self):
        """Calculate the output size after all conv and pooling layers."""
        # Start with input size: 40 x 101
        h, w = 40, 101
        
        # After conv1 + pool1: stride [2,2]
        h, w = h // 2, w // 2  # 20 x 50
        
        # After conv2 + pool2: stride [2,2]  
        h, w = h // 2, w // 2  # 10 x 25
        
        # After conv3 + pool3: stride [2,2]
        h, w = h // 2, w // 2  # 5 x 12
        
        # After conv4 + adaptive pool: output is (5, 1)
        h, w = 5, 1  # Adaptive pooling forces this size
        
        return 56 * h * w  # 56 filters × 5 × 1 = 280
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Input shape: (batch_size, 1, 40, 101)
        
        # 1st Conv + Pool
        x = self.pool1(self.relu1(self.conv1(x)))  # (batch, 14, 20, 50)
        
        # 2nd Conv + Pool
        x = self.pool2(self.relu2(self.conv2(x)))  # (batch, 28, 10, 25)
        
        # 3rd Conv + Pool  
        x = self.pool3(self.relu3(self.conv3(x)))  # (batch, 56, 5, 12)
        
        # 4th Conv + Pool
        x = self.pool4(self.relu4(self.conv4(x)))  # (batch, 56, 5, 1)
        
        # Flatten for dense layer
        x = x.view(x.size(0), -1)  # (batch, 56*5*1)
        
        # Dropout + Dense
        x = self.dropout(x)
        x = self.fc(x)  # (batch, num_classes)
        
        return x

class AudioDataset(Dataset):
    """Dataset class for loading and preprocessing Thai profanity audio data."""
    
    def __init__(self, df, transform=None, target_length=16000):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_length = target_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio
        audio = self.load_audio_segment(
            row['file_path'], 
            row['start_time'], 
            row['end_time']
        )
        
        if audio is None:
            # Return a zero tensor if audio loading fails
            audio = np.zeros(self.target_length)
        
        # Convert to mel spectrogram
        mel_spec = self.audio_to_mel_spectrogram(audio)
        
        # Apply transforms if any
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        # Convert to tensor and add channel dimension
        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)  # (1, 40, 101)
        
        # Get label
        label = label_map[row['label']]
        
        return mel_spec, label
    
    def preprocess_audio(self, audio, sr=16000):
        """Advanced preprocessing: Hamming window, pre-emphasis, noise threshold, resampling, normalization, padding/truncation."""
        # Remove DC offset
        audio = audio - np.mean(audio)
        # Apply Hamming window
        if len(audio) > 1:
            audio = audio * np.hamming(len(audio))
        # Pre-emphasis filter
        audio = librosa.effects.preemphasis(audio, coef=0.97)
        # Noise thresholding
        noise_threshold = 0.005
        audio = np.where(np.abs(audio) < noise_threshold, 0, audio)
        # RMS normalization
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            audio = audio * (0.1 / rms)
        # Pad or truncate to target length
        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)), 'constant')
        elif len(audio) > self.target_length:
            start_idx = (len(audio) - self.target_length) // 2
            audio = audio[start_idx:start_idx + self.target_length]
        return audio

    def load_audio_segment(self, file_path, start_time, end_time):
        """Load and preprocess audio segment with advanced pipeline."""
        try:
            file_path = file_path.replace('\\', '/')
            if not os.path.exists(file_path):
                return None
            metadata = torchaudio.info(file_path)
            sr = metadata.sample_rate
            # Add padding (match Wav2Vec2 script)
            padding = 0.2
            start_time = max(0, start_time - padding)
            end_time = min(end_time + padding, metadata.num_frames / sr)
            audio, sr = torchaudio.load(
                file_path,
                frame_offset=int(start_time * sr),
                num_frames=int((end_time - start_time) * sr)
            )
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            # Resample to 16kHz if needed
            if sr != 16000:
                audio = torchaudio.functional.resample(audio, sr, 16000)
                sr = 16000
            audio_np = audio.squeeze().numpy()
            # Advanced preprocessing
            audio_np = self.preprocess_audio(audio_np, sr=sr)
            return audio_np
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def audio_to_mel_spectrogram(self, audio):
        """Convert audio to mel spectrogram (40 x 101)."""
        # Parameters for mel spectrogram
        n_fft = 1024
        hop_length = 160  # 10ms hop at 16kHz
        n_mels = 40
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=16000,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=0,
            fmax=8000
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
        
        # Ensure target size (40, 101)
        target_frames = 101
        if mel_spec.shape[1] < target_frames:
            # Pad with zeros
            pad_width = target_frames - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), 'constant')
        elif mel_spec.shape[1] > target_frames:
            # Truncate
            mel_spec = mel_spec[:, :target_frames]
        
        return mel_spec

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_lightweight_cnn(csv_file, output_dir, num_epochs=50, batch_size=32, learning_rate=0.001):
    """Train the lightweight CNN model."""
    
    print("🚀 Training Lightweight CNN for Thai Profanity Detection")
    print("="*60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("📊 Loading dataset...")
    df = pd.read_csv(csv_file)
    
    print(f"Dataset: {len(df)} samples")
    print("Class distribution:")
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        print(f"  {label}: {count}")
    
    # Calculate class weights
    labels = [label_map[label] for label in df['label']]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"\nClass weights:")
    for i, (label_name, weight) in enumerate(zip(label_map.keys(), class_weights)):
        print(f"  {label_name}: {weight:.3f}")
    
    # Split dataset
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    
    # Create datasets and dataloaders
    train_dataset = AudioDataset(train_df)
    val_dataset = AudioDataset(val_df)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4 if device.type == 'cuda' else 0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4 if device.type == 'cuda' else 0
    )
    
    # Initialize model
    print("\n🤖 Initializing Lightweight CNN...")
    model = LightweightCNN(num_classes=num_labels, dropout_rate=0.5)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)  # For class imbalance
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  ✅ New best validation accuracy: {val_acc:.2f}%")
        
        print("-" * 50)
    
    # Save the best model
    os.makedirs(output_dir, exist_ok=True)
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save({
            'model_state_dict': best_model_state,
            'model_config': {
                'num_classes': num_labels,
                'dropout_rate': 0.5
            },
            'label_map': label_map,
            'class_weights': class_weights,
            'best_val_acc': best_val_acc
        }, os.path.join(output_dir, 'lightweight_cnn_model.pth'))
        
        print(f"\n✅ Best model saved to {output_dir}")
        print(f"Final validation accuracy: {best_val_acc:.2f}%")
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    # Convert numeric labels back to string labels
    pred_labels = [rev_label_map[p] for p in all_preds]
    true_labels = [rev_label_map[l] for l in all_labels]
    
    # Print classification report
    report = classification_report(true_labels, pred_labels, digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Calculate additional metrics
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    print(f"\nAdditional Metrics:")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    
    return model, best_val_acc

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Lightweight CNN for Thai Profanity Detection")
    parser.add_argument("--input-csv", type=str, default="./csv/train.csv",
                       help="Path to training CSV file")
    parser.add_argument("--output-dir", type=str, default="./models/lightweight_cnn",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_csv):
        print(f"❌ Error: CSV file not found: {args.input_csv}")
        return
    
    # Train the model
    model, best_acc = train_lightweight_cnn(
        csv_file=args.input_csv,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print(f"\n🎉 Training completed! Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()

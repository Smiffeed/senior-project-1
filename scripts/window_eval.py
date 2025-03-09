import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Define label mapping (same as training)
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

rev_label_map = {v: k for k, v in label_map.items()}

def preprocess_audio_segment(file_path, start_time, end_time):
    """Load and preprocess audio segment with basic noise reduction"""
    # Reference preprocessing from fine_tune_wav2vec2_sen.py
    metadata = torchaudio.info(file_path)
    sr = metadata.sample_rate
    
    # Load the specific segment
    audio, sr = torchaudio.load(file_path, 
                              frame_offset=int(start_time * sr), 
                              num_frames=int((end_time - start_time) * sr))
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    
    # Normalize audio
    audio = (audio - audio.mean()) / (audio.std() + 1e-8)
    
    # Ensure correct shape: [batch_size, sequence_length]
    audio = audio.squeeze()  # Remove any extra dimensions
    if audio.dim() == 0:
        audio = audio.unsqueeze(0)  # Add sequence dimension if needed
    
    return audio.numpy()  # Convert to numpy array as feature extractor expects numpy input

def predict(model, feature_extractor, audio):
    """Make prediction for an audio segment"""
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1)
        
    return prediction.item(), probs.squeeze().cpu().numpy()

def plot_confusion_matrix(true_labels, pred_labels, output_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=label_map.keys(),
                yticklabels=label_map.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def evaluate_model(model_path, eval_csv, output_dir):
    """Evaluate model using sliding windows"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and feature extractor
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model.eval()
    
    # Read evaluation CSV
    df = pd.read_csv(eval_csv)
    
    all_true_labels = []
    all_pred_labels = []
    results = []
    
    # Process each file
    unique_files = df['file_path'].unique()
    for file_path in unique_files:
        # Normalize file path
        file_path = os.path.normpath(file_path).replace('\\', '/')
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            file_df = df[df['file_path'] == file_path]
            audio_length = torchaudio.info(file_path).num_frames / torchaudio.info(file_path).sample_rate
            
            # Create sliding windows
            window_size = 0.5  # seconds
            hop_length = 0.25  # seconds
            
            # Process each profanity segment in the file
            for _, row in file_df.iterrows():
                true_start = row['start_time']
                true_end = row['end_time']
                true_label = row['label']
                
                # Find all windows that overlap with this segment
                window_starts = np.arange(
                    max(0, true_start - window_size),
                    min(audio_length - window_size, true_end),
                    hop_length
                )
                
                for window_start in window_starts:
                    window_end = window_start + window_size
                    
                    # Calculate overlap with true segment
                    overlap_start = max(window_start, true_start)
                    overlap_end = min(window_end, true_end)
                    overlap_duration = overlap_end - overlap_start
                    
                    # If overlap is significant (e.g., >25% of window)
                    if overlap_duration > (window_size * 0.25):
                        # Process audio segment
                        audio_segment = preprocess_audio_segment(file_path, window_start, window_end)
                        pred_class, probabilities = predict(model, feature_extractor, audio_segment)
                        
                        # Store results
                        all_true_labels.append(label_map[true_label])
                        all_pred_labels.append(pred_class)
                        
                        # Store detailed results
                        results.append({
                            'file': os.path.basename(file_path),
                            'window_start': window_start,
                            'window_end': window_end,
                            'true_start': true_start,
                            'true_end': true_end,
                            'overlap_duration': overlap_duration,
                            'true_label': true_label,
                            'predicted_label': rev_label_map[pred_class],
                            'confidence': probabilities[pred_class]
                        })
            
            # Also process non-profanity regions
            all_profanity_times = sorted([(row['start_time'], row['end_time']) 
                                        for _, row in file_df.iterrows()])
            
            current_time = 0
            for prof_start, prof_end in all_profanity_times:
                # Process the non-profanity region before this profanity
                if current_time < prof_start:
                    for window_start in np.arange(current_time, prof_start - window_size, hop_length):
                        window_end = window_start + window_size
                        
                        # Process audio segment
                        audio_segment = preprocess_audio_segment(file_path, window_start, window_end)
                        pred_class, probabilities = predict(model, feature_extractor, audio_segment)
                        
                        # Store results
                        all_true_labels.append(label_map['none'])
                        all_pred_labels.append(pred_class)
                        
                        results.append({
                            'file': os.path.basename(file_path),
                            'window_start': window_start,
                            'window_end': window_end,
                            'true_start': window_start,
                            'true_end': window_end,
                            'overlap_duration': window_size,
                            'true_label': 'none',
                            'predicted_label': rev_label_map[pred_class],
                            'confidence': probabilities[pred_class]
                        })
                
                current_time = prof_end
            
            # Process remaining non-profanity region at the end
            if current_time < audio_length:
                for window_start in np.arange(current_time, audio_length - window_size, hop_length):
                    window_end = window_start + window_size
                    
                    # Process audio segment
                    audio_segment = preprocess_audio_segment(file_path, window_start, window_end)
                    pred_class, probabilities = predict(model, feature_extractor, audio_segment)
                    
                    # Store results for non-profanity
                    all_true_labels.append(label_map['none'])
                    all_pred_labels.append(pred_class)
                    
                    results.append({
                        'file': os.path.basename(file_path),
                        'window_start': window_start,
                        'window_end': window_end,
                        'true_start': window_start,
                        'true_end': window_end,
                        'overlap_duration': window_size,
                        'true_label': 'none',
                        'predicted_label': rev_label_map[pred_class],
                        'confidence': probabilities[pred_class]
                    })
                
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            continue
    
    # Calculate metrics
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(all_true_labels, all_pred_labels, output_dir)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'window_evaluation_results.csv'), index=False)
    
    return accuracy, results_df

if __name__ == "__main__":
    model_path = "./models/ham_audio"
    eval_csv = "./csv/eval.csv"
    output_dir = "./evaluation_results"
    
    os.makedirs(output_dir, exist_ok=True)
    accuracy, results = evaluate_model(model_path, eval_csv, output_dir)
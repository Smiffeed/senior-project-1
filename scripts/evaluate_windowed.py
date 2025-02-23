import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import pandas as pd
import numpy as np
import os
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.font_manager as fm

# Define label mapping (same as in fine-tuning)
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

# Reverse label mapping for output
rev_label_map = {v: k for k, v in label_map.items()}

def setup_thai_font():
    """Setup Thai font for matplotlib"""
    try:
        font_path = './fonts/THSarabunNew.ttf'  # Adjust path as needed
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'Cordia New'
    except:
        print("Warning: Thai font not found. Using default font.")

def preprocess_audio(file_path, start_time, end_time):
    # Load audio
    audio, sr = librosa.load(file_path, sr=16000, offset=start_time, duration=end_time-start_time)
    
    # Apply same preprocessing as in training
    # Noise reduction using librosa
    audio_reduced_noise = librosa.effects.preemphasis(audio)
    
    # Normalize audio
    audio_normalized = librosa.util.normalize(audio_reduced_noise)
    
    return audio_normalized

def evaluate_window(model, feature_extractor, file_path, start_time, end_time):
    try:
        # Process audio with same preprocessing as training
        audio = preprocess_audio(file_path, start_time, end_time)
        
        # Apply feature extractor
        inputs = feature_extractor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_label_id = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_label_id].item()
            
        return rev_label_map[predicted_label_id], confidence
        
    except Exception as e:
        print(f"Error processing window {file_path} ({start_time}-{end_time}): {str(e)}")
        return "error", 0.0

def plot_confusion_matrix(true_labels, pred_labels, labels):
    # Setup Thai font before plotting
    setup_thai_font()
    
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    plt.figure(figsize=(12, 10))
    
    # Create heatmap with Thai labels
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=labels, 
                yticklabels=labels,
                cmap='YlOrRd')
    
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig('./plots/confusion_matrix.png', 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

def main():
    # Setup Thai font
    setup_thai_font()
    
    # Load the model
    model_path = './models/ham_FFT_audio'  # Update this path to your best model
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load the windowed eval data
    df = pd.read_csv('./csv/eval_windowed.csv')
    
    # Create results DataFrame
    results = []
    
    # Process each window
    for idx, row in df.iterrows():
        predicted_label, confidence = evaluate_window(
            model,
            feature_extractor,
            row['file_path'],
            row['start_time'],
            row['end_time']
        )
        
        results.append({
            'file_path': row['file_path'],
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'true_label': row['label'],
            'predicted_label': predicted_label,
            'confidence': confidence
        })
        
        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} windows")
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv('./csv/eval_results.csv', index=False)
    
    # Modified filtering: Include cases where true label is profanity, regardless of prediction
    profanity_results = results_df[results_df['true_label'] != 'none'].copy()
    
    # Get unique profanity labels (excluding 'none')
    profanity_labels = [label for label in label_map.keys() if label != 'none']
    
    # Print misclassified cases where profanity was detected as 'none'
    none_misclassifications = profanity_results[profanity_results['predicted_label'] == 'none']
    print("\n=== Profanity Words Misclassified as 'none' ===")
    for _, row in none_misclassifications.iterrows():
        print(f"File: {row['file_path']}")
        print(f"Time: {row['start_time']:.2f}-{row['end_time']:.2f}")
        print(f"True label: {row['true_label']}")
        print(f"Confidence: {row['confidence']:.4f}\n")
    
    # For classification report and confusion matrix, replace 'none' predictions 
    # with a special label 'missed_profanity' to include in metrics
    profanity_results.loc[profanity_results['predicted_label'] == 'none', 'predicted_label'] = 'missed_profanity'
    
    # Generate classification report with modified labels
    report = classification_report(
        profanity_results['true_label'],
        profanity_results['predicted_label'],
        labels=profanity_labels + ['missed_profanity'],
        digits=4,
        zero_division=0
    )
    
    # Save and print classification report
    os.makedirs('./evaluation_results', exist_ok=True)
    with open('./evaluation_results/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write("Classification Report:\n")
        f.write(report)
    
    print("\n=== Classification Report ===")
    print(report)
    
    # Calculate overall accuracy (excluding 'none')
    correct_predictions = (profanity_results['true_label'] == profanity_results['predicted_label']).sum()
    total_predictions = len(profanity_results)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f"\nOverall Accuracy (excluding 'none'): {accuracy:.4f}")
    print(f"Total Profanity Windows: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    
    # Add binary profanity detection metrics
    all_results = pd.DataFrame(results)
    all_results['is_true_profanity'] = all_results['true_label'] != 'none'
    all_results['is_predicted_profanity'] = all_results['predicted_label'] != 'none'
    
    binary_correct = (all_results['is_true_profanity'] == all_results['is_predicted_profanity']).sum()
    binary_total = len(all_results)
    binary_accuracy = binary_correct / binary_total if binary_total > 0 else 0
    
    print("\n=== Binary Profanity Detection Metrics ===")
    print(f"Binary Accuracy: {binary_accuracy:.4f}")
    print(f"Total Windows: {binary_total}")
    print(f"Correct Binary Classifications: {binary_correct}")
    
    # Ensure plots directory exists
    os.makedirs('./plots', exist_ok=True)
    
    # Update confusion matrix plotting to include missed_profanity
    plot_confusion_matrix(
        profanity_results['true_label'].values,
        profanity_results['predicted_label'].values,
        profanity_labels + ['missed_profanity']
    )

if __name__ == "__main__":
    main() 
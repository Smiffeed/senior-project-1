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
        pass

def preprocess_audio(file_path, start_time, end_time):
    """Load and preprocess audio segment to match training."""
    try:
        file_path = file_path.replace('\\', '/')
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None

        metadata = torchaudio.info(file_path)
        sr = metadata.sample_rate
        audio_length_sec = metadata.num_frames / sr

        padding = 0.2
        start_time = max(0, start_time - padding)
        end_time = min(end_time + padding, audio_length_sec)

        if end_time <= start_time:
            return None

        audio, sr = torchaudio.load(
            file_path,
            frame_offset=int(start_time * sr),
            num_frames=int((end_time - start_time) * sr)
        )

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        audio_np = audio.squeeze().numpy()

        if len(audio_np) == 0:
            return None

        # Apply Hamming window and pre-emphasis
        audio_np = audio_np * np.hamming(len(audio_np))
        audio_np = librosa.effects.preemphasis(audio_np)

        # Simple noise reduction
        noise_threshold = 0.005
        audio_np = np.where(np.abs(audio_np) < noise_threshold, 0, audio_np)

        # Normalize
        audio_np = (audio_np - audio_np.mean()) / (audio_np.std() + 1e-8)

        return audio_np
    except Exception as e:
        print(f"Error preprocessing {file_path} ({start_time}-{end_time}): {e}")
        return None


def evaluate_window(model, feature_extractor, file_path, start_time, end_time):
    try:
        # Process audio with same preprocessing as training
        audio = preprocess_audio(file_path, start_time, end_time)
        if audio is None:
            return "error", 0.0
        
        # Apply feature extractor
        inputs = feature_extractor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=16000
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            logits = model(**inputs).logits
            predictions = torch.softmax(logits, dim=-1)
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
    model_path = './models/cw_ham_v2'  # Update this path to your best model
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load the windowed eval data
    df = pd.read_csv('./csv/eval_windowed_0.25s.csv')
    
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
    
    # Binary profanity detection metrics with detailed breakdown
    all_results = pd.DataFrame(results)
    
    # For each row, determine if it's a true profanity and if the prediction was correct
    all_results['is_true_profanity'] = all_results['true_label'] != 'none'
    all_results['is_predicted_profanity'] = all_results['predicted_label'] != 'none'
    all_results['is_correct_prediction'] = all_results['true_label'] == all_results['predicted_label']
    
    # Calculate True Positives, False Positives, True Negatives, False Negatives
    tp = ((all_results['is_true_profanity']) & 
          (all_results['is_predicted_profanity']) & 
          (all_results['is_correct_prediction'])).sum()
    
    fp = ((all_results['is_predicted_profanity']) & 
          (~all_results['is_correct_prediction'])).sum()
    
    tn = ((~all_results['is_true_profanity']) & 
          (~all_results['is_predicted_profanity'])).sum()
    
    fn = (all_results['is_true_profanity'] & 
          (~all_results['is_predicted_profanity'] | 
           ~all_results['is_correct_prediction'])).sum()
    
    # Calculate balanced accuracy
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    balanced_accuracy = (sensitivity + specificity) / 2
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity  # Same as sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n=== Binary Profanity Detection Metrics (Window-level) ===")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nDetailed Counts:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"Total Windows: {len(all_results)}")
    
    # Calculate class-wise proportions
    total_profanity = tp + fn
    total_none = tn + fp
    print("\nClass Distribution:")
    print(f"Profanity samples: {total_profanity} ({total_profanity/len(all_results):.2%})")
    print(f"None samples: {total_none} ({total_none/len(all_results):.2%})")
    
    # Add normalized word-level evaluation
    print("\n" + "="*60)
    print("NORMALIZED WORD-LEVEL EVALUATION")
    print("="*60)
    
    # Group consecutive profanity predictions into word-level detections
    def normalize_to_word_level(df):
        word_detections = []
        
        for file_path in df['file_path'].unique():
            file_data = df[df['file_path'] == file_path].sort_values('start_time')
            
            current_detection = None
            for _, row in file_data.iterrows():
                if row['predicted_label'] != 'none':
                    if (current_detection is None or 
                        row['predicted_label'] != current_detection['predicted_label'] or
                        row['start_time'] > current_detection['end_time'] + 0.1):  # Gap > 0.1s = new word
                        # Save previous detection
                        if current_detection:
                            word_detections.append(current_detection)
                        # Start new detection
                        current_detection = {
                            'file_path': file_path,
                            'start_time': row['start_time'],
                            'end_time': row['end_time'],
                            'predicted_label': row['predicted_label'],
                            'confidence': row['confidence']
                        }
                    else:
                        # Extend current detection
                        current_detection['end_time'] = row['end_time']
                        current_detection['confidence'] = max(current_detection['confidence'], row['confidence'])
            
            # Don't forget the last detection
            if current_detection:
                word_detections.append(current_detection)
        
        return pd.DataFrame(word_detections)
    
    # Get word-level predictions
    word_predictions = normalize_to_word_level(all_results[all_results['predicted_label'] != 'none'])
    
    # Get word-level ground truth (from original data, need to group true profanity)
    true_words = normalize_to_word_level(all_results[all_results['true_label'] != 'none'])
    
    # Calculate word-level metrics
    total_true_words = len(true_words)
    total_pred_words = len(word_predictions)
    
    # Count correct word-level detections (overlap-based matching)
    correct_word_detections = 0
    for _, pred in word_predictions.iterrows():
        # Check if this prediction overlaps with any true word
        overlapping_true = true_words[
            (true_words['file_path'] == pred['file_path']) &
            (true_words['start_time'] < pred['end_time']) &
            (true_words['end_time'] > pred['start_time'])
        ]
        
        # Check if labels match
        if len(overlapping_true) > 0:
            if any(overlapping_true['predicted_label'] == pred['predicted_label']):
                correct_word_detections += 1
    
    # Word-level metrics
    word_precision = correct_word_detections / total_pred_words if total_pred_words > 0 else 0
    word_recall = correct_word_detections / total_true_words if total_true_words > 0 else 0
    word_f1 = 2 * (word_precision * word_recall) / (word_precision + word_recall) if (word_precision + word_recall) > 0 else 0
    
    print(f"\n=== Word-Level Detection Metrics ===")
    print(f"Total True Words: {total_true_words}")
    print(f"Total Predicted Words: {total_pred_words}")
    print(f"Correct Word Detections: {correct_word_detections}")
    print(f"Word-Level Precision: {word_precision:.4f}")
    print(f"Word-Level Recall: {word_recall:.4f}")
    print(f"Word-Level F1-Score: {word_f1:.4f}")
    
    print(f"\n=== Comparison: Window vs Word Level ===")
    print(f"Window-level F1: {f1:.4f}")
    print(f"Word-level F1: {word_f1:.4f}")
    print(f"Difference: {f1 - word_f1:.4f}")
    
    if f1 > word_f1:
        print("⚠️  Window-level metrics are inflated due to overlapping windows")
        print("📊 Word-level metrics provide more realistic performance assessment")
    
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
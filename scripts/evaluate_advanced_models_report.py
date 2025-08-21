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
from datetime import datetime
import json
import argparse
import sys

def log_to_report(message):
    """Log message to console with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")

# Define label mapping (same as in fine-tuning)
label_map = {
    'none': 0,
    'เย็ด': 1,
    'กู': 2,
    'มึง': 3,
    'เหี้ย': 4
}

# Reverse label mapping for output
rev_label_map = {v: k for k, v in label_map.items()}

def create_output_structure(base_dir="evaluation_reports"):
    """Create organized output directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"evaluation_{timestamp}")
    
    subdirs = {
        'base': output_dir,
        'plots': os.path.join(output_dir, 'plots'),
        'reports': os.path.join(output_dir, 'reports'),
        'data': os.path.join(output_dir, 'data'),
        'confusion_matrices': os.path.join(output_dir, 'plots', 'confusion_matrices'),
        'roc_curves': os.path.join(output_dir, 'plots', 'roc_curves'),
        'det_curves': os.path.join(output_dir, 'plots', 'det_curves'),
        'analysis': os.path.join(output_dir, 'analysis')
    }
    
    # Create all directories
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    return output_dir, subdirs

def setup_thai_font():
    """Setup Thai font for matplotlib"""
    try:
        font_path = './fonts/THSarabunNew.ttf'  # Adjust path as needed
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'Cordia New'
    except:
        pass

def advanced_preprocess_audio(file_path, start_time, end_time):
    """
    Advanced audio preprocessing that matches advanced_model_training.py preprocessing.
    Includes enhanced noise reduction, spectral processing, and normalization.
    """
    try:
        file_path = file_path.replace('\\', '/')
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None

        # Load audio metadata
        metadata = torchaudio.info(file_path)
        sr = metadata.sample_rate
        audio_length_sec = metadata.num_frames / sr

        # Add padding around the segment (matching training)
        padding = 0.2
        start_time = max(0, start_time - padding)
        end_time = min(end_time + padding, audio_length_sec)

        if end_time <= start_time:
            return None

        # Load audio segment
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

        audio_np = audio.squeeze().numpy()

        if len(audio_np) == 0:
            return None

        # ADVANCED PREPROCESSING PIPELINE (matching advanced_model_training.py)
        
        # 1. Pre-emphasis filter (same as training)
        audio_np = librosa.effects.preemphasis(audio_np, coef=0.97)
        
        # 2. Advanced noise reduction using spectral subtraction
        if len(audio_np) > 1024:
            # Estimate noise from first and last 10% of signal
            noise_start = audio_np[:len(audio_np)//10]
            noise_end = audio_np[-len(audio_np)//10:]
            noise_estimate = np.concatenate([noise_start, noise_end])
            noise_power = np.var(noise_estimate)
            
            # Apply spectral subtraction
            stft = librosa.stft(audio_np, n_fft=512, hop_length=256)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise spectrum
            noise_magnitude = np.mean(magnitude[:, :magnitude.shape[1]//10], axis=1, keepdims=True)
            
            # Spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            beta = 0.01  # Spectral floor
            enhanced_magnitude = magnitude - alpha * noise_magnitude
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
            
            # Reconstruct audio
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            audio_np = librosa.istft(enhanced_stft, hop_length=256)
        
        # 3. Apply Hamming window for spectral shaping
        if len(audio_np) > 1:
            audio_np = audio_np * np.hamming(len(audio_np))
        
        # 4. Advanced noise gate (threshold-based noise reduction)
        noise_threshold = 0.005
        rms_window = 512
        for i in range(0, len(audio_np) - rms_window, rms_window // 2):
            window = audio_np[i:i + rms_window]
            rms = np.sqrt(np.mean(window ** 2))
            if rms < noise_threshold:
                audio_np[i:i + rms_window] *= 0.1  # Reduce noise instead of zeroing
        
        # 5. RMS normalization (matching training)
        rms = np.sqrt(np.mean(audio_np ** 2))
        if rms > 0:
            target_rms = 0.1  # Target RMS level
            audio_np = audio_np * (target_rms / rms)
        
        # 6. Dynamic range compression
        # Apply soft compression to reduce dynamic range
        threshold = 0.5
        ratio = 4.0
        audio_abs = np.abs(audio_np)
        mask = audio_abs > threshold
        compressed = np.copy(audio_np)
        compressed[mask] = np.sign(audio_np[mask]) * (
            threshold + (audio_abs[mask] - threshold) / ratio
        )
        audio_np = compressed
        
        # 7. High-pass filter to remove low-frequency noise
        if len(audio_np) > 100:
            # Simple high-pass filter using difference
            audio_np = librosa.effects.preemphasis(audio_np, coef=0.95)
        
        # 8. Z-score normalization (final step, matching training)
        mean = np.mean(audio_np)
        std = np.std(audio_np)
        if std > 1e-8:
            audio_np = (audio_np - mean) / std
        else:
            audio_np = audio_np - mean
        
        # 9. Ensure consistent length (pad or truncate to reasonable length)
        target_length = 16000  # 1 second at 16kHz
        if len(audio_np) < target_length:
            # Zero-pad
            audio_np = np.pad(audio_np, (0, target_length - len(audio_np)), 'constant')
        elif len(audio_np) > target_length:
            # Truncate from center to preserve most important part
            start_idx = (len(audio_np) - target_length) // 2
            audio_np = audio_np[start_idx:start_idx + target_length]

        return audio_np
        
    except Exception as e:
        print(f"Error in advanced preprocessing {file_path} ({start_time}-{end_time}): {e}")
        return None

def evaluate_window(model, feature_extractor, file_path, start_time, end_time, threshold=0.5):
    try:
        # Process audio with advanced preprocessing matching training
        audio = advanced_preprocess_audio(file_path, start_time, end_time)
        if audio is None:
            return "error", 0.0, 0.0
        
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
            
            # Calculate profanity probability (sum of all non-'none' classes)
            profanity_prob = predictions[0][1:].sum().item()  # Skip index 0 which is 'none'
            
            # Apply threshold for binary classification
            if profanity_prob >= threshold:
                # Find the most likely profanity class
                profanity_predictions = predictions[0][1:]  # Exclude 'none'
                predicted_profanity_id = torch.argmax(profanity_predictions).item() + 1  # +1 to account for skipping 'none'
                predicted_label = rev_label_map[predicted_profanity_id]
                confidence = predictions[0][predicted_profanity_id].item()
            else:
                # Classify as 'none'
                predicted_label = 'none'
                confidence = predictions[0][0].item()  # Confidence for 'none'
            
        return predicted_label, confidence, profanity_prob
        
    except Exception as e:
        print(f"Error processing window {file_path} ({start_time}-{end_time}): {str(e)}")
        return "error", 0.0, 0.0

def plot_confusion_matrix(true_labels, pred_labels, labels, output_dirs=None, filename='confusion_matrix_advanced.png'):
    # Setup Thai font before plotting
    setup_thai_font()
    
    try:
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)
        plt.figure(figsize=(12, 10))
        
        # Create heatmap with Thai labels
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=labels, 
                    yticklabels=labels,
                    cmap='YlOrRd')
        
        plt.title('Word-Level Confusion Matrix (Advanced Preprocessing)', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot to organized directory
        if output_dirs and 'confusion_matrices' in output_dirs:
            save_path = os.path.join(output_dirs['confusion_matrices'], filename)
        else:
            os.makedirs('./plots', exist_ok=True)
            save_path = f'./plots/{filename}'
            
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✅ Confusion matrix saved to: {save_path}")
        
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        print("Skipping confusion matrix generation due to insufficient data variety")

def calculate_iou(pred_start, pred_end, true_start, true_end):
    """
    Calculate Intersection over Union (IoU) for two time intervals.
    
    Args:
        pred_start, pred_end: Predicted time interval
        true_start, true_end: Ground truth time interval
    
    Returns:
        IoU score (float between 0 and 1)
    """
    # Calculate intersection
    intersection_start = max(pred_start, true_start)
    intersection_end = min(pred_end, true_end)
    
    # Check if there's actually an intersection
    if intersection_start >= intersection_end:
        return 0.0
    
    intersection_duration = intersection_end - intersection_start
    
    # Calculate union
    union_start = min(pred_start, true_start)
    union_end = max(pred_end, true_end)
    union_duration = union_end - union_start
    
    # Calculate IoU
    if union_duration == 0:
        return 0.0
    
    iou = intersection_duration / union_duration
    return iou

def evaluate_word_level_iou(predictions_df, ground_truth_df, iou_threshold=0.5):
    """
    Evaluate word-level predictions using IoU with a specified threshold.
    
    Args:
        predictions_df: DataFrame with predicted words (columns: file_path, start_time, end_time, predicted_label)
        ground_truth_df: DataFrame with ground truth words (columns: file_path, start_time, end_time, true_label)
        iou_threshold: IoU threshold for considering a prediction as correct (default: 0.5)
    
    Returns:
        Dictionary with IoU-based metrics
    """
    if len(predictions_df) == 0 or len(ground_truth_df) == 0:
        return {
            'iou_precision': 0.0,
            'iou_recall': 0.0,
            'iou_f1': 0.0,
            'total_predictions': len(predictions_df),
            'total_ground_truth': len(ground_truth_df),
            'correct_predictions_iou': 0,
            'mean_iou': 0.0,
            'iou_threshold': iou_threshold
        }
    
    correct_predictions = 0
    total_iou_sum = 0.0
    prediction_matches = []
    
    # For each prediction, find the best matching ground truth
    for pred_idx, pred_row in predictions_df.iterrows():
        best_iou = 0.0
        best_match = None
        
        # Find ground truth words in the same file
        same_file_gt = ground_truth_df[ground_truth_df['file_path'] == pred_row['file_path']]
        
        for gt_idx, gt_row in same_file_gt.iterrows():
            # Calculate IoU between prediction and ground truth
            iou = calculate_iou(
                pred_row['start_time'], pred_row['end_time'],
                gt_row['start_time'], gt_row['end_time']
            )
            
            # Check if this is the best match so far
            if iou > best_iou:
                best_iou = iou
                best_match = {
                    'pred_label': pred_row['predicted_label'],
                    'true_label': gt_row['true_label'],
                    'iou': iou,
                    'pred_idx': pred_idx,
                    'gt_idx': gt_idx
                }
        
        # Record the match information
        prediction_matches.append({
            'pred_idx': pred_idx,
            'best_iou': best_iou,
            'best_match': best_match,
            'file_path': pred_row['file_path'],
            'pred_start': pred_row['start_time'],
            'pred_end': pred_row['end_time'],
            'pred_label': pred_row['predicted_label']
        })
        
        total_iou_sum += best_iou
        
        # Check if prediction is correct (IoU >= threshold AND labels match)
        if best_match and best_iou >= iou_threshold and best_match['pred_label'] == best_match['true_label']:
            correct_predictions += 1
    
    # Calculate metrics
    total_predictions = len(predictions_df)
    total_ground_truth = len(ground_truth_df)
    
    iou_precision = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    iou_recall = correct_predictions / total_ground_truth if total_ground_truth > 0 else 0.0
    iou_f1 = 2 * (iou_precision * iou_recall) / (iou_precision + iou_recall) if (iou_precision + iou_recall) > 0 else 0.0
    mean_iou = total_iou_sum / total_predictions if total_predictions > 0 else 0.0
    
    return {
        'iou_precision': iou_precision,
        'iou_recall': iou_recall,
        'iou_f1': iou_f1,
        'total_predictions': total_predictions,
        'total_ground_truth': total_ground_truth,
        'correct_predictions_iou': correct_predictions,
        'mean_iou': mean_iou,
        'iou_threshold': iou_threshold,
        'prediction_matches': prediction_matches
    }

def plot_binary_confusion_matrix(true_labels, pred_labels, threshold, output_dirs=None):
    """Create binary confusion matrix for profane vs non-profane classification"""
    # Setup Thai font before plotting
    setup_thai_font()
    
    try:
        # Convert to binary labels
        binary_true = ['Profane' if label != 'none' else 'Non-Profane' for label in true_labels]
        binary_pred = ['Profane' if label != 'none' else 'Non-Profane' for label in pred_labels]
        
        cm = confusion_matrix(binary_true, binary_pred, labels=['Non-Profane', 'Profane'])
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=['Non-Profane', 'Profane'], 
                    yticklabels=['Non-Profane', 'Profane'],
                    cmap='Blues')
        
        plt.title(f'Binary Confusion Matrix (Threshold: {threshold})', fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Calculate and display percentages
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = (cm[i, j] / total) * 100
                plt.text(j + 0.5, i + 0.7, f'{percentage:.1f}%', 
                        ha='center', va='center', fontsize=10, color='red')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot to organized directory
        filename = f'binary_confusion_matrix_threshold_{threshold}.png'
        if output_dirs and 'confusion_matrices' in output_dirs:
            save_path = os.path.join(output_dirs['confusion_matrices'], filename)
        else:
            os.makedirs('./plots', exist_ok=True)
            save_path = f'./plots/{filename}'
            
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✅ Binary confusion matrix saved to: {save_path}")
        
        # Print binary classification metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        print(f"\n=== Binary Classification Metrics (Threshold: {threshold}) ===")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 
                'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        
    except Exception as e:
        print(f"Error creating binary confusion matrix: {e}")
        return None

def main():
    # Setup Thai font
    setup_thai_font()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Comprehensive Profanity Detection Evaluation with Organized Output')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Threshold for binary profanity classification (default: 0.5)')
    parser.add_argument('--model_path', type=str, default='./models/5_class_profanity_best_model',
                       help='Path to the trained model')
    parser.add_argument('--csv_file', type=str, default='./csv/eval_0.5s/stride_0.25s_hybrid.csv',
                       help='Path to the CSV file with windowed evaluation data')
    parser.add_argument('--output_dir', type=str, default='evaluation_reports',
                       help='Base output directory for organized results (default: evaluation_reports)')
    
    args = parser.parse_args()
    threshold = args.threshold
    model_path = args.model_path
    csv_file = args.csv_file
    
    # Create organized output structure
    print("🗂️ Creating organized output structure...")
    output_dir, output_dirs = create_output_structure(args.output_dir)
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Subdirectories created:")
    for name, path in output_dirs.items():
        print(f"   • {name}: {os.path.relpath(path)}")
    print()
    
    print(f"🎯 COMPREHENSIVE PROFANITY DETECTION EVALUATION")
    print(f"📈 Using binary classification threshold: {threshold}")
    print(f"📂 Using CSV file: {csv_file}")
    print(f"🤖 Using model: {model_path}")
    print(f"📊 This script will generate:")
    print(f"   • Comprehensive evaluation report")
    print(f"   • Binary confusion matrix (Profane vs Non-Profane)")
    print(f"   • Multi-class confusion matrix (specific profanity words)")
    print(f"   • ROC curve with current threshold marked")
    print(f"   • DET curve for binary classification")
    print(f"   • IoU analysis results")
    print(f"   • Detailed CSV outputs with predictions")
    print(f"   • Classification reports")
    print()
    
    # Create a comprehensive report file
    report_lines = []
    def log_to_report(text):
        print(text)
        report_lines.append(text)
    
    log_to_report("=" * 80)
    log_to_report("🎯 COMPREHENSIVE PROFANITY DETECTION EVALUATION REPORT")
    log_to_report("=" * 80)
    log_to_report(f"Evaluation Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_to_report(f"Model Path: {model_path}")
    log_to_report(f"CSV File: {csv_file}")
    log_to_report(f"Threshold: {threshold}")
    log_to_report(f"Output Directory: {output_dir}")
    log_to_report("")
    
    # Load the model (support for advanced models)

    # Try to detect model type and load accordingly
    if os.path.exists(f"{model_path}/config.json"):
        # Load as HuggingFace model
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        print(f"Loaded HuggingFace Wav2Vec2 model from {model_path}")
    else:
        # Try to load as PyTorch model (for CNN-LSTM, Transformer, etc.)
        print(f"Model path {model_path} not found or not a HuggingFace model")
        print("Please update model_path to point to your trained model")
        return
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    print("Using ADVANCED preprocessing pipeline matching advanced_model_training.py")
    
    # Load the windowed eval data
    print(f"Loading evaluation data from: {csv_file}")
    
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        print("Available CSV files in ./csv/:")
        csv_dir = "./csv/"
        if os.path.exists(csv_dir):
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            for f in sorted(csv_files):
                print(f"   {csv_dir}{f}")
        print("\nExample usage:")
        print(f"   python evaluate_advanced_models.py --csv_file csv/eval_windowed_0.3s.csv")
        return
    
    df = pd.read_csv(csv_file)

    # Extract window information from filename for reporting
    import re
    window_info = "unknown"
    if "windowed" in csv_file:
        # Extract window size from filename like "eval_windowed_0.25s.csv"
        match = re.search(r'windowed_(\d+\.?\d*)s', csv_file)
        if match:
            window_info = f"{match.group(1)}s windows"
    elif "stride" in csv_file:
        # Extract stride info from filename like "stride_0.25s.csv"
        stride_match = re.search(r'stride_(\d+\.?\d*)s', csv_file)
        if stride_match:
            window_info = f"0.5s windows with {stride_match.group(1)}s stride"

    print(f"✅ Loaded {len(df)} windowed samples for evaluation")
    print(f"📏 Window configuration: {window_info}")
    
    # Create results DataFrame
    results = []
    
    # Process each window
    total_windows = len(df)
    for idx, row in df.iterrows():
        predicted_label, confidence, profanity_prob = evaluate_window(
            model,
            feature_extractor,
            row['file_path'],
            row['start_time'],
            row['end_time'],
            threshold
        )
        
        results.append({
            'file_path': row['file_path'],
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'true_label': row['label'],
            'predicted_label': predicted_label,
            'confidence': confidence,
            'profanity_prob': profanity_prob
        })
        
        # Print progress
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{total_windows} windows ({(idx + 1)/total_windows*100:.1f}%)")
    
    # Convert results to DataFrame and save to organized output
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dirs['data'], 'eval_results_advanced.csv')
    results_df.to_csv(results_path, index=False)
    
    log_to_report(f"Saved detailed results to: {results_path}")
    
    # Modified filtering: Include cases where true label is profanity, regardless of prediction
    profanity_results = results_df[results_df['true_label'] != 'none'].copy()
    
    # Get unique profanity labels (excluding 'none')
    profanity_labels = [label for label in label_map.keys() if label != 'none']
    
    # Print misclassified cases where profanity was detected as 'none'
    none_misclassifications = profanity_results[profanity_results['predicted_label'] == 'none']
    print(f"\n=== Profanity Words Misclassified as 'none' ({len(none_misclassifications)} cases) ===")
    for i, (_, row) in enumerate(none_misclassifications.iterrows()):
        if i < 10:  # Show first 10 cases
            print(f"File: {os.path.basename(row['file_path'])}")
            print(f"Time: {row['start_time']:.2f}-{row['end_time']:.2f}")
            print(f"True label: {row['true_label']}")
            print(f"Confidence: {row['confidence']:.4f}\n")
        elif i == 10:
            print(f"... and {len(none_misclassifications) - 10} more cases")
            break
    
    # Filter out 'none' predictions from profanity results for classification report
    profanity_results_filtered = profanity_results[profanity_results['predicted_label'] != 'none'].copy()
    
    # Generate classification report with profanity labels only
    if len(profanity_results_filtered) > 0:
        report = classification_report(
            profanity_results_filtered['true_label'],
            profanity_results_filtered['predicted_label'],
            labels=profanity_labels,
            digits=4,
            zero_division=0
        )
        
        # Save and print classification report to organized output
        report_path = os.path.join(output_dirs['reports'], 'classification_report_advanced.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Classification Report (Advanced Preprocessing):\n")
            f.write(report)
        
        log_to_report("\n=== Classification Report (Advanced Preprocessing) ===")
        log_to_report(report)
        
        # Calculate overall accuracy (excluding 'none' predictions)
        correct_predictions = (profanity_results_filtered['true_label'] == profanity_results_filtered['predicted_label']).sum()
        total_predictions = len(profanity_results_filtered)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"\nOverall Accuracy (excluding 'none' predictions): {accuracy:.4f}")
        print(f"Total Profanity Windows with Profanity Predictions: {total_predictions}")
        print(f"Correct Predictions: {correct_predictions}")
        
        # Also report missed profanity (for information only)
        missed_profanity = len(profanity_results[profanity_results['predicted_label'] == 'none'])
        total_profanity_windows = len(profanity_results)
        print(f"Missed Profanity (predicted as 'none'): {missed_profanity}/{total_profanity_windows}")
    

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

    # Calculate class-wise proportions FIRST (before using them)
    total_profanity = tp + fn
    total_none = tn + fp

    print("\n=== Binary Profanity Detection Metrics (Window-level, Advanced Preprocessing) ===")
    print(f"Using threshold: {threshold}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    
    # Generate binary confusion matrix
    print(f"\n=== Creating Binary Confusion Matrix ===")
    binary_metrics = plot_binary_confusion_matrix(
        all_results['true_label'].values,
        all_results['predicted_label'].values,
        threshold
    )
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nDetailed Counts:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"Total Windows: {len(all_results)}")

    # ADD: Explicit Binary Classification Accuracy (weighted for imbalance)
    binary_accuracy = (tp + tn) / len(all_results)
    print(f"\n🎯 BINARY CLASSIFICATION ACCURACY:")
    print(f"Simple Binary Accuracy: {binary_accuracy:.4f}")
    print(f"Balanced Binary Accuracy: {balanced_accuracy:.4f}")
    print(f"(Balanced accounts for class imbalance: {total_profanity/len(all_results):.1%} profane vs {total_none/len(all_results):.1%} clean)")

    print("\nClass Distribution:")
    print(f"Profanity samples: {total_profanity} ({total_profanity/len(all_results):.2%})")
    print(f"None samples: {total_none} ({total_none/len(all_results):.2%})")

    # ROC/AUC calculation and plot
    from sklearn.metrics import roc_curve, roc_auc_score
    # For ROC, need binary ground truth and probability scores for 'profanity' (not 'none')
    # Use the profanity_prob field which contains the actual probability of being profanity
    y_true = all_results['is_true_profanity'].astype(int).values
    y_scores = all_results['profanity_prob'].values

    # Compute ROC curve and AUC
    try:
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)
        print(f"\n=== ROC/AUC for Binary Profanity Detection ===")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Current threshold: {threshold}")

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        # Mark the current threshold on the ROC curve
        current_threshold_idx = np.argmin(np.abs(thresholds_roc - threshold))
        plt.plot(fpr[current_threshold_idx], tpr[current_threshold_idx], 'ro', markersize=8, 
                label=f'Current threshold ({threshold})')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Binary Profanity Detection (Threshold: {threshold})')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        # Save ROC curve to organized output
        roc_path = os.path.join(output_dirs['roc_curves'], f'roc_curve_binary_profanity_threshold_{threshold}.png')
        plt.savefig(roc_path, dpi=300)
        plt.close()
        log_to_report(f"✅ ROC curve saved to: {roc_path}")
        # DET curve calculation and plot
        from scipy.stats import norm
        import matplotlib.ticker as mticker
        print("\n=== DET Curve for Binary Profanity Detection ===")
        # Miss rate = 1 - TPR, False alarm rate = FPR
        miss_rate = 1 - tpr
        false_alarm_rate = fpr
        plt.figure(figsize=(8, 6))
        plt.plot(norm.ppf(false_alarm_rate), norm.ppf(miss_rate), label='DET curve')
        plt.xlabel('False Alarm Rate (FAR) [norm dev]')
        plt.ylabel('Miss Rate (MR) [norm dev]')
        plt.title('DET Curve for Binary Profanity Detection')
        plt.grid(True)
        plt.legend(loc='upper right')
        # Set axis ticks to show rates
        ticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]
        tick_labels = [str(t) for t in ticks]
        plt.xticks(norm.ppf(ticks), tick_labels, rotation=45)
        plt.yticks(norm.ppf(ticks), tick_labels)
        plt.tight_layout()
        
        # Save DET curve to organized output
        det_path = os.path.join(output_dirs['det_curves'], 'det_curve_binary_profanity.png')
        plt.savefig(det_path, dpi=300)
        plt.close()
        log_to_report(f"✅ DET curve saved to: {det_path}")
    except Exception as e:
        print(f"⚠️ Could not compute ROC/AUC: {e}")
    
    # ADD: Multi-class Classification Accuracy (for profanity words only)
    print(f"\n=== MULTI-CLASS PROFANITY CLASSIFICATION ACCURACY ===")
    profanity_only = all_results[all_results['true_label'] != 'none'].copy()
    
    if len(profanity_only) > 0:
        # Simple accuracy for profanity classification
        correct_profanity_class = (profanity_only['true_label'] == profanity_only['predicted_label']).sum()
        profanity_class_accuracy = correct_profanity_class / len(profanity_only)
        
        # Weighted accuracy for profanity classification (accounts for class imbalance within profanity)
        from sklearn.metrics import accuracy_score
        weighted_profanity_accuracy = accuracy_score(
            profanity_only['true_label'], 
            profanity_only['predicted_label'],
            sample_weight=None  # Will calculate class weights manually
        )
        
        # Calculate class weights for profanity types
        profanity_class_counts = profanity_only['true_label'].value_counts()
        total_profanity_samples = len(profanity_only)
        
        print(f"Total Profanity Samples: {total_profanity_samples}")
        print(f"Correct Profanity Classifications: {correct_profanity_class}")
        print(f"Simple Profanity Classification Accuracy: {profanity_class_accuracy:.4f}")
        
        # Calculate balanced accuracy for each profanity class
        profanity_class_balanced_scores = []
        print(f"\nPer-Class Profanity Accuracy:")
        for label in profanity_labels:
            if label in profanity_class_counts:
                # True positives for this specific class
                class_tp = ((profanity_only['true_label'] == label) & 
                           (profanity_only['predicted_label'] == label)).sum()
                # Total instances of this class
                class_total = (profanity_only['true_label'] == label).sum()
                # Class accuracy
                class_acc = class_tp / class_total if class_total > 0 else 0
                class_weight = class_total / total_profanity_samples
                profanity_class_balanced_scores.append(class_acc)
                
                print(f"  {label}: {class_tp}/{class_total} = {class_acc:.3f} (weight: {class_weight:.3f})")
        
        # Weighted profanity classification accuracy
        if profanity_class_balanced_scores:
            weighted_profanity_class_accuracy = np.mean(profanity_class_balanced_scores)
            print(f"\nBalanced Profanity Classification Accuracy: {weighted_profanity_class_accuracy:.4f}")
        else:
            weighted_profanity_class_accuracy = 0.0
    else:
        profanity_class_accuracy = 0.0
        weighted_profanity_class_accuracy = 0.0
        print("No profanity samples found for classification accuracy")
    
    # Add normalized word-level evaluation
    print("\n" + "="*60)
    print("NORMALIZED WORD-LEVEL EVALUATION (Advanced Preprocessing)")
    print("="*60)
    
    # Group consecutive profanity predictions into word-level detections
    def normalize_to_word_level(df, label_column='predicted_label'):
        word_detections = []
        
        for file_path in df['file_path'].unique():
            file_data = df[df['file_path'] == file_path].sort_values('start_time')
            
            current_detection = None
            for _, row in file_data.iterrows():
                if row[label_column] != 'none':
                    if (current_detection is None or 
                        row[label_column] != current_detection[label_column] or
                        row['start_time'] > current_detection['end_time'] + 0.1):  # Gap > 0.1s = new word
                        # Save previous detection
                        if current_detection:
                            word_detections.append(current_detection)
                        # Start new detection
                        current_detection = {
                            'file_path': file_path,
                            'start_time': row['start_time'],
                            'end_time': row['end_time'],
                            label_column: row[label_column],
                            'confidence': row.get('confidence', 1.0)
                        }
                    else:
                        # Extend current detection
                        current_detection['end_time'] = row['end_time']
                        if 'confidence' in row:
                            current_detection['confidence'] = max(current_detection['confidence'], row['confidence'])
            
            # Don't forget the last detection
            if current_detection:
                word_detections.append(current_detection)
        
        return pd.DataFrame(word_detections)
    
    # Get word-level predictions
    word_predictions = normalize_to_word_level(all_results[all_results['predicted_label'] != 'none'], 'predicted_label')
    
    # Get word-level ground truth (from original data, need to group true profanity)
    true_words = normalize_to_word_level(all_results[all_results['true_label'] != 'none'], 'true_label')
    
    # Calculate word-level metrics
    total_true_words = len(true_words)
    total_pred_words = len(word_predictions)
    
    # Count correct word-level detections (overlap-based matching)
    correct_word_detections = 0
    if total_pred_words > 0 and total_true_words > 0:
        for _, pred in word_predictions.iterrows():
            # Check if this prediction overlaps with any true word
            overlapping_true = true_words[
                (true_words['file_path'] == pred['file_path']) &
                (true_words['start_time'] < pred['end_time']) &
                (true_words['end_time'] > pred['start_time'])
            ]
            
            # Check if labels match - use correct column names
            if len(overlapping_true) > 0:
                # Get the true labels from overlapping true words
                overlapping_labels = overlapping_true['true_label'].tolist()
                
                if pred['predicted_label'] in overlapping_labels:
                    correct_word_detections += 1
    
    # Word-level metrics
    word_precision = correct_word_detections / total_pred_words if total_pred_words > 0 else 0
    word_recall = correct_word_detections / total_true_words if total_true_words > 0 else 0
    word_f1 = 2 * (word_precision * word_recall) / (word_precision + word_recall) if (word_precision + word_recall) > 0 else 0
    
    print(f"\n=== Word-Level Detection Metrics (Advanced Preprocessing) ===")
    print(f"Total True Words: {total_true_words}")
    print(f"Total Predicted Words: {total_pred_words}")
    print(f"Correct Word Detections: {correct_word_detections}")
    print(f"Word-Level Precision: {word_precision:.4f}")
    print(f"Word-Level Recall: {word_recall:.4f}")
    print(f"Word-Level F1-Score: {word_f1:.4f}")
    
    # ADD: IoU-based Word-Level Evaluation with 0.5 threshold
    print(f"\n=== IoU-based Word-Level Evaluation (Threshold: 0.5) ===")
    
    # Prepare DataFrames for IoU evaluation
    if total_pred_words > 0 and total_true_words > 0:
        # Ensure column names are consistent
        predictions_for_iou = word_predictions.copy()
        if 'predicted_label' not in predictions_for_iou.columns:
            predictions_for_iou = predictions_for_iou.rename(columns={'predicted_label': 'predicted_label'})
        
        ground_truth_for_iou = true_words.copy()  
        if 'true_label' not in ground_truth_for_iou.columns:
            ground_truth_for_iou = ground_truth_for_iou.rename(columns={'true_label': 'true_label'})
        
        # Evaluate with IoU threshold of 0.5
        iou_results = evaluate_word_level_iou(predictions_for_iou, ground_truth_for_iou, iou_threshold=0.5)
        
        print(f"IoU Threshold: {iou_results['iou_threshold']}")
        print(f"Total Predicted Words: {iou_results['total_predictions']}")
        print(f"Total Ground Truth Words: {iou_results['total_ground_truth']}")
        print(f"Correct Predictions (IoU ≥ 0.5 + Label Match): {iou_results['correct_predictions_iou']}")
        print(f"IoU-based Precision: {iou_results['iou_precision']:.4f}")
        print(f"IoU-based Recall: {iou_results['iou_recall']:.4f}")
        print(f"IoU-based F1-Score: {iou_results['iou_f1']:.4f}")
        print(f"Mean IoU: {iou_results['mean_iou']:.4f}")
        
        # Additional IoU analysis - different thresholds
        print(f"\n=== IoU Analysis with Different Thresholds ===")
        iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        for iou_thresh in iou_thresholds:
            iou_results_thresh = evaluate_word_level_iou(predictions_for_iou, ground_truth_for_iou, iou_threshold=iou_thresh)
            print(f"IoU ≥ {iou_thresh}: Precision={iou_results_thresh['iou_precision']:.3f}, "
                  f"Recall={iou_results_thresh['iou_recall']:.3f}, "
                  f"F1={iou_results_thresh['iou_f1']:.3f}")
        
        # Save IoU results for further analysis
        iou_matches_df = pd.DataFrame(iou_results['prediction_matches'])
        
        # Save IoU analysis results to organized output
        iou_results_path = os.path.join(output_dirs['data'], 'iou_analysis_results.csv')
        iou_matches_df.to_csv(iou_results_path, index=False)
        log_to_report(f"✅ IoU analysis results saved to: {iou_results_path}")
        
    else:
        print("No predictions or ground truth words available for IoU evaluation")
        iou_results = None
    
    # ADD NEW: Word-level accuracy against ORIGINAL ground truth
    print(f"\n=== Word-Level vs Original Ground Truth Accuracy ===")
    
    # Load original ground truth words
    try:
        original_df = pd.read_csv('./csv/eval_5labels.csv')
        # Convert original ground truth to word format (already individual words)
        original_gt_words = original_df[original_df['label'] != 'none'].copy()
        
        total_original_words = len(original_gt_words)
        correct_vs_original = 0
        
        if total_pred_words > 0 and total_original_words > 0:
            for _, pred in word_predictions.iterrows():
                # Check if this prediction overlaps with any original ground truth word
                overlapping_original = original_gt_words[
                    (original_gt_words['file_path'] == pred['file_path']) &
                    (original_gt_words['start_time'] < pred['end_time']) &
                    (original_gt_words['end_time'] > pred['start_time'])
                ]
                
                # Check if labels match with original ground truth
                if len(overlapping_original) > 0:
                    overlapping_labels = overlapping_original['label'].tolist()
                    if pred['predicted_label'] in overlapping_labels:
                        correct_vs_original += 1
        
        # Calculate metrics against original ground truth
        original_precision = correct_vs_original / total_pred_words if total_pred_words > 0 else 0
        original_recall = correct_vs_original / total_original_words if total_original_words > 0 else 0
        original_f1 = 2 * (original_precision * original_recall) / (original_precision + original_recall) if (original_precision + original_recall) > 0 else 0
        
        print(f"Total Original GT Words: {total_original_words}")
        print(f"Total Predicted Words: {total_pred_words}")
        print(f"Correct vs Original GT: {correct_vs_original}")
        print(f"Precision vs Original GT: {original_precision:.4f}")
        print(f"Recall vs Original GT: {original_recall:.4f}")
        print(f"F1-Score vs Original GT: {original_f1:.4f}")
        
        # ADD: IoU evaluation against original ground truth
        print(f"\n=== IoU-based Evaluation vs Original Ground Truth ===")
        if total_pred_words > 0:
            # Prepare original ground truth for IoU evaluation  
            original_gt_for_iou = original_gt_words.copy()
            if 'label' in original_gt_for_iou.columns:
                original_gt_for_iou = original_gt_for_iou.rename(columns={'label': 'true_label'})
            
            # Evaluate IoU against original ground truth
            iou_vs_original = evaluate_word_level_iou(predictions_for_iou, original_gt_for_iou, iou_threshold=0.5)
            
            print(f"IoU vs Original GT (threshold=0.5):")
            print(f"  Precision: {iou_vs_original['iou_precision']:.4f}")
            print(f"  Recall: {iou_vs_original['iou_recall']:.4f}")
            print(f"  F1-Score: {iou_vs_original['iou_f1']:.4f}")
            print(f"  Mean IoU: {iou_vs_original['mean_iou']:.4f}")
            print(f"  Correct Predictions: {iou_vs_original['correct_predictions_iou']}/{iou_vs_original['total_predictions']}")
            
            # Save IoU vs original results to organized output
            iou_original_matches_df = pd.DataFrame(iou_vs_original['prediction_matches'])
            iou_original_path = os.path.join(output_dirs['data'], 'iou_vs_original_analysis.csv')
            iou_original_matches_df.to_csv(iou_original_path, index=False)
            log_to_report(f"✅ IoU vs original GT results saved to: {iou_original_path}")
        else:
            iou_vs_original = None
        
        print(f"\n🎯 ACCURACY COMPARISON:")
        print(f"Windowed-based Word F1: {word_f1:.4f}")
        print(f"Original GT-based Word F1: {original_f1:.4f}")
        if 'iou_results' in locals() and iou_results:
            print(f"IoU-based F1 (vs windowed GT): {iou_results['iou_f1']:.4f}")
        if 'iou_vs_original' in locals() and iou_vs_original:
            print(f"IoU-based F1 (vs original GT): {iou_vs_original['iou_f1']:.4f}")
        print(f"Difference: {word_f1 - original_f1:.4f}")
        
        if original_f1 < word_f1:
            print("⚠️  Original GT-based metrics are more realistic")
            print("📊 This shows true performance against actual labeled data")
        
        # Detailed class-wise breakdown vs original ground truth
        print(f"\n=== Class-wise Performance vs Original Ground Truth ===")
        for label in profanity_labels:
            # Original ground truth count for this class
            original_class_count = len(original_gt_words[original_gt_words['label'] == label])
            
            # Predicted words for this class
            pred_class_words = word_predictions[word_predictions['predicted_label'] == label] if len(word_predictions) > 0 else pd.DataFrame()
            pred_class_count = len(pred_class_words)
            
            # Count correct predictions for this class vs original GT
            correct_class_vs_original = 0
            if pred_class_count > 0 and original_class_count > 0:
                for _, pred in pred_class_words.iterrows():
                    overlapping_original = original_gt_words[
                        (original_gt_words['file_path'] == pred['file_path']) &
                        (original_gt_words['label'] == label) &
                        (original_gt_words['start_time'] < pred['end_time']) &
                        (original_gt_words['end_time'] > pred['start_time'])
                    ]
                    if len(overlapping_original) > 0:
                        correct_class_vs_original += 1
            
            # Calculate class-specific metrics
            class_precision = correct_class_vs_original / pred_class_count if pred_class_count > 0 else 0
            class_recall = correct_class_vs_original / original_class_count if original_class_count > 0 else 0
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
            
            print(f"{label}: {correct_class_vs_original}/{original_class_count} recall={class_recall:.3f}, {correct_class_vs_original}/{pred_class_count} precision={class_precision:.3f}, F1={class_f1:.3f}")
        
    except Exception as e:
        print(f"Could not load original ground truth: {e}")
        original_f1 = 0
    
    print(f"\n=== Comparison: Window vs Word Level (Advanced Preprocessing) ===")
    print(f"Window-level F1: {f1:.4f}")
    print(f"Word-level F1: {word_f1:.4f}")
    print(f"Original GT Word F1: {original_f1:.4f}")
    print(f"Window vs Word difference: {f1 - word_f1:.4f}")
    print(f"Word vs Original GT difference: {word_f1 - original_f1:.4f}")
    
    if f1 > word_f1:
        print("⚠️  Window-level metrics are inflated due to overlapping windows")
    if word_f1 > original_f1:
        print("⚠️  Windowed GT metrics may be inflated due to data augmentation")
    print("📊 Original GT-based metrics provide the most realistic performance assessment")
    
    # ADD: COMPREHENSIVE ACCURACY SUMMARY
    print(f"\n" + "="*70)
    print("🎯 COMPREHENSIVE ACCURACY SUMMARY")
    print("="*70)
    
    print(f"\n1️⃣ BINARY CLASSIFICATION ACCURACY (Profane vs Non-Profane):")
    print(f"   Simple Binary Accuracy: {binary_accuracy:.4f}")
    print(f"   Balanced Binary Accuracy: {balanced_accuracy:.4f} ⭐ (Recommended for imbalanced data)")
    print(f"   Binary F1-Score: {f1:.4f}")
    
    print(f"\n2️⃣ MULTI-CLASS CLASSIFICATION ACCURACY (Which profanity word?):")
    print(f"   Simple Profanity Class Accuracy: {profanity_class_accuracy:.4f}")
    print(f"   Balanced Profanity Class Accuracy: {weighted_profanity_class_accuracy:.4f} ⭐ (Accounts for class imbalance)")
    
    print(f"\n3️⃣ WORD-LEVEL ACCURACY (Realistic performance):")
    print(f"   Word-level F1 (vs windowed GT): {word_f1:.4f}")
    print(f"   Word-level F1 (vs original GT): {original_f1:.4f} ⭐ (Most realistic)")
    if 'iou_results' in locals() and iou_results:
        print(f"   IoU-based F1 (threshold=0.5): {iou_results['iou_f1']:.4f} ⭐ (Temporal overlap)")
        print(f"   Mean IoU score: {iou_results['mean_iou']:.4f}")
    
    print(f"\n📊 KEY INSIGHTS:")
    print(f"   • Binary detection handles profane vs clean: {balanced_accuracy:.1%}")
    print(f"   • Profanity classification handles which word: {weighted_profanity_class_accuracy:.1%}")
    print(f"   • Word-level performance in real scenarios: {original_f1:.1%}")
    if 'iou_results' in locals() and iou_results:
        print(f"   • IoU-based word localization accuracy: {iou_results['iou_f1']:.1%}")
    print(f"   • Data imbalance: {total_profanity/len(all_results):.1%} profane vs {total_none/len(all_results):.1%} clean")
    print(f"   • Current threshold: {threshold} (adjust with --threshold parameter)")
    
    # Add threshold guidance
    print(f"\n🎯 THRESHOLD GUIDANCE:")
    print(f"   • Current threshold: {threshold}")
    if binary_metrics:
        print(f"   • Current F1-Score: {binary_metrics['f1']:.3f}")
        print(f"   • Current Precision: {binary_metrics['precision']:.3f}")
        print(f"   • Current Recall: {binary_metrics['recall']:.3f}")
    print(f"   • Lower threshold (e.g., 0.3): Higher recall, more false positives")
    print(f"   • Higher threshold (e.g., 0.7): Higher precision, more false negatives")
    print(f"   • Balanced threshold (0.5): Good starting point")
    print(f"   • Use ROC curve to find optimal threshold for your use case")
    
    # Detailed word/instance statistics as requested
    print(f"\n=== Detailed Word/Instance Statistics ===")
    total_label_instances = len(all_results[all_results['true_label'] != 'none'])
    total_real_words_after_merge = total_true_words
    
    print(f"Total Label Instances (segments): {total_label_instances}")
    print(f"Total Real Words (after merge overlap): {total_real_words_after_merge}")
    print(f"Reduction ratio: {total_label_instances / total_real_words_after_merge:.2f}x" if total_real_words_after_merge > 0 else "N/A")
    
    # Breakdown by class
    print("\n=== Breakdown by Class ===")
    print("Format: [Original Ground Truth] → [Windowed Ground Truth] → [Predicted Instances] → [Merged Words]")
    
    # Load original eval.csv to get true ground truth counts
    try:
        original_df = pd.read_csv('./csv/eval_5labels.csv')
        print("(Showing original ground truth from eval.csv)")
    except:
        original_df = None
        print("(Could not load original eval.csv)")
    
    # Load windowed data to get windowed ground truth (use the same file being evaluated)
    try:
        windowed_df = pd.read_csv(csv_file)
        csv_filename = os.path.basename(csv_file)
        print(f"(Showing windowed ground truth from {csv_filename})")
    except:
        windowed_df = None
        print(f"(Could not load windowed eval data from {csv_file})")
    
    for label in profanity_labels:
        # Original ground truth count
        if original_df is not None:
            original_count = len(original_df[original_df['label'] == label])
        else:
            original_count = "N/A"
        
        # Windowed ground truth count
        if windowed_df is not None:
            windowed_ground_truth = len(windowed_df[windowed_df['label'] == label])
        else:
            windowed_ground_truth = "N/A"
            
        # Predicted instances count (from evaluation results)
        predicted_instances = len(all_results[all_results['predicted_label'] == label])
        
        # Merged word count (true words from ground truth)
        class_words = len(true_words[true_words['true_label'] == label]) if len(true_words) > 0 else 0
        
        print(f"{label}: [{original_count}] → [{windowed_ground_truth}] → {predicted_instances} predicted → {class_words} words")
    
    # Summary comparison
    if original_df is not None:
        total_original = len(original_df[original_df['label'] != 'none'])
        total_windowed_gt = len(windowed_df[windowed_df['label'] != 'none']) if windowed_df is not None else "N/A"
        total_predicted = len(all_results[all_results['predicted_label'] != 'none'])
        
        print(f"\n📊 SUMMARY COMPARISON:")
        print(f"Original Ground Truth: {total_original} profanity instances")
        if windowed_df is not None:
            print(f"Windowed Ground Truth: {total_windowed_gt} profanity instances")
            print(f"Predicted Instances: {total_predicted} profanity instances")
            print(f"Merged Words: {total_real_words_after_merge} profanity words")
            print(f"Original → Windowed GT: {total_windowed_gt/total_original:.2f}x increase")
            print(f"Windowed GT → Predicted: {total_predicted/total_windowed_gt:.2f}x ratio" if total_windowed_gt > 0 else "N/A")
            print(f"Original → Merged: {total_real_words_after_merge/total_original:.2f}x ratio")
        else:
            print(f"Predicted Instances: {total_predicted} profanity instances")
            print(f"Merged Words: {total_real_words_after_merge} profanity words")
            print(f"Original → Predicted: {total_predicted/total_original:.2f}x ratio")
            print(f"Original → Merged: {total_real_words_after_merge/total_original:.2f}x ratio")
    
    # Predicted vs actual breakdown
    print("\n=== Prediction Statistics ===")
    total_pred_instances = len(all_results[all_results['predicted_label'] != 'none'])
    total_pred_words_merged = total_pred_words
    print(f"Total Predicted Instances (segments): {total_pred_instances}")
    print(f"Total Predicted Words (after merge): {total_pred_words_merged}")
    print(f"Prediction reduction ratio: {total_pred_instances / total_pred_words_merged:.2f}x" if total_pred_words_merged > 0 else "N/A")
    
    # Performance comparison note
    print(f"\n=== Advanced Preprocessing Benefits ===")
    print("✅ Enhanced noise reduction using spectral subtraction")
    print("✅ RMS normalization for consistent signal levels")
    print("✅ Dynamic range compression for robust feature extraction")
    print("✅ High-pass filtering for low-frequency noise removal")
    print("✅ Consistent preprocessing pipeline matching training")
    
    # Ensure plots directory exists
    os.makedirs('./plots', exist_ok=True)
    
    # Create WORD-LEVEL confusion matrix (merged predictions vs windowed ground truth)
    print(f"\n=== Generating Word-Level Confusion Matrix ===")
    
    if len(word_predictions) > 0 and len(true_words) > 0:
        # Create word-level predictions and ground truth for confusion matrix
        word_level_true = []
        word_level_pred = []
        
        # For each predicted word, find matching ground truth
        for _, pred_word in word_predictions.iterrows():
            # Find overlapping true words
            overlapping_true = true_words[
                (true_words['file_path'] == pred_word['file_path']) &
                (true_words['start_time'] < pred_word['end_time']) &
                (true_words['end_time'] > pred_word['start_time'])
            ]
            
            if len(overlapping_true) > 0:
                # Use the first overlapping true label (should be the closest match)
                true_label = overlapping_true.iloc[0]['true_label']
                word_level_true.append(true_label)
                word_level_pred.append(pred_word['predicted_label'])
        
        # For ground truth words that weren't detected, add as missed
        # For ground truth words that weren't detected, we'll skip them in confusion matrix
        # (since we only want to show confusion among actual predictions)
        
        # Create confusion matrix for word-level predictions (only detected words)
        if len(word_level_true) > 0:
            print(f"Creating word-level confusion matrix with {len(word_level_true)} word comparisons")
            print(f"Note: Only showing detected words, not missed detections")
            
            # Plot word-level confusion matrix
            plot_confusion_matrix(
                word_level_true,
                word_level_pred,
                profanity_labels
            )
            print("✅ Word-level confusion matrix saved to ./plots/confusion_matrix_advanced.png")
        else:
            print("⚠️ No word-level comparisons found for confusion matrix")
    else:
        print("⚠️ No word predictions or true words found for confusion matrix")
    
    # Also create window-level confusion matrix for comparison
    if len(profanity_results_filtered) > 0:
        print(f"\n=== Generating Window-Level Confusion Matrix (for comparison) ===")
        
        # Modify plot function to save with different name
        def plot_window_confusion_matrix(true_labels, pred_labels, labels):
            setup_thai_font()
            
            try:
                cm = confusion_matrix(true_labels, pred_labels, labels=labels)
                plt.figure(figsize=(12, 10))
                
                sns.heatmap(cm, annot=True, fmt='d', 
                            xticklabels=labels, 
                            yticklabels=labels,
                            cmap='YlOrRd')
                
                plt.title('Window-Level Confusion Matrix (Advanced Preprocessing)', fontsize=16, pad=20)
                plt.ylabel('True Label', fontsize=12)
                plt.xlabel('Predicted Label', fontsize=12)
                
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=45)
                plt.tight_layout()
                
                # Save window-level confusion matrix to organized output
                window_cm_path = os.path.join(output_dirs['confusion_matrices'], 'confusion_matrix_window_level.png')
                plt.savefig(window_cm_path, 
                            bbox_inches='tight', 
                            dpi=300)
                plt.close()
                log_to_report(f"✅ Window-level confusion matrix saved to: {window_cm_path}")
                
            except Exception as e:
                print(f"Error creating window-level confusion matrix: {e}")
        
        plot_window_confusion_matrix(
            profanity_results_filtered['true_label'].values,
            profanity_results_filtered['predicted_label'].values,
            profanity_labels
        )
        
        # Generate comprehensive evaluation summary report
        generate_evaluation_summary(output_dirs['reports'], output_dirs)
        
        print(f"\n{'='*80}")
        print(f"🎉 COMPREHENSIVE EVALUATION COMPLETE!")
        print(f"{'='*80}")
        print(f"📂 All evaluation results organized in: {output_dirs['base']}")
        print(f"📊 Plots and visualizations: {output_dirs['plots']}")
        print(f"📋 Reports and summaries: {output_dirs['reports']}")
        print(f"💾 Data and analysis files: {output_dirs['data']}")
        print(f"📈 Confusion matrices: {output_dirs['confusion_matrices']}")
        print(f"📉 ROC/DET curves: {output_dirs['roc_curves']}, {output_dirs['det_curves']}")
        print(f"🔍 IoU analysis: {output_dirs['analysis']}")
        print(f"{'='*80}")

def generate_evaluation_summary(report_dir, output_dirs):
    """Generate a comprehensive evaluation summary report"""
    summary_path = os.path.join(report_dir, 'evaluation_summary.md')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Thai Profanity Detection Model - Comprehensive Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Evaluation Overview\n\n")
        f.write("This report contains comprehensive evaluation results for the Thai profanity detection model.\n\n")
        
        f.write("## Directory Structure\n\n")
        f.write("```\n")
        f.write(f"📂 {os.path.basename(output_dirs['base'])}/\n")
        f.write("├── 📊 plots/\n")
        f.write("│   ├── confusion_matrices/    # Confusion matrix visualizations\n")
        f.write("│   ├── roc_curves/           # ROC curve analysis\n")
        f.write("│   └── det_curves/           # DET curve analysis\n")
        f.write("├── 📋 reports/\n")
        f.write("│   ├── classification_report.txt    # Detailed classification metrics\n")
        f.write("│   └── evaluation_summary.md       # This summary report\n")
        f.write("├── 💾 data/\n")
        f.write("│   ├── evaluation_results.csv      # Raw evaluation results\n")
        f.write("│   ├── iou_analysis_results.csv    # IoU analysis data\n")
        f.write("│   └── iou_vs_original_analysis.csv # IoU comparison data\n")
        f.write("└── 🔍 analysis/\n")
        f.write("    └── (IoU analysis files)\n")
        f.write("```\n\n")
        
        f.write("## Key Files\n\n")
        f.write("- **Classification Report**: Detailed precision, recall, F1-score for each class\n")
        f.write("- **Confusion Matrices**: Visual representation of classification performance\n")
        f.write("- **ROC/DET Curves**: Binary classification performance analysis\n")
        f.write("- **IoU Analysis**: Temporal localization accuracy evaluation\n")
        f.write("- **Evaluation Results**: Complete raw data for further analysis\n\n")
        
        f.write("## Usage\n\n")
        f.write("1. Review the classification report for detailed metrics\n")
        f.write("2. Examine confusion matrices for class-wise performance patterns\n")
        f.write("3. Analyze ROC/DET curves for binary detection performance\n")
        f.write("4. Check IoU analysis for temporal localization accuracy\n")
        f.write("5. Use raw data files for custom analysis and visualization\n\n")
        
        f.write("---\n")
        f.write("*Generated by Thai Profanity Detection Evaluation System*\n")
    
    log_to_report(f"✅ Comprehensive evaluation summary saved to: {summary_path}")

if __name__ == "__main__":
    main()

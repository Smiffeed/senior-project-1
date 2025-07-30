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

def evaluate_window(model, feature_extractor, file_path, start_time, end_time):
    try:
        # Process audio with advanced preprocessing matching training
        audio = advanced_preprocess_audio(file_path, start_time, end_time)
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
        
        # Save plot
        plt.savefig('./plots/confusion_matrix_advanced.png', 
                    bbox_inches='tight', 
                    dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        print("Skipping confusion matrix generation due to insufficient data variety")

def main():
    # Setup Thai font
    setup_thai_font()
    
    # Load the model (support for advanced models)
    model_path = './models/cw_ham_v2'  # Update this path to your best model

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
    df = pd.read_csv('./csv/eval_windowed_0.6s.csv')  # Using the 0.3s version for better coverage
    
    print(f"Loaded {len(df)} windowed samples for evaluation")
    
    # Create results DataFrame
    results = []
    
    # Process each window
    total_windows = len(df)
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
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{total_windows} windows ({(idx + 1)/total_windows*100:.1f}%)")
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv('./csv/eval_results_advanced.csv', index=False)
    
    print(f"Saved detailed results to ./csv/eval_results_advanced.csv")
    
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
    
    # For classification report and confusion matrix, replace 'none' predictions 
    # with a special label 'missed_profanity' to include in metrics
    profanity_results.loc[profanity_results['predicted_label'] == 'none', 'predicted_label'] = 'missed_profanity'
    
    # Generate classification report with modified labels
    if len(profanity_results) > 0:
        report = classification_report(
            profanity_results['true_label'],
            profanity_results['predicted_label'],
            labels=profanity_labels + ['missed_profanity'],
            digits=4,
            zero_division=0
        )
        
        # Save and print classification report
        os.makedirs('./evaluation_results', exist_ok=True)
        with open('./evaluation_results/classification_report_advanced.txt', 'w', encoding='utf-8') as f:
            f.write("Classification Report (Advanced Preprocessing):\n")
            f.write(report)
        
        print("\n=== Classification Report (Advanced Preprocessing) ===")
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
    
    # For each row, determine if it's a true profanity and if the prediction was c7rrect
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
    
    # ADD: Explicit Binary Classification Accuracy (weighted for imbalance)
    binary_accuracy = (tp + tn) / len(all_results)
    print(f"\n🎯 BINARY CLASSIFICATION ACCURACY:")
    print(f"Simple Binary Accuracy: {binary_accuracy:.4f}")
    print(f"Balanced Binary Accuracy: {balanced_accuracy:.4f}")
    print(f"(Balanced accounts for class imbalance: {total_profanity/len(all_results):.1%} profane vs {total_none/len(all_results):.1%} clean)")
    
    print("\nClass Distribution:")
    print(f"Profanity samples: {total_profanity} ({total_profanity/len(all_results):.2%})")
    print(f"None samples: {total_none} ({total_none/len(all_results):.2%})")
    
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
    
    # ADD NEW: Word-level accuracy against ORIGINAL ground truth
    print(f"\n=== Word-Level vs Original Ground Truth Accuracy ===")
    
    # Load original ground truth words
    try:
        original_df = pd.read_csv('./csv/eval.csv')
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
        
        print(f"\n🎯 ACCURACY COMPARISON:")
        print(f"Windowed-based Word F1: {word_f1:.4f}")
        print(f"Original GT-based Word F1: {original_f1:.4f}")
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
    
    print(f"\n📊 KEY INSIGHTS:")
    print(f"   • Binary detection handles profane vs clean: {balanced_accuracy:.1%}")
    print(f"   • Profanity classification handles which word: {weighted_profanity_class_accuracy:.1%}")
    print(f"   • Word-level performance in real scenarios: {original_f1:.1%}")
    print(f"   • Data imbalance: {total_profanity/len(all_results):.1%} profane vs {total_none/len(all_results):.1%} clean")
    
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
        original_df = pd.read_csv('./csv/eval.csv')
        print("(Showing original ground truth from eval.csv)")
    except:
        original_df = None
        print("(Could not load original eval.csv)")
    
    # Load windowed data to get windowed ground truth
    try:
        windowed_df = pd.read_csv('./csv/eval_windowed_0.6s.csv')
        print("(Showing windowed ground truth from eval_windowed_0.6s.csv)")
    except:
        windowed_df = None
        print("(Could not load windowed eval data)")
    
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
        for _, true_word in true_words.iterrows():
            # Check if this true word was detected by any prediction
            overlapping_pred = word_predictions[
                (word_predictions['file_path'] == true_word['file_path']) &
                (word_predictions['start_time'] < true_word['end_time']) &
                (word_predictions['end_time'] > true_word['start_time'])
            ]
            
            if len(overlapping_pred) == 0:
                # This true word was not detected
                word_level_true.append(true_word['true_label'])
                word_level_pred.append('missed_profanity')
        
        # Create confusion matrix for word-level predictions
        if len(word_level_true) > 0:
            print(f"Creating word-level confusion matrix with {len(word_level_true)} word comparisons")
            
            # Plot word-level confusion matrix
            plot_confusion_matrix(
                word_level_true,
                word_level_pred,
                profanity_labels + ['missed_profanity']
            )
            print("✅ Word-level confusion matrix saved to ./plots/confusion_matrix_advanced.png")
        else:
            print("⚠️ No word-level comparisons found for confusion matrix")
    else:
        print("⚠️ No word predictions or true words found for confusion matrix")
    
    # Also create window-level confusion matrix for comparison
    if len(profanity_results) > 0:
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
                
                # Save with different filename
                plt.savefig('./plots/confusion_matrix_window_level.png', 
                            bbox_inches='tight', 
                            dpi=300)
                plt.close()
                
            except Exception as e:
                print(f"Error creating window-level confusion matrix: {e}")
        
        plot_window_confusion_matrix(
            profanity_results['true_label'].values,
            profanity_results['predicted_label'].values,
            profanity_labels + ['missed_profanity']
        )
        print("✅ Window-level confusion matrix saved to ./plots/confusion_matrix_window_level.png")

if __name__ == "__main__":
    main()

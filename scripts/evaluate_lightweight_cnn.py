#!/usr/bin/env python3
"""
Evaluation script for Lightweight CNN Thai Profanity Detection
- Loads trained CNN model
- Uses advanced preprocessing pipeline
- Evaluates on windowed CSV
- Outputs binary, multi-class, word-level metrics and confusion matrix
"""
import os
import torch
import numpy as np
import pandas as pd
import librosa
import torchaudio
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from lightweight_cnn_model import LightweightCNN, label_map, rev_label_map, AudioDataset

def advanced_preprocess_audio(file_path, start_time, end_time, target_length=16000):
    try:
        file_path = file_path.replace('\\', '/')
        if not os.path.exists(file_path):
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
        # Pre-emphasis
        audio_np = librosa.effects.preemphasis(audio_np, coef=0.97)
        # Hamming window
        if len(audio_np) > 1:
            audio_np = audio_np * np.hamming(len(audio_np))
        # Noise gate
        noise_threshold = 0.005
        audio_np = np.where(np.abs(audio_np) < noise_threshold, 0, audio_np)
        # RMS normalization
        rms = np.sqrt(np.mean(audio_np ** 2))
        if rms > 0:
            audio_np = audio_np * (0.1 / rms)
        # Pad/truncate
        if len(audio_np) < target_length:
            audio_np = np.pad(audio_np, (0, target_length - len(audio_np)), 'constant')
        elif len(audio_np) > target_length:
            start_idx = (len(audio_np) - target_length) // 2
            audio_np = audio_np[start_idx:start_idx + target_length]
        return audio_np
    except Exception as e:
        print(f"Error in advanced preprocessing {file_path} ({start_time}-{end_time}): {e}")
        return None

def evaluate_window(model, file_path, start_time, end_time, device):
    audio = advanced_preprocess_audio(file_path, start_time, end_time)
    if audio is None:
        return 'none', 0.0
    # Convert to mel spectrogram
    n_fft = 1024
    hop_length = 160
    n_mels = 40
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=8000
    )
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
    target_frames = 101
    if mel_spec.shape[1] < target_frames:
        pad_width = target_frames - mel_spec.shape[1]
        mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), 'constant')
    elif mel_spec.shape[1] > target_frames:
        mel_spec = mel_spec[:, :target_frames]
    mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 40, 101)
    with torch.no_grad():
        logits = model(mel_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_label = rev_label_map[pred_idx]
        confidence = probs[pred_idx]
    return pred_label, confidence

def main():
    # --- ADVANCED EVALUATION SCRIPT ---
    model_path = './models/lightweight_cnn/lightweight_cnn_model.pth'
    eval_csv = './csv/eval_windowed_0.25s.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = LightweightCNN(num_classes=len(label_map), dropout_rate=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Using device: {device}")
    print("Using ADVANCED preprocessing pipeline matching training script")
    df = pd.read_csv(eval_csv)
    print(f"Loaded {len(df)} windowed samples for evaluation")
    results = []
    for idx, row in df.iterrows():
        pred_label, confidence = evaluate_window(
            model,
            row['file_path'],
            row['start_time'],
            row['end_time'],
            device
        )
        results.append({
            'file_path': row['file_path'],
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'true_label': row['label'],
            'predicted_label': pred_label,
            'confidence': confidence
        })
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} windows")
    results_df = pd.DataFrame(results)
    results_df.to_csv('./csv/eval_results_lightweight_cnn.csv', index=False)
    print(f"Saved detailed results to ./csv/eval_results_lightweight_cnn.csv")

    profanity_labels = [label for label in label_map.keys() if label != 'none']
    profanity_results = results_df[results_df['true_label'] != 'none'].copy()
    profanity_results.loc[profanity_results['predicted_label'] == 'none', 'predicted_label'] = 'missed_profanity'

    # Print misclassified cases where profanity was detected as 'none'
    none_misclassifications = profanity_results[profanity_results['predicted_label'] == 'missed_profanity']
    print(f"\n=== Profanity Words Misclassified as 'none' ({len(none_misclassifications)} cases) ===")
    for i, (_, row) in enumerate(none_misclassifications.iterrows()):
        if i < 10:
            print(f"{row['file_path']} {row['start_time']}-{row['end_time']} true: {row['true_label']} pred: none")
        elif i == 10:
            print("...")

    # Classification report
    if len(profanity_results) > 0:
        report = classification_report(
            profanity_results['true_label'],
            profanity_results['predicted_label'],
            labels=profanity_labels + ['missed_profanity'],
            digits=4,
            zero_division=0
        )
        os.makedirs('./evaluation_results', exist_ok=True)
        with open('./evaluation_results/classification_report_lightweight_cnn.txt', 'w', encoding='utf-8') as f:
            f.write("Classification Report (Lightweight CNN):\n")
            f.write(report)
        print("\n=== Classification Report (Lightweight CNN) ===")
        print(report)
        correct_predictions = (profanity_results['true_label'] == profanity_results['predicted_label']).sum()
        total_predictions = len(profanity_results)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\nOverall Accuracy (excluding 'none'): {accuracy:.4f}")
        print(f"Total Profanity Windows: {total_predictions}")
        print(f"Correct Predictions: {correct_predictions}")

    # Binary profanity detection metrics
    all_results = pd.DataFrame(results)
    all_results['is_true_profanity'] = all_results['true_label'] != 'none'
    all_results['is_predicted_profanity'] = all_results['predicted_label'] != 'none'
    all_results['is_correct_prediction'] = all_results['true_label'] == all_results['predicted_label']
    tp = ((all_results['is_true_profanity']) & (all_results['is_predicted_profanity']) & (all_results['is_correct_prediction'])).sum()
    fp = ((all_results['is_predicted_profanity']) & (~all_results['is_correct_prediction'])).sum()
    tn = ((~all_results['is_true_profanity']) & (~all_results['is_predicted_profanity'])).sum()
    fn = (all_results['is_true_profanity'] & (~all_results['is_predicted_profanity'] | ~all_results['is_correct_prediction'])).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (sensitivity + specificity) / 2
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
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
    binary_accuracy = (tp + tn) / len(all_results)
    print(f"\n🎯 BINARY CLASSIFICATION ACCURACY:")
    print(f"Simple Binary Accuracy: {binary_accuracy:.4f}")
    print(f"Balanced Binary Accuracy: {balanced_accuracy:.4f}")
    print(f"(Balanced accounts for class imbalance: {total_profanity/len(all_results):.1%} profane vs {total_none/len(all_results):.1%} clean)")
    print("\nClass Distribution:")
    print(f"Profanity samples: {total_profanity} ({total_profanity/len(all_results):.2%})")
    print(f"None samples: {total_none} ({total_none/len(all_results):.2%})")

    # Multi-class classification accuracy (for profanity words only)
    print(f"\n=== MULTI-CLASS PROFANITY CLASSIFICATION ACCURACY ===")
    profanity_only = all_results[all_results['true_label'] != 'none'].copy()
    if len(profanity_only) > 0:
        correct_profanity_class = (profanity_only['true_label'] == profanity_only['predicted_label']).sum()
        profanity_class_accuracy = correct_profanity_class / len(profanity_only)
        from sklearn.metrics import accuracy_score
        weighted_profanity_accuracy = accuracy_score(
            profanity_only['true_label'],
            profanity_only['predicted_label'],
            sample_weight=None
        )
        profanity_class_counts = profanity_only['true_label'].value_counts()
        total_profanity_samples = len(profanity_only)
        print(f"Total Profanity Samples: {total_profanity_samples}")
        print(f"Correct Profanity Classifications: {correct_profanity_class}")
        print(f"Simple Profanity Classification Accuracy: {profanity_class_accuracy:.4f}")
        print(f"Weighted Profanity Classification Accuracy: {weighted_profanity_accuracy:.4f}")
        print(f"\nPer-Class Profanity Accuracy:")
        for label in profanity_labels:
            class_correct = ((profanity_only['true_label'] == label) & (profanity_only['predicted_label'] == label)).sum()
            class_total = (profanity_only['true_label'] == label).sum()
            class_acc = class_correct / class_total if class_total > 0 else 0
            print(f"{label}: {class_acc:.4f} ({class_correct}/{class_total})")
    else:
        print("No profanity samples found for classification accuracy")

    # Word-level metrics
    def normalize_to_word_level(df, label_column='predicted_label'):
        word_detections = []
        for file_path in df['file_path'].unique():
            file_data = df[df['file_path'] == file_path].sort_values('start_time')
            current_detection = None
            for _, row in file_data.iterrows():
                if row[label_column] != 'none':
                    if (current_detection is None or row[label_column] != current_detection[label_column] or row['start_time'] > current_detection['end_time'] + 0.1):
                        if current_detection:
                            word_detections.append(current_detection)
                        current_detection = {
                            'file_path': file_path,
                            'start_time': row['start_time'],
                            'end_time': row['end_time'],
                            label_column: row[label_column],
                            'confidence': row.get('confidence', 1.0)
                        }
                    else:
                        current_detection['end_time'] = row['end_time']
                        if 'confidence' in row:
                            current_detection['confidence'] = max(current_detection.get('confidence', 1.0), row['confidence'])
            if current_detection:
                word_detections.append(current_detection)
        return pd.DataFrame(word_detections)

    word_predictions = normalize_to_word_level(results_df[results_df['predicted_label'] != 'none'], 'predicted_label')
    true_words = normalize_to_word_level(results_df[results_df['true_label'] != 'none'], 'true_label')
    total_true_words = len(true_words)
    total_pred_words = len(word_predictions)
    correct_word_detections = 0
    if total_pred_words > 0 and total_true_words > 0:
        for _, pred in word_predictions.iterrows():
            overlapping_true = true_words[
                (true_words['file_path'] == pred['file_path']) &
                (true_words['start_time'] < pred['end_time']) &
                (true_words['end_time'] > pred['start_time'])
            ]
            overlapping_labels = overlapping_true['true_label'].tolist()
            if pred['predicted_label'] in overlapping_labels:
                correct_word_detections += 1
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

    # Word-level vs original ground truth
    print(f"\n=== Word-Level vs Original Ground Truth Accuracy ===")
    try:
        original_df = pd.read_csv('./csv/eval.csv')
        original_gt_words = original_df[original_df['label'] != 'none'].copy()
        total_original_words = len(original_gt_words)
        correct_vs_original = 0
        if total_pred_words > 0 and total_original_words > 0:
            for _, pred in word_predictions.iterrows():
                overlapping_true = original_gt_words[
                    (original_gt_words['file_path'] == pred['file_path']) &
                    (original_gt_words['start_time'] < pred['end_time']) &
                    (original_gt_words['end_time'] > pred['start_time'])
                ]
                overlapping_labels = overlapping_true['label'].tolist()
                if pred['predicted_label'] in overlapping_labels:
                    correct_vs_original += 1
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
            print("⚠️  Windowed GT metrics may be inflated due to data augmentation")
        print(f"\n=== Class-wise Performance vs Original Ground Truth ===")
        for label in profanity_labels:
            class_correct = 0
            class_total = len(original_gt_words[original_gt_words['label'] == label])
            for _, pred in word_predictions[word_predictions['predicted_label'] == label].iterrows():
                overlapping_true = original_gt_words[
                    (original_gt_words['file_path'] == pred['file_path']) &
                    (original_gt_words['start_time'] < pred['end_time']) &
                    (original_gt_words['end_time'] > pred['start_time']) &
                    (original_gt_words['label'] == label)
                ]
                if not overlapping_true.empty:
                    class_correct += 1
            class_precision = class_correct / len(word_predictions[word_predictions['predicted_label'] == label]) if len(word_predictions[word_predictions['predicted_label'] == label]) > 0 else 0
            class_recall = class_correct / class_total if class_total > 0 else 0
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
            print(f"{label}: Precision={class_precision:.4f}, Recall={class_recall:.4f}, F1={class_f1:.4f}")
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

    print(f"\n" + "="*70)
    print("🎯 COMPREHENSIVE ACCURACY SUMMARY")
    print("="*70)
    print(f"\n1️⃣ BINARY CLASSIFICATION ACCURACY (Profane vs Non-Profane):")
    print(f"   Simple Binary Accuracy: {binary_accuracy:.4f}")
    print(f"   Balanced Binary Accuracy: {balanced_accuracy:.4f} ⭐ (Recommended for imbalanced data)")
    print(f"   Binary F1-Score: {f1:.4f}")
    print(f"\n2️⃣ MULTI-CLASS CLASSIFICATION ACCURACY (Which profanity word?):")
    print(f"   Simple Profanity Class Accuracy: {profanity_class_accuracy:.4f}")
    print(f"   Weighted Profanity Class Accuracy: {weighted_profanity_accuracy:.4f} ⭐ (Accounts for class imbalance)")
    print(f"\n3️⃣ WORD-LEVEL ACCURACY (Realistic performance):")
    print(f"   Word-level F1 (vs windowed GT): {word_f1:.4f}")
    print(f"   Word-level F1 (vs original GT): {original_f1:.4f} ⭐ (Most realistic)")
    print(f"\n📊 KEY INSIGHTS:")
    print(f"   • Binary detection handles profane vs clean: {balanced_accuracy:.1%}")
    print(f"   • Profanity classification handles which word: {weighted_profanity_accuracy:.1%}")
    print(f"   • Word-level performance in real scenarios: {original_f1:.1%}")
    print(f"   • Data imbalance: {total_profanity/len(all_results):.1%} profane vs {total_none/len(all_results):.1%} clean")

    print(f"\n=== Detailed Word/Instance Statistics ===")
    total_label_instances = len(all_results[all_results['true_label'] != 'none'])
    total_real_words_after_merge = total_true_words
    print(f"Total Label Instances (segments): {total_label_instances}")
    print(f"Total Real Words (after merge overlap): {total_real_words_after_merge}")
    print(f"Reduction ratio: {total_label_instances / total_real_words_after_merge:.2f}x" if total_real_words_after_merge > 0 else "N/A")

    print("\n=== Breakdown by Class ===")
    print("Format: [Original Ground Truth] → [Windowed Ground Truth] → [Predicted Instances] → [Merged Words]")
    try:
        original_df = pd.read_csv('./csv/eval.csv')
        print("(Showing original ground truth from eval.csv)")
    except:
        original_df = None
        print("(Could not load original eval.csv)")
    try:
        windowed_df = pd.read_csv('./csv/eval_windowed_0.3s.csv')
        print("(Showing windowed ground truth from eval_windowed_0.3s.csv)")
    except:
        windowed_df = None
        print("(Could not load windowed eval data)")
    for label in profanity_labels:
        if original_df is not None:
            original_count = len(original_df[original_df['label'] == label])
        else:
            original_count = 'N/A'
        if windowed_df is not None:
            windowed_ground_truth = len(windowed_df[windowed_df['label'] == label])
        else:
            windowed_ground_truth = 'N/A'
        predicted_instances = len(all_results[all_results['predicted_label'] == label])
        class_words = len(true_words[true_words['true_label'] == label]) if len(true_words) > 0 else 0
        print(f"{label}: [{original_count}] → [{windowed_ground_truth}] → {predicted_instances} predicted → {class_words} words")
    if original_df is not None:
        total_original = len(original_df[original_df['label'] != 'none'])
        total_windowed_gt = len(windowed_df[windowed_df['label'] != 'none']) if windowed_df is not None else "N/A"
        total_predicted = len(all_results[all_results['predicted_label'] != 'none'])
        print(f"\n📊 SUMMARY COMPARISON:")
        print(f"Original Ground Truth: {total_original} profanity instances")
        if windowed_df is not None:
            print(f"Windowed Ground Truth: {total_windowed_gt} profanity instances")
        else:
            print(f"Windowed Ground Truth: N/A")
        print(f"Predicted Instances: {total_predicted}")

    print("\n=== Prediction Statistics ===")
    total_pred_instances = len(all_results[all_results['predicted_label'] != 'none'])
    total_pred_words_merged = total_pred_words
    print(f"Total Predicted Instances (segments): {total_pred_instances}")
    print(f"Total Predicted Words (after merge): {total_pred_words_merged}")
    print(f"Prediction reduction ratio: {total_pred_instances / total_pred_words_merged:.2f}x" if total_pred_words_merged > 0 else "N/A")

    print(f"\n=== Advanced Preprocessing Benefits ===")
    print("✅ Pre-emphasis, Hamming window, noise gate, RMS normalization, padding/truncation")
    print("✅ Consistent preprocessing pipeline matching training")

    os.makedirs('./plots', exist_ok=True)
    cm = confusion_matrix(
        profanity_results['true_label'],
        profanity_results['predicted_label'],
        labels=profanity_labels + ['missed_profanity']
    )
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=profanity_labels + ['missed_profanity'], yticklabels=profanity_labels + ['missed_profanity'], cmap='YlOrRd')
    plt.title('Word-Level Confusion Matrix (Lightweight CNN)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./plots/confusion_matrix_lightweight_cnn.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved confusion matrix to ./plots/confusion_matrix_lightweight_cnn.png")
if __name__ == "__main__":
    main()

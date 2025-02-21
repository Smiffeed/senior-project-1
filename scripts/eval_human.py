import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import os
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# Label mappings
label_map = {
    'none': 0,
    'เย็ดแม่': 1,
    'กู': 2,
    'มึง': 3,
    'เหี้ย': 4,
    'ควย': 5,
    'สวะ': 6,
    'หี': 7,
    'แตด': 8
}

rev_label_map = {v: k for k, v in label_map.items()}

def setup_thai_font():
    """Setup Thai font for matplotlib"""
    try:
        font_path = './fonts/THSarabunNew.ttf'  # Adjust path as needed
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'Cordia New'
    except:
        print("Warning: Thai font not found. Using default font.")

def preprocess_audio(audio_path, target_sr=16000, segment_length=16000):
    """
    Preprocess audio file into segments
    Returns: list of segments, audio duration
    """
    waveform, sr = torchaudio.load(audio_path)
    
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # Convert stereo to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Normalize audio
    waveform = waveform / torch.max(torch.abs(waveform))
    
    # Split into segments
    segments = []
    hop_length = int(target_sr * 0.5)  # 0.5 second hop
    
    for start in range(0, waveform.shape[1], hop_length):
        end = start + segment_length
        if end > waveform.shape[1]:
            # Pad last segment if needed
            segment = torch.zeros((1, segment_length))
            segment[0, :waveform.shape[1]-start] = waveform[0, start:]
        else:
            segment = waveform[:, start:end]
        segments.append(segment)
    
    duration = waveform.shape[1] / target_sr
    return segments, duration

def predict(model, feature_extractor, audio_segment):
    """
    Make prediction for a single audio segment
    Returns: predicted class index and probabilities
    """
    inputs = feature_extractor(
        audio_segment.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    return predicted_class, probabilities.squeeze().cpu().numpy()

def load_timestamp_labels(label_file):
    """Load timestamp ranges and labels from txt file"""
    labels = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                start_time = float(parts[0])
                end_time = float(parts[1])
                label = parts[2]
                labels.append({
                    'start': start_time,
                    'end': end_time,
                    'label': label
                })
    return labels

def evaluate_predictions(segment_predictions, segment_times, true_labels):
    """
    Evaluate predictions against true timestamp ranges
    Returns matched predictions and ground truth for accuracy calculation
    """
    pred_results = []
    true_results = []
    
    # For each segment prediction
    for pred, time in zip(segment_predictions, segment_times):
        # Find the corresponding true label for this timestamp
        true_label = 'none'
        for label_info in true_labels:
            if label_info['start'] <= time < label_info['end']:
                true_label = label_info['label']
                break
        
        pred_label = rev_label_map[pred]
        
        # Only include non-'none' cases in results
        if true_label != 'none' or pred_label != 'none':
            pred_results.append(pred_label)
            true_results.append(true_label)
    
    return pred_results, true_results

def evaluate_model(model_path, test_dir, output_dir):
    """Evaluate the model on test data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model.eval()
    
    all_predictions = []
    all_true_labels = []
    results = []
    
    print("\nStarting evaluation...")
    
    with torch.no_grad():
        audio_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
        
        for audio_file in audio_files:
            file_path = os.path.join(test_dir, audio_file)
            label_file = os.path.join(test_dir, 'labels', audio_file.replace('.wav', '.txt'))
            
            if not os.path.exists(label_file):
                print(f"Warning: No label file found for {audio_file}")
                continue
            
            try:
                # Load true labels
                true_labels = load_timestamp_labels(label_file)
                
                # Process audio
                segments, _ = preprocess_audio(file_path)
                segment_predictions = []
                segment_times = []
                
                # Generate predictions for each segment
                for i, segment in enumerate(segments):
                    predicted_class, probabilities = predict(model, feature_extractor, segment)
                    segment_time = (i * 0.5)  # 0.5s hop length
                    
                    # Only store non-none predictions with their timestamps
                    if predicted_class != 0:  # not 'none'
                        pred_label = rev_label_map[predicted_class]
                        confidence = probabilities[predicted_class]
                        print(f"File: {audio_file} | Time: {segment_time:.2f}s | Detected: {pred_label} | Confidence: {confidence:.3f}")
                    
                    segment_predictions.append(predicted_class)
                    segment_times.append(segment_time)
                
                # Evaluate predictions
                pred_results, true_results = evaluate_predictions(
                    segment_predictions,
                    segment_times,
                    true_labels
                )
                
                all_predictions.extend(pred_results)
                all_true_labels.extend(true_results)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
    
    # Calculate and save results
    if all_predictions:
        # Filter out 'none' labels from both lists
        filtered_predictions = []
        filtered_true_labels = []
        for pred, true in zip(all_predictions, all_true_labels):
            if true != 'none' or pred != 'none':
                filtered_predictions.append(pred)
                filtered_true_labels.append(true)
        
        report = classification_report(
            filtered_true_labels,
            filtered_predictions,
            labels=[label for label in label_map.keys() if label != 'none'],
            digits=4,
            zero_division=0
        )
        
        # Save results
        with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w', encoding='utf-8') as f:
            f.write("Classification Report:\n")
            f.write(report)
        
        print("\n=== Classification Report ===")
        print(report)
        
        # Calculate accuracy excluding 'none' cases
        accuracy = sum(p == t for p, t in zip(filtered_predictions, filtered_true_labels)) / len(filtered_predictions) if filtered_predictions else 0
        print(f"\nOverall Accuracy (excluding 'none'): {accuracy:.4f}")
    else:
        print("No predictions were made. Check if test files exist and are properly labeled.")

if __name__ == "__main__":
    # Setup Thai font
    setup_thai_font()
    
    # Paths
    model_path = "./models/humanv1"
    test_dir = "./eval"  # Directory containing wav files, with labels in ./eval/label
    output_dir = "./evaluation_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation
    evaluate_model(model_path, test_dir, output_dir) 
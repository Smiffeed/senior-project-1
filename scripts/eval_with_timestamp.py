import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define label mapping (same as training)
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

# Reverse label mapping for output
rev_label_map = {v: k for k, v in label_map.items()}

def setup_thai_font():
    """Setup font for Thai language support"""
    try:
        # Try different Thai fonts
        font_paths = [
            'C:/Windows/Fonts/THSarabunNew.ttf',  # TH Sarabun New
            'C:/Windows/Fonts/tahoma.ttf',        # Tahoma
            'C:/Windows/Fonts/arial.ttf',         # Arial
        ]
        
        font_found = False
        for font_path in font_paths:
            if os.path.exists(font_path):
                import matplotlib.font_manager as fm
                fm.fontManager.addfont(font_path)
                plt.rcParams['font.family'] = os.path.splitext(os.path.basename(font_path))[0]
                font_found = True
                print(f"Using font: {plt.rcParams['font.family']}")
                break
        
        if not font_found:
            print("Warning: No suitable Thai font found. Labels might not display correctly.")
            
    except Exception as e:
        print(f"Warning: Error setting up Thai font: {str(e)}")

def preprocess_audio_segment(file_path, start_time, end_time):
    """Load and preprocess a specific segment of audio"""
    # Load the full audio
    audio, sr = torchaudio.load(file_path)
    
    # Convert time to samples
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # Extract segment
    audio = audio[:, start_sample:end_sample]
    
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Normalize audio
    audio = (audio - audio.mean()) / (audio.std() + 1e-7)
    
    return audio.squeeze().numpy()

def predict(model, feature_extractor, audio_segment):
    inputs = feature_extractor(audio_segment, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(model.device)
    
    if 'attention_mask' not in inputs:
        attention_mask = torch.ones_like(input_values)
    else:
        attention_mask = inputs.attention_mask.to(model.device)

    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()
        
        return predicted_class, probabilities[0].tolist()

def evaluate_model(model_path, eval_csv_path, output_dir):
    """Evaluate the model using segments defined in CSV file"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and feature extractor
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model.eval()
    
    # Read CSV file
    df = pd.read_csv(eval_csv_path)
    
    all_predictions = []
    all_true_labels = []
    results = []
    
    # For non-none accuracy calculation
    non_none_predictions = []
    non_none_true_labels = []
    
    print("\nStarting evaluation...")
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            try:
                file_path = row['File Name'].strip()
                start_time = float(row['Start Time (s)'])
                end_time = float(row['End Time (s)'])
                true_label = row['Label']
                
                # Skip if label not in label_map
                if true_label not in label_map:
                    print(f"Warning: Unknown label '{true_label}' in row {idx+1}")
                    continue
                
                # Process audio segment
                audio_segment = preprocess_audio_segment(file_path, start_time, end_time)
                predicted_class, probabilities = predict(model, feature_extractor, audio_segment)
                
                predicted_label = rev_label_map[predicted_class]
                confidence = probabilities[predicted_class]
                
                # Print simple result
                print(f"Segment {idx+1:<4} | File: {os.path.basename(file_path):<30} | "
                      f"Time: [{start_time:.2f}-{end_time:.2f}] | "
                      f"True: {true_label:<10} | Predicted: {predicted_label:<10} | "
                      f"Confidence: {confidence:.3f}")
                
                # Store all predictions for full confusion matrix
                all_predictions.append(predicted_class)
                all_true_labels.append(label_map[true_label])
                
                # Store non-none predictions for accuracy calculation
                if true_label != 'none':
                    non_none_predictions.append(predicted_class)
                    non_none_true_labels.append(label_map[true_label])
                
                results.append({
                    'file': os.path.basename(file_path),
                    'segment': f"{start_time:.2f}-{end_time:.2f}",
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'correct': true_label == predicted_label
                })
                
            except Exception as e:
                print(f"Error processing row {idx+1}: {str(e)}")
    
    # Generate and save results
    if all_predictions:
        # Full classification report (including 'none')
        full_report = classification_report(
            all_true_labels,
            all_predictions,
            labels=list(label_map.values()),
            target_names=list(label_map.keys()),
            digits=4,
            zero_division=0
        )
        
        # Non-none classification report
        non_none_labels = {k: v for k, v in label_map.items() if k != 'none'}
        non_none_report = classification_report(
            non_none_true_labels,
            non_none_predictions,
            labels=list(non_none_labels.values()),
            target_names=list(non_none_labels.keys()),
            digits=4,
            zero_division=0
        )
        
        # Save results
        with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w', encoding='utf-8') as f:
            f.write("Full Classification Report (including 'none'):\n")
            f.write(full_report)
            f.write("\n\nClassification Report (excluding 'none'):\n")
            f.write(non_none_report)
            f.write("\n\nDetailed Results:\n")
            for result in results:
                f.write(f"\nFile: {result['file']}")
                f.write(f"\nSegment: {result['segment']}")
                f.write(f"\nTrue Label: {result['true_label']}")
                f.write(f"\nPredicted Label: {result['predicted_label']}")
                f.write(f"\nConfidence: {result['confidence']:.2f}")
                f.write(f"\nCorrect: {result['correct']}\n")
        
        # Plot confusion matrix
        cm = confusion_matrix(all_true_labels, all_predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_map.keys(),
                    yticklabels=label_map.keys())
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n=== Full Classification Report (including 'none') ===")
        print(full_report)
        
        print("\n=== Classification Report (excluding 'none') ===")
        print(non_none_report)
        
        # Calculate accuracies
        full_accuracy = (np.array(all_predictions) == np.array(all_true_labels)).mean()
        non_none_accuracy = (np.array(non_none_predictions) == np.array(non_none_true_labels)).mean()
        
        print(f"\nOverall Accuracy (including 'none'): {full_accuracy:.4f}")
        print(f"Accuracy (excluding 'none'): {non_none_accuracy:.4f}")
    else:
        print("No predictions were made. Check if CSV file is properly formatted and audio files exist.")

if __name__ == "__main__":
    # Paths - use absolute paths or correct relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    model_path = os.path.join(project_dir, "models", "clear_audio_train_fold_5")  # Update this to your model path
    eval_csv_path = os.path.join(project_dir, "csv", "eval.csv")
    output_dir = os.path.join(project_dir, "evaluation_results")
    
    # Verify paths exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not os.path.exists(eval_csv_path):
        raise FileNotFoundError(f"CSV file not found: {eval_csv_path}")
    
    print(f"Using model path: {model_path}")
    print(f"Using CSV path: {eval_csv_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation
    evaluate_model(model_path, eval_csv_path, output_dir)
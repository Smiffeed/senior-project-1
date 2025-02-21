import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set these environment variables for verbose output
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["DATASETS_VERBOSITY"] = "info"
os.environ["PYTHONPATH"] = "."

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

def preprocess_audio(file_path, segment_length=16000, hop_length=8000):
    """
    segment_length=16000 represents 1 second at 16kHz
    hop_length=8000 represents 0.5 second overlap
    """
    audio, sr = torchaudio.load(file_path)
    
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Normalize audio
    audio = (audio - audio.mean()) / (audio.std() + 1e-7)
    
    audio = audio.squeeze().numpy()
    
    # Split audio into overlapping segments
    segments = []
    for start in range(0, len(audio), hop_length):
        end = start + segment_length
        segment = audio[start:end]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))
        segments.append(segment)
    
    return segments, len(audio) / 16000

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

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot and save confusion matrix"""
    # Set Thai font
    plt.rcParams['font.family'] = 'Cordia New'  # or 'Tahoma' or 'Arial Unicode MS'
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_map.keys(),
                yticklabels=label_map.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    # Save with high DPI for better quality
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), 
                dpi=300, 
                bbox_inches='tight')
    plt.close()

def evaluate_model(model_path, test_file_path, output_dir, noise_reduce_factor=1.5):
    """
    Evaluate the model on test data
    
    Args:
        model_path: Path to the model
        test_file_path: Path to test file or directory
        output_dir: Directory to save results
        noise_reduce_factor: Controls noise reduction strength (higher = more aggressive)
    """
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
        if os.path.isfile(test_file_path):
            files = [test_file_path]
        else:
            files = [os.path.join(test_file_path, f) for f in os.listdir(test_file_path) 
                    if f.endswith('.wav') and '(' in f]
        
        for file_path in files:
            file_name = os.path.basename(file_path)
            
            try:
                true_label = file_name.split('(')[0].strip()
                true_label_id = label_map.get(true_label)
                
                if true_label_id is None:
                    print(f"Warning: Unknown label '{true_label}' in file {file_name}")
                    continue
                
                segments, _ = preprocess_audio(file_path)
                segment_predictions = []
                segment_probabilities = []
                
                for segment in segments:
                    predicted_class, probabilities = predict(model, feature_extractor, segment)
                    segment_predictions.append(predicted_class)
                    segment_probabilities.append(probabilities)
                
                # Get final prediction
                non_none_predictions = [p for p in segment_predictions if p != 0]
                final_prediction = max(set(non_none_predictions), key=non_none_predictions.count) if non_none_predictions else max(set(segment_predictions), key=segment_predictions.count)
                predicted_label = rev_label_map[final_prediction]
                avg_confidence = np.mean([probs[final_prediction] for probs in segment_probabilities])
                
                # Print simple result
                print(f"File: {file_name:<30} | True: {true_label:<10} | Predicted: {predicted_label:<10} | Confidence: {avg_confidence:.3f}")
                
                all_predictions.append(final_prediction)
                all_true_labels.append(true_label_id)
                results.append({
                    'file': file_name,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'confidence': avg_confidence,
                    'correct': true_label == predicted_label
                })
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
    
    # Generate and save results
    if all_predictions:
        # Filter out 'none' labels (label_id 0) for accuracy calculation
        non_none_indices = [i for i, label in enumerate(all_true_labels) if label != 0]
        filtered_true_labels = [all_true_labels[i] for i in non_none_indices]
        filtered_predictions = [all_predictions[i] for i in non_none_indices]
        
        # Generate full report including 'none'
        report = classification_report(
            all_true_labels,
            all_predictions,
            labels=list(label_map.values()),
            target_names=list(label_map.keys()),
            digits=4,
            zero_division=0
        )
        
        # Calculate accuracy excluding 'none'
        non_none_accuracy = (np.array(filtered_predictions) == np.array(filtered_true_labels)).mean() if filtered_true_labels else 0
        
        # Save results
        with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w', encoding='utf-8') as f:
            f.write("Classification Report (Including 'none'):\n")
            f.write(report)
            f.write(f"\nAccuracy (Excluding 'none'): {non_none_accuracy:.4f}")
            f.write("\n\nDetailed Results:\n")
            for result in results:
                f.write(f"\nFile: {result['file']}")
                f.write(f"\nTrue Label: {result['true_label']}")
                f.write(f"\nPredicted Label: {result['predicted_label']}")
                f.write(f"\nConfidence: {result['confidence']:.2f}")
                f.write(f"\nCorrect: {result['correct']}\n")
        
        print("\n=== Classification Report (Including 'none') ===")
        print(report)
        
        print(f"\nAccuracy (Excluding 'none'): {non_none_accuracy:.4f}")
        
        # Generate confusion matrix
        plot_confusion_matrix(all_true_labels, all_predictions, output_dir)
    else:
        print("No predictions were made. Check if test files exist and are properly labeled.")

if __name__ == "__main__":
    # Setup Thai font
    setup_thai_font()
    
    # Paths
    model_path = "./models/mixed_audio_train_fold_1"
    test_file_path = "./eval_syn"  # Can be a single file or directory
    output_dir = "./evaluation_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run evaluation
    evaluate_model(model_path, test_file_path, output_dir)
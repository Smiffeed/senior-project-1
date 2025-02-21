# -*- coding: utf-8 -*-

import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Define label mapping (same as in training)
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

def load_audio_segment(file_path, start_time, end_time, feature_extractor):
    """Load and preprocess audio segment."""
    try:
        audio, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Convert timestamps to samples
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Extract segment
        audio_segment = audio[:, start_sample:end_sample]
        
        # Resample if necessary
        if sr != 16000:
            audio_segment = torchaudio.functional.resample(audio_segment, sr, 16000)
        
        # Process with feature extractor
        inputs = feature_extractor(audio_segment.squeeze().numpy(), 
                                 sampling_rate=16000, 
                                 return_tensors="pt", 
                                 padding=True)
        
        return inputs
    except Exception as e:
        print(f"Error in load_audio_segment for {file_path}: {str(e)}")
        raise e

def evaluate_model(model_path, eval_csv):
    """Evaluate model performance using data from CSV file."""
    # Load model and feature extractor
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Read CSV file and handle different possible column names
    df = pd.read_csv(eval_csv)
    print("\nAvailable columns in CSV:", df.columns.tolist())
    
    true_labels = []
    predicted_labels = []
    confidences = []
    
    # Process each row in the CSV
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row['label'] not in label_map:
            print(f"Skipping unknown label: {row['label']}")
            continue
            
        try:
            # Load and process audio segment
            inputs = load_audio_segment(
                row['file_path'],
                float(row['start_time']),
                float(row['end_time']),
                feature_extractor
            )
            
            # Get prediction
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1)
                confidence = torch.max(probs, dim=-1)[0]
                
                true_labels.append(label_map[row['label']])
                predicted_labels.append(prediction.item())
                confidences.append(confidence.item())
            
        except Exception as e:
            print(f"Error processing {row['file_path']}: {e}")
            continue
    
    # Check if we have any predictions
    if len(true_labels) == 0:
        raise ValueError("No predictions were made successfully. Check the audio files and paths.")
    
    # Calculate metrics
    class_names = list(label_map.keys())
    report = classification_report(true_labels, predicted_labels, 
                                 target_names=class_names, 
                                 digits=4,
                                 zero_division=0)  # Added zero_division parameter
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    
    # Set font family that supports Thai characters
    plt.rcParams['font.family'] = 'Cordia New'
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Calculate average confidence per class
    class_confidences = {class_name: [] for class_name in class_names}
    for pred, conf in zip(predicted_labels, confidences):
        class_name = class_names[pred]  # This should now be safe
        class_confidences[class_name].append(conf)
    
    avg_confidences = {class_name: np.mean(confs) if confs else 0 
                      for class_name, confs in class_confidences.items()}
    
    # Calculate accuracy
    accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
    
    # Plot accuracy
    plt.figure()
    plt.plot(range(len(true_labels)), np.array(true_labels) == np.array(predicted_labels), 'o', label='Correct')
    plt.title('Accuracy Plot')
    plt.xlabel('Sample Index')
    plt.ylabel('Correct Prediction')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    
    return {
        'classification_report': report,
        'confusion_matrix': cm,
        'average_confidences': avg_confidences,
        'accuracy': accuracy
    }

if __name__ == "__main__":
    model_path = "./models/humanv1"
    eval_csv = "./csv/eval.csv"  # Path to your evaluation CSV file
    
    print("Starting evaluation...")
    results = evaluate_model(model_path, eval_csv)
    
    print("\nClassification Report:")
    print(results['classification_report'])
    
    print("\nAverage Confidence per Class:")
    for class_name, conf in results['average_confidences'].items():
        print(f"{class_name}: {conf:.4f}")
    
    print(f"\nOverall Accuracy: {results['accuracy']:.4f}")
    print("\nConfusion matrix has been saved as 'confusion_matrix.png'")
    print("Accuracy plot has been saved as 'accuracy_plot.png'")

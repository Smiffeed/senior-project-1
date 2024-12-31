import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import os
import numpy as np
from pydub import AudioSegment

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
        
        # Add debug print
        print(f"Predicted class index: {predicted_class}")
        print(f"Probabilities: {probabilities[0].tolist()}")

    return predicted_class, probabilities[0].tolist()

def merge_detections(detections, threshold=0.4, min_gap=0.5):
    """
    Merge detections with improved handling of different profanity words
    threshold: minimum probability to keep the detection
    min_gap: minimum gap in seconds between detections of the same word
    """
    if not detections:
        return []
        
    # Sort by start time
    detections = sorted(detections, key=lambda x: x[0])
    
    merged = []
    current_group = list(detections[0])
    
    for start, end, prob, word in detections[1:]:
        # If this detection starts after the current group ends (plus min_gap)
        # or if it's a different word
        if start - current_group[1] >= min_gap or word != current_group[3]:
            if current_group[2] >= threshold:
                merged.append(tuple(current_group))
            current_group = [start, end, prob, word]
        else:
            # Merge only if it's the same word and has higher probability
            if prob > current_group[2]:
                current_group[1] = end
                current_group[2] = prob
            else:
                current_group[1] = max(current_group[1], end)
    
    # Add the last group if it meets the threshold
    if current_group[2] >= threshold:
        merged.append(tuple(current_group))
    
    return merged

def censor_audio(file_path, detections):
    try:
        audio = AudioSegment.from_wav(file_path)
        
        # Check if beep.wav exists
        beep_path = "./server/beep.wav"
        if not os.path.exists(beep_path):
            raise FileNotFoundError(f"Beep sound file not found at {beep_path}")
            
        beep = AudioSegment.from_wav(beep_path)

        # Sort detections by start time
        detections = sorted(detections, key=lambda x: x[0])
        
        # Create a new audio segment
        censored_audio = audio[:int(detections[0][0] * 1000)] if detections else audio

        for i, (start, end, _) in enumerate(detections):
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            segment_duration = end_ms - start_ms
            
            # Adjust beep duration to match the censored segment
            adjusted_beep = beep[:segment_duration] if len(beep) > segment_duration else beep + AudioSegment.silent(duration=segment_duration - len(beep))
            
            # Add beep
            censored_audio += adjusted_beep
            
            # Add clean audio until next detection or end
            next_start = int(detections[i+1][0] * 1000) if i < len(detections)-1 else len(audio)
            censored_audio += audio[end_ms:next_start]

        censored_file_path = file_path.rsplit('.', 1)[0] + '_censored.wav'
        censored_audio.export(censored_file_path, format="wav")
        return censored_file_path
        
    except Exception as e:
        print(f"Error in censor_audio: {str(e)}")
        raise

def main():
    # Load your fine-tuned model and feature extractor
    model_path = "./server/models/fine_tuned_wav2vec2_fold_best"  # Update this to your model's path
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

    # Set the model to evaluation mode
    model.eval()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Test file or directory
    test_path = "./test/test2.wav"  # Update this to your test audio file or directory

    def process_file(file_path):
        segments, duration = preprocess_audio(file_path)
        results = []
        
        # Labels matching the fine-tuned model
        labels = ["none", "เย็ดแม่", "กู", "มึง", "เหี้ย"]
        
        for i, segment in enumerate(segments):
            prediction, probabilities = predict(model, feature_extractor, segment)
            
            # Get the probability for the predicted class
            max_prob = probabilities[prediction]
            
            # Debug print for all segments
            print(f"\nSegment {i}:")
            print(f"Predicted class: {labels[prediction]} (index: {prediction})")
            print(f"Probabilities for each class:")
            for label, prob in zip(labels, probabilities):
                print(f"- {label}: {prob:.3f}")
            
            # Check all profanity classes
            for class_idx, prob in enumerate(probabilities):
                if class_idx != 0 and prob > 0.4:  # Skip 'none' class
                    start_time = i * 0.5  # 0.5 seconds hop length
                    end_time = min(start_time + 1, duration)
                    detected_word = labels[class_idx]
                    results.append((start_time, end_time, prob, detected_word))
        
        merged_results = merge_detections(results)
        
        print(f"\nFile: {file_path}")
        if merged_results:
            print("\nDetected profanity:")
            for start, end, prob, word in merged_results:
                print(f"- '{word}' at {start:.2f}s to {end:.2f}s (probability: {prob:.2f})")
            
            try:
                censored_file = censor_audio(file_path, [(start, end, prob) for start, end, prob, _ in merged_results])
                print(f"\nCensored audio saved as: {censored_file}")
            except Exception as e:
                print(f"\nError during censoring: {str(e)}")
        else:
            print("No profanity detected")
        print()

    if os.path.isfile(test_path):
        # Single file
        process_file(test_path)
    elif os.path.isdir(test_path):
        # Directory
        for file_name in os.listdir(test_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(test_path, file_name)
                process_file(file_path)
    else:
        print(f"Error: {test_path} is not a valid file or directory")

if __name__ == "__main__":
    main()

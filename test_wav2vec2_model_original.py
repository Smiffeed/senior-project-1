import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import os
import numpy as np
from pydub import AudioSegment

def preprocess_audio(file_path, segment_length=16000, hop_length=8000):
    audio, sr = torchaudio.load(file_path)
    
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    audio = audio.squeeze().numpy()
    
    # Split audio into overlapping segments
    segments = []
    for start in range(0, len(audio), hop_length):
        end = start + segment_length
        segment = audio[start:end]
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))
        segments.append(segment)
    
    return segments, len(audio) / 16000  # Return segments and total duration

def predict(model, feature_extractor, audio_segment):
    inputs = feature_extractor(audio_segment, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(model.device)
    
    # Generate attention mask if it's not provided
    if 'attention_mask' not in inputs:
        attention_mask = torch.ones_like(input_values)
    else:
        attention_mask = inputs.attention_mask.to(model.device)

    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()

    return predicted_class, probabilities[0][1].item()  # Return class and probability of profanity

def merge_detections(detections, threshold=0.5, min_gap=1.0):
    merged = []
    for start, end, prob in sorted(detections):
        if not merged or start - merged[-1][1] >= min_gap:
            merged.append([start, end, prob])
        else:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] = max(merged[-1][2], prob)
    return [tuple(d) for d in merged if d[2] >= threshold]

def censor_audio(file_path, detections):
    audio = AudioSegment.from_wav(file_path)
    beep = AudioSegment.from_wav("beep.wav")  # Make sure you have a beep.wav file

    for start, end, _ in detections:
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        segment_duration = end_ms - start_ms
        
        # Adjust beep duration to match the censored segment
        adjusted_beep = beep[:segment_duration] if len(beep) > segment_duration else beep + AudioSegment.silent(duration=segment_duration - len(beep))
        
        audio = audio[:start_ms] + adjusted_beep + audio[end_ms:]

    censored_file_path = file_path.rsplit('.', 1)[0] + '_censored.wav'
    audio.export(censored_file_path, format="wav")
    return censored_file_path

def main():
    # Load your fine-tuned model and feature extractor
    model_path = "./models/fine_tuned_wav2vec2"  # Update this to your model's path
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

    # Set the model to evaluation mode
    model.eval()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Test file or directory
    test_path = "test_marie.wav"  # Update this to your test audio file or directory

    def process_file(file_path):
        segments, duration = preprocess_audio(file_path)
        results = []
        for i, segment in enumerate(segments):
            prediction, probability = predict(model, feature_extractor, segment)
            if prediction == 1:  # Profanity detected
                start_time = i * 0.5  # 0.5 seconds hop length
                end_time = min(start_time + 1, duration)  # 1 second segment or end of file
                results.append((start_time, end_time, probability))
        
        merged_results = merge_detections(results, threshold=0.9)
        
        print(f"File: {file_path}")
        if merged_results:
            for start, end, prob in merged_results:
                print(f"Profanity detected from {start:.2f}s to {end:.2f}s (probability: {prob:.2f})")
            
            censored_file = censor_audio(file_path, merged_results)
            print(f"Censored audio saved as: {censored_file}")
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

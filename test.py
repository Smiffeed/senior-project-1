from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

# 1. Load the fine-tuned model for classification
model_path = "./fine_tuned_wav2vec2_th_profanity"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path, local_files_only=True)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, local_files_only=True)

# Load the ASR model for transcription
asr_model_name = "airesearch/wav2vec2-large-xlsr-53-th"
asr_processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_name)

# 2. Prepare your input data
def prepare_audio(file_path):
    audio, sample_rate = torchaudio.load(file_path)
    audio = audio.squeeze().numpy()
    
    # Resample if necessary
    if sample_rate != feature_extractor.sampling_rate:
        audio = torchaudio.functional.resample(torch.tensor(audio), sample_rate, feature_extractor.sampling_rate).numpy()
    
    # Limit audio length to 30 seconds (adjust as needed)
    max_length = 30 * feature_extractor.sampling_rate
    if len(audio) > max_length:
        audio = audio[:max_length]
    
    return audio

# 3. Run inference
def run_inference(audio_file):
    audio = prepare_audio(audio_file)
    chunk_length = 10 * feature_extractor.sampling_rate  # 10 seconds per chunk
    predictions = []
    timestamps = []

    # Transcribe the entire audio
    inputs = asr_processor(audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = asr_model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = asr_processor.batch_decode(predicted_ids)[0]

    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i+chunk_length]
        inputs = feature_extractor(chunk, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        prediction = torch.softmax(logits, dim=1)
        predictions.append(prediction)
        timestamps.append((i / feature_extractor.sampling_rate, (i + len(chunk)) / feature_extractor.sampling_rate))

    # Aggregate predictions
    final_predictions = torch.cat(predictions, dim=0)
    return final_predictions, timestamps, transcription

def censor_audio(audio_file, predictions, timestamps, threshold=0.5):
    audio = AudioSegment.from_wav(audio_file)
    beep = Sine(440).to_audio_segment(duration=1000)  # 1-second beep at 440 Hz

    censored_segments = []
    for pred, (start, end) in zip(predictions, timestamps):
        if pred[1] > threshold:  # Assuming index 1 is the profanity class
            censored_segments.append((int(start * 1000), int(end * 1000)))

    for start, end in censored_segments:
        segment_duration = end - start
        censor_beep = beep[:segment_duration]
        audio = audio[:start] + censor_beep + audio[end:]

    return audio, censored_segments

# Example usage
audio_file = "2.wav"
predictions, timestamps, transcription = run_inference(audio_file)
print(f"Transcription: {transcription}")

censored_audio, censored_segments = censor_audio(audio_file, predictions, timestamps)

print("Censored segments (start_time, end_time):")
for start, end in censored_segments:
    print(f"({start/1000:.2f}s, {end/1000:.2f}s)")

# Save the censored audio
censored_audio.export("censored_" + audio_file, format="wav")

# If you know the class labels, you can map the predicted class to its label
class_labels = ["non-profanity", "profanity"]  # Adjust these labels based on your model
overall_prediction = torch.mean(predictions, dim=0)
predicted_class = torch.argmax(overall_prediction).item()
print(f"Overall prediction: {class_labels[predicted_class]}")

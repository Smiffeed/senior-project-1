import torch
import torchaudio
import pandas as pd
from datasets import Dataset, Audio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load and preprocess the dataset
df = pd.read_csv('audio_dataset.csv')

# Split the dataset
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Create dataset objects
def create_dataset(dataframe):
    def load_audio(file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        return {"array": waveform.squeeze().numpy(), "sampling_rate": sample_rate}

    dataset = Dataset.from_pandas(dataframe)
    dataset = dataset.map(lambda x: {"audio": load_audio(x["file_path"])}, remove_columns=["file_path"])
    return dataset

train_dataset = create_dataset(train_df)
eval_dataset = create_dataset(eval_df)

# Load the pre-trained model and feature extractor
model_name = "airesearch/wav2vec2-large-xlsr-53-th"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Preprocess function
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    sample_rate = feature_extractor.sampling_rate
    
    # Resample if necessary
    if examples["audio"][0]["sampling_rate"] != sample_rate:
        audio_arrays = [torchaudio.functional.resample(torch.tensor(audio), examples["audio"][0]["sampling_rate"], sample_rate).numpy() for audio in audio_arrays]
    
    # Pad or truncate
    max_length = 16000  # Adjust this value based on your needs
    audio_arrays = [audio[:max_length] if len(audio) > max_length else np.pad(audio, (0, max_length - len(audio))) for audio in audio_arrays]
    
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=sample_rate, 
        padding=True,
        return_tensors="pt"
    )
    
    inputs["labels"] = examples["label"]
    return inputs

# Preprocess datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=1000,
    load_best_model_at_end=True,
)

# Define compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_wav2vec2_th_profanity")
feature_extractor.save_pretrained("./fine_tuned_wav2vec2_th_profanity")

import torch
import torchaudio
import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple

# Machine learning imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# Transformers imports
from transformers import (
    Wav2Vec2ForSequenceClassification, 
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Config,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Set environment variables
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["DATASETS_VERBOSITY"] = "info"
os.environ["PYTHONPATH"] = "."

# Configuration
@dataclass
class ModelConfig:
    # Data settings
    csv_file: str = './csv/main.csv'
    model_name: str = "airesearch/wav2vec2-large-xlsr-53-th"
    output_dir: str = './models/profanity_classifier'
    
    # Audio preprocessing
    sampling_rate: int = 16000
    max_length: int = 16000
    audio_padding: float = 0.2
    
    # Training parameters
    num_folds: int = 5
    batch_size: int = 16
    learning_rate: float = 3e-5
    num_epochs: int = 100
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    save_steps: int = 50
    eval_steps: int = 50
    
    # Augmentation settings
    augment_short_words: bool = True
    noise_threshold: float = 0.005
    
    # Label mapping
    label_map: Dict[str, int] = None
    
    def __post_init__(self):
        if self.label_map is None:
            self.label_map = {
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
        self.num_labels = len(self.label_map)
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


class AudioProcessor:
    """Class for audio preprocessing operations"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def preprocess_audio(self, file_path, start_time, end_time):
        """Load and preprocess audio segment with improved signal processing"""
        try:
            # Get sample rate
            metadata = torchaudio.info(file_path)
            sr = metadata.sample_rate
            
            # Load the specific segment
            audio, sr = torchaudio.load(
                file_path, 
                frame_offset=int(start_time * sr), 
                num_frames=int((end_time - start_time) * sr)
            )
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Convert to numpy array for processing
            audio_np = audio.squeeze().numpy()
            
            # Apply Hamming window for smoother frequency response
            window_length = len(audio_np)
            hamming_window = np.hamming(window_length)
            audio_np = audio_np * hamming_window
            
            # Apply pre-emphasis filter to emphasize high frequencies
            audio_np = librosa.effects.preemphasis(audio_np, coef=0.97)
            
            # Spectral subtraction for noise reduction
            noise_threshold = self.config.noise_threshold
            audio_np = np.where(np.abs(audio_np) < noise_threshold, 0, audio_np)
            
            # Convert back to torch tensor
            audio = torch.from_numpy(audio_np).unsqueeze(0)
            
            # Resample to target sampling rate if needed
            if sr != self.config.sampling_rate:
                audio = torchaudio.functional.resample(audio, sr, self.config.sampling_rate)
            
            # Normalize audio to zero mean and unit variance
            audio = (audio - audio.mean()) / (audio.std() + 1e-8)
            
            # Pad or truncate to fixed length
            if audio.shape[1] < self.config.max_length:
                audio = torch.nn.functional.pad(audio, (0, self.config.max_length - audio.shape[1]))
            else:
                audio = audio[:, :self.config.max_length]
            
            return audio.squeeze().numpy()
        
        except Exception as e:
            print(f"Error processing audio file {file_path}: {str(e)}")
            # Return zeros as fallback
            return np.zeros(self.config.max_length)
    
    def augment_audio(self, audio):
        """Apply comprehensive audio augmentation techniques"""
        augmented = audio.copy()
        
        # Select random augmentation types
        augmentation_types = np.random.choice([
            'noise', 'pitch', 'speed', 'time_mask', 'reverb'
        ], size=np.random.randint(1, 4), replace=False)
        
        for aug_type in augmentation_types:
            if aug_type == 'noise':
                # Add different types of noise
                noise_type = np.random.choice(['gaussian', 'pink'])
                noise_level = np.random.uniform(0.001, 0.005)
                
                if noise_type == 'gaussian':
                    noise = np.random.normal(0, noise_level, len(augmented))
                else:  # pink noise
                    noise = librosa.core.pink_noise(len(augmented)) * noise_level
                    
                augmented += noise
                
            elif aug_type == 'pitch':
                # Pitch shifting with controlled range
                pitch_shift = np.random.uniform(-2, 2)
                augmented = librosa.effects.pitch_shift(
                    augmented, 
                    sr=self.config.sampling_rate, 
                    n_steps=pitch_shift,
                    bins_per_octave=200
                )
                
            elif aug_type == 'speed':
                # Time stretching with careful range to avoid distortion
                speed_factor = np.random.uniform(0.85, 1.15)
                augmented = librosa.effects.time_stretch(augmented, rate=speed_factor)
                
            elif aug_type == 'time_mask':
                # Time masking simulates missing information
                mask_size = int(len(augmented) * np.random.uniform(0.05, 0.15))
                mask_start = np.random.randint(0, len(augmented) - mask_size)
                augmented[mask_start:mask_start + mask_size] = 0
                
            elif aug_type == 'reverb':
                # Simple reverb effect
                reverb_delay = np.random.randint(1000, 3000)
                decay = np.random.uniform(0.1, 0.5)
                reverb = np.exp(-decay * np.linspace(0, 1, reverb_delay))
                augmented = np.convolve(augmented, reverb, mode='full')[:len(augmented)]
        
        # Normalize after augmentation
        if np.max(np.abs(augmented)) > 0:
            augmented = augmented / (np.max(np.abs(augmented)) + 1e-6)
            
        return augmented
    
    def augment_short_words(self, audio, label):
        """Enhanced augmentation specifically for short words that need more samples"""
        short_words = ['กู', 'มึง']
        
        if label in short_words and self.config.augment_short_words:
            augmented_samples = []
            
            # Create multiple variations of short words
            for _ in range(3):
                aug_audio = audio.copy()
                
                # Apply chain of carefully tuned augmentations
                # 1. Volume variation
                volume_factor = np.random.uniform(0.8, 1.2)
                aug_audio = aug_audio * volume_factor
                
                # 2. Time stretching with controlled range
                stretch_factor = np.random.uniform(0.85, 1.15)
                aug_audio = librosa.effects.time_stretch(aug_audio, rate=stretch_factor)
                
                # 3. Pitch shifting with finer control
                n_steps = np.random.uniform(-2, 2)
                aug_audio = librosa.effects.pitch_shift(
                    aug_audio, 
                    sr=self.config.sampling_rate, 
                    n_steps=n_steps,
                    bins_per_octave=200
                )
                
                # 4. Add subtle background noise
                noise_level = np.random.uniform(0.0005, 0.002)
                noise = np.random.normal(0, noise_level, len(aug_audio))
                aug_audio += noise
                
                # Normalize
                if np.max(np.abs(aug_audio)) > 0:
                    aug_audio = aug_audio / (np.max(np.abs(aug_audio)) + 1e-6)
                
                augmented_samples.append(aug_audio)
            
            return augmented_samples
            
        return [audio]  # Return original if not a short word


class DatasetPreparer:
    """Class for preparing and processing the dataset"""
    
    def __init__(self, config: ModelConfig, feature_extractor, audio_processor: AudioProcessor):
        self.config = config
        self.feature_extractor = feature_extractor
        self.audio_processor = audio_processor
    
    def load_dataset(self, csv_file):
        """Load dataset from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded dataset with {len(df)} samples")
            
            # Check for missing labels
            missing_labels = [label for label in df['label'].unique() if label not in self.config.label_map]
            if missing_labels:
                print(f"Warning: Found labels not in label_map: {missing_labels}")
                
            return df
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return pd.DataFrame()
    
    def prepare_dataset(self, df):
        """Prepare dataset for Hugging Face Trainer"""
        
        def process_example(example):
            file_path = example['file_path'].replace('\\', '/')
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None
                
            # Get audio length first
            try:
                metadata = torchaudio.info(file_path)
                audio_length = metadata.num_frames / metadata.sample_rate
                
                # Add padding around the segments
                padding = self.config.audio_padding
                start_time = max(0, example['start_time'] - padding)
                end_time = min(example['end_time'] + padding, audio_length)
                
                # Process audio with noise removal
                audio = self.audio_processor.preprocess_audio(file_path, start_time, end_time)
                
                # Apply feature extractor
                inputs = self.feature_extractor(
                    audio, 
                    sampling_rate=self.config.sampling_rate, 
                    return_tensors="pt", 
                    padding=True
                )
                
                # Check if label is valid
                label = self.config.label_map.get(example['label'])
                if label is None:
                    print(f"Warning: Label {example['label']} not found in label_map for file {file_path}")
                    return None
                
                return {
                    'input_values': inputs.input_values.squeeze().numpy(),
                    'attention_mask': inputs.attention_mask.squeeze().numpy(),
                    'label': label,
                    'original_label': example['label']  # Keep original label for augmentation
                }
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                return None
        
        dataset = Dataset.from_pandas(df)
        processed_dataset = dataset.map(process_example, remove_columns=dataset.column_names)
        filtered_dataset = processed_dataset.filter(lambda x: x is not None)
        
        return filtered_dataset
    
    def augment_dataset(self, dataset):
        """Augment dataset with focus on underrepresented classes"""
        
        # Calculate class distribution
        labels = [item['label'] for item in dataset]
        class_counts = {i: labels.count(i) for i in range(self.config.num_labels)}
        
        print("\nClass distribution before augmentation:")
        for label_id, count in class_counts.items():
            label_name = self.config.id_to_label.get(label_id, f"Unknown-{label_id}")
            print(f"{label_name}: {count} samples")
        
        augmented_examples = []
        
        # Focus augmentation on underrepresented classes
        for item in dataset:
            label_id = item['label']
            label_name = self.config.id_to_label.get(label_id)
            
            # For classes with few samples, create more augmentations
            if class_counts[label_id] < 10:
                num_augmentations = 5
            elif class_counts[label_id] < 20:
                num_augmentations = 3
            elif label_name in ['กู', 'มึง']:  # Special focus on short words
                num_augmentations = 2
            else:
                num_augmentations = 0
            
            for _ in range(num_augmentations):
                # Apply augmentation
                aug_input = self.audio_processor.augment_audio(np.array(item['input_values']))
                
                # Create new example
                augmented_examples.append({
                    'input_values': aug_input,
                    'attention_mask': item['attention_mask'],
                    'label': item['label'],
                    'original_label': item.get('original_label', '')
                })
        
        # Add augmented examples to dataset
        augmented_dataset = dataset.add_items(augmented_examples)
        
        # Calculate new class distribution
        new_labels = [item['label'] for item in augmented_dataset]
        new_class_counts = {i: new_labels.count(i) for i in range(self.config.num_labels)}
        
        print("\nClass distribution after augmentation:")
        for label_id, count in new_class_counts.items():
            label_name = self.config.id_to_label.get(label_id, f"Unknown-{label_id}")
            print(f"{label_name}: {count} samples")
        
        return augmented_dataset


class CustomTrainer(Trainer):
    """Custom trainer with weighted loss function"""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply class weights to CrossEntropyLoss
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


class ProfanityClassifier:
    """Main class for profanity classification model"""
    
    def __init__(self, config: ModelConfig = None):
        """Initialize the profanity classifier"""
        self.config = config or ModelConfig()
        
        # Initialize feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.config.model_name,
            return_attention_mask=True,
            do_normalize=True,
        )
        
        # Initialize processors
        self.audio_processor = AudioProcessor(self.config)
        self.dataset_preparer = DatasetPreparer(
            self.config, 
            self.feature_extractor,
            self.audio_processor
        )
        
        # For storing results
        self.fold_performances = {}
        self.best_model_path = None
    
    def calculate_class_weights(self, df):
        """Calculate balanced class weights"""
        labels = [self.config.label_map[label] for label in df['label']]
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        return torch.FloatTensor(class_weights)
    
    def collate_fn(self, batch):
        """Custom collation function for variable length inputs"""
        input_values = [torch.tensor(item['input_values']).squeeze() for item in batch]
        attention_mask = [torch.tensor(item['attention_mask']).squeeze() for item in batch]
        labels = torch.tensor([item['label'] for item in batch])
        
        input_values = pad_sequence(input_values, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        
        return {
            'input_values': input_values,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted')
        
        accuracy = (predictions == labels).astype(np.float32).mean().item()
        
        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    
    def train_fold(self, train_dataset, val_dataset, fold_num):
        """Train model for a specific fold"""
        # Output directory for this fold
        fold_output_dir = f"{self.config.output_dir}_fold_{fold_num}"
        Path(fold_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize Wav2Vec2 model with classification head
        config = Wav2Vec2Config.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            finetuning_task="audio-classification"
        )
        
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.config.model_name,
            config=config,
        )
        
        # Calculate class distribution in training set
        train_labels = [item['label'] for item in train_dataset]
        unique_labels = np.unique(train_labels)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_labels,
            y=train_labels
        )
        
        # Map class weights to all possible classes
        full_class_weights = np.ones(self.config.num_labels)
        for i, label in enumerate(unique_labels):
            full_class_weights[label] = class_weights[i]
            
        class_weights_tensor = torch.FloatTensor(full_class_weights)
        
        # Print class weights
        print(f"\nClass weights for fold {fold_num}:")
        for label_id, weight in enumerate(full_class_weights):
            label_name = self.config.id_to_label.get(label_id, f"Unknown-{label_id}")
            print(f"{label_name}: {weight:.4f}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            logging_dir=f"{fold_output_dir}/logs",
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.eval_steps,
            learning_rate=self.config.learning_rate,
            save_total_limit=2,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            fp16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            eval_strategy="steps",
            push_to_hub=False,
            remove_unused_columns=True,
        )
        
        # Initialize custom trainer
        trainer = CustomTrainer(
            class_weights=class_weights_tensor,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
            data_collator=self.collate_fn,
        )
        
        # Train the model
        print(f"\nTraining model for fold {fold_num}...")
        trainer.train()
        
        # Evaluate on validation set
        eval_results = trainer.evaluate()
        print(f"\nFold {fold_num} validation results:")
        print(eval_results)
        
        # Save the model
        trainer.save_model(fold_output_dir)
        self.feature_extractor.save_pretrained(fold_output_dir)
        
        return {
            'model_path': fold_output_dir,
            'metrics': eval_results
        }
    
    def train_kfold(self):
        """Train model using k-fold cross validation"""
        print("Starting k-fold cross validation training...")
        
        # Load dataset
        df = self.dataset_preparer.load_dataset(self.config.csv_file)
        if df.empty:
            print("Dataset is empty. Cannot proceed with training.")
            return
        
        # Handle imbalanced class distribution
        for label, count in df['label'].value_counts().items():
            if count < 2:
                # Find the row with this label
                row_to_duplicate = df[df['label'] == label].iloc[0]
                # Add it to the dataframe again
                df = pd.concat([df, pd.DataFrame([row_to_duplicate])], ignore_index=True)
        
        # Prepare dataset
        dataset = self.dataset_preparer.prepare_dataset(df)
        
        # Augment dataset
        dataset = self.dataset_preparer.augment_dataset(dataset)
        
        # K-fold cross validation
        kf = KFold(n_splits=self.config.num_folds, shuffle=True, random_state=42)
        best_accuracy = 0
        best_fold = 0
        
        # Convert dataset to list for k-fold
        dataset_list = list(range(len(dataset)))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_list), 1):
            print(f"\n{'='*50}\nTraining fold {fold}/{self.config.num_folds}\n{'='*50}")
            
            # Create train and validation datasets
            train_subset = dataset.select(train_idx)
            val_subset = dataset.select(val_idx)
            
            print(f"Train set: {len(train_subset)} samples")
            print(f"Validation set: {len(val_subset)} samples")
            
            # Train model for this fold
            fold_results = self.train_fold(train_subset, val_subset, fold)
            
            # Track best model
            accuracy = fold_results['metrics'].get('accuracy', 0)
            self.fold_performances[fold] = {
                'accuracy': accuracy,
                'metrics': fold_results['metrics'],
                'model_path': fold_results['model_path']
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_fold = fold
        
        # Save best model
        best_model_path = self.fold_performances[best_fold]['model_path']
        best_model_save_path = f"{self.config.output_dir}_best_model"
        
        print(f"\nBest model is from fold {best_fold} with accuracy {best_accuracy:.4f}")
        print(f"Copying best model to {best_model_save_path}")
        
        # Load and save best model
        model = Wav2Vec2ForSequenceClassification.from_pretrained(best_model_path)
        model.save_pretrained(best_model_save_path)
        self.feature_extractor.save_pretrained(best_model_save_path)
        
        self.best_model_path = best_model_save_path
        
        # Generate performance visualization
        self.visualize_performance()
        
        return self.best_model_path
    
    def visualize_performance(self):
        """Create visualizations of model performance"""
        if not self.fold_performances:
            print("No performance data available for visualization")
            return
        
        # Create output directory for visualizations
        vis_dir = os.path.join(self.config.output_dir, 'visualizations')
        Path(vis_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. K-fold accuracy comparison
        plt.figure(figsize=(10, 6))
        
        # Prepare data
        folds = list(self.fold_performances.keys())
        accuracies = [data['accuracy'] for data in self.fold_performances.values()]
        
        # Create bar plot with custom styling
        sns.set_style("whitegrid")
        bars = plt.bar(folds, accuracies, color=sns.color_palette("viridis", len(folds)))
        
        # Customize plot
        plt.title('Model Accuracy Across K-Folds', fontsize=16)
        plt.xlabel('Fold Number', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.ylim(0, 1.0)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'kfold_accuracy.png'), dpi=300)
        plt.close()
        
        print(f"Performance visualizations saved to {vis_dir}")
    
    def evaluate(self, test_data_path, model_path=None):
        """Evaluate model on test data"""
        # Use best model path if not specified
        model_path = model_path or self.best_model_path
        if not model_path:
            print("No model available for evaluation")
            return
        
        print(f"Evaluating model from {model_path}")
        
        # Load model
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        test_dataset = self.dataset_preparer.prepare_dataset(test_df)
        
        # Evaluate model
        predictions = []
        true_labels = []
        
        for item in test_dataset:
            # Prepare inputs
            input_values = torch.tensor([item['input_values']]).float()
            attention_mask = torch.tensor([item['attention_mask']]).float()
            
            # Get prediction
            with torch.no_grad():
                outputs = model(input_values=input_values, attention_mask=attention_mask)
                pred = torch.argmax(outputs.logits, dim=-1).item()
                
            # Store prediction and true label
            predictions.append(pred)
            true_labels.append(item['label'])
        
        # Calculate metrics
        accuracy = sum(p == l for p, l in zip(predictions, true_labels)) / len(true_labels)
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Get classification report
        class_names = [self.config.id_to_label.get(i, f"Unknown-{i}") 
                      for i in range(self.config.num_labels)]
        report = classification_report(true_labels, predictions, 
                                     target_names=class_names, 
                                     zero_division=0)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        
        # Save confusion matrix
        vis_dir = os.path.join(self.config.output_dir, 'visualizations')
        Path(vis_dir).mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm
        }


def main():
    """Main function to run the training and evaluation"""
    # Initialize configuration
    config = ModelConfig(
        csv_file='./csv/main.csv',
        model_name="airesearch/wav2vec2-large-xlsr-53-th",
        output_dir='./models/profanity_classifier',
        num_folds=5,
        batch_size=16,
        learning_rate=3e-5,
        num_epochs=100
    )
    
    # Initialize classifier
    classifier = ProfanityClassifier(config)
    
    # Train model using k-fold cross validation
    best_model_path = classifier.train_kfold()
    
    # Evaluate model on test set if available
    test_data_path = './csv/test.csv'
    if os.path.exists(test_data_path):
        classifier.evaluate(test_data_path, best_model_path)
    else:
        print(f"Test data not found at {test_data_path}. Skipping evaluation.")


if __name__ == "__main__":
    main()
import os
import torch
import numpy as np
import pandas as pd
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import Wav2Vec2FeatureExtractor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from pathlib import Path

class ModelEvaluator:
    def __init__(
        self, 
        model_path, 
        eval_csv_path,
        output_dir='./evaluation_results',
        window_size=0.5,
        window_overlap=0.25,
        target_sr=16000
    ):
        self.model_path = model_path
        self.eval_csv_path = eval_csv_path
        self.output_dir = output_dir
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.target_sr = target_sr
        
        self.label_map = {
            'none': 0, 'เย็ด': 1, 'กู': 2, 'มึง': 3, 'เหี้ย': 4,
            'ควย': 5, 'สวะ': 6, 'หี': 7, 'แตด': 8
        }
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model and feature extractor
        self.setup()
        
    def setup(self):
        """Load model and feature extractor"""
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        print(f"Loading model from {self.model_path}")
        
        try:
            # Try using the helper loader
            import sys
            sys.path.append(os.path.abspath(self.model_path))
            if os.path.exists(os.path.join(self.model_path, "load_helper.py")):
                print("Using load helper")
                # Make sure we have the right path
                if self.model_path not in sys.path:
                    sys.path.append(os.path.abspath(self.model_path))
                # Dynamically import the module
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "load_helper", 
                    os.path.join(self.model_path, "load_helper.py")
                )
                load_helper = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(load_helper)
                
                # Now use the loaded module
                self.model = load_helper.load_model(num_labels=len(self.label_map), device=self.device)
            else:
                # Fall back to standard loading
                from improved_model import create_improved_model
                
                # Create the model
                self.model = create_improved_model(
                    "airesearch/wav2vec2-large-xlsr-53-th", 
                    num_labels=len(self.label_map)
                )
                
                # Try to load weights
                model_weights_path = os.path.join(self.model_path, "pytorch_model.bin")
                if os.path.exists(model_weights_path):
                    print(f"Loading weights from {model_weights_path}")
                    self.model.load_state_dict(
                        torch.load(model_weights_path, map_location=self.device),
                        strict=False
                    )
                else:
                    print("No model weights found, using uninitialized model")
        
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
        # Load feature extractor
        print("Loading feature extractor")
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_path)
        except:
            print("Falling back to base model's feature extractor")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "airesearch/wav2vec2-large-xlsr-53-th"
            )
        
    def preprocess_audio(self, file_path, start_time=None, end_time=None):
        """Process a single audio file or segment"""
        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Extract segment if requested
            if start_time is not None and end_time is not None:
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                waveform = waveform[:, start_sample:end_sample]
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
                waveform = resampler(waveform)
            
            # Apply peak normalization
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
            
            # Convert to numpy
            audio_array = waveform.squeeze().numpy()
            
            # Process with feature extractor
            inputs = self.feature_extractor(
                audio_array, 
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding=True
            )
            
            return inputs
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
            
    def predict_single(self, file_path, start_time=None, end_time=None):
        """Make prediction for a single audio file or segment"""
        inputs = self.preprocess_audio(file_path, start_time, end_time)
        if inputs is None:
            return None, None
        
        # Move to device
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if 'attention_mask' in inputs else None
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_values=input_values, attention_mask=attention_mask)
    
        # Handle different output formats
        if isinstance(outputs, dict):
            # Custom model output
            logits = outputs['logits'] if 'logits' in outputs else outputs.get('last_hidden_state', None)
        else:
            # Standard transformers output
            logits = outputs.logits
    
        # Get predicted class and confidence
        probabilities = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, pred_class].item()
        
        return self.inv_label_map[pred_class], confidence
        
    def evaluate_from_csv(self):
        """Evaluate model on dataset specified in CSV file"""
        # Load CSV
        df = pd.read_csv(self.eval_csv_path)
        print(f"Loaded evaluation data with {len(df)} samples")
        
        # Create results dataframe
        results = []
        
        # Process each row
        for _, row in tqdm(df.iterrows(), total=len(df)):
            file_path = row['file_path']
            start_time = row['start_time']
            end_time = row['end_time']
            true_label = row['label'] if 'label' in row else None
            
            # Make prediction
            pred_label, confidence = self.predict_single(file_path, start_time, end_time)
            
            # Save results
            results.append({
                'file_path': file_path,
                'start_time': start_time,
                'end_time': end_time,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': confidence
            })
        
        # Convert to dataframe
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = os.path.join(self.output_dir, 'evaluation_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
        
        # Calculate metrics
        self.calculate_metrics(results_df)
        
    def calculate_metrics(self, results_df):
        """Calculate and save evaluation metrics"""
        # Filter out rows with NaN values
        filtered_df = results_df.dropna(subset=['true_label', 'predicted_label'])
        
        # Get true and predicted labels
        y_true = [self.label_map.get(label, 0) for label in filtered_df['true_label']]
        y_pred = [self.label_map.get(label, 0) for label in filtered_df['predicted_label']]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Save metrics to file
        metrics_path = os.path.join(self.output_dir, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Weighted Precision: {precision:.4f}\n")
            f.write(f"Weighted Recall: {recall:.4f}\n")
            f.write(f"Weighted F1: {f1:.4f}\n\n")
            
            f.write("Per-class metrics:\n")
            for i, label in self.inv_label_map.items():
                f.write(f"{label}:\n")
                f.write(f"  Precision: {class_precision[i]:.4f}\n")
                f.write(f"  Recall: {class_recall[i]:.4f}\n")
                f.write(f"  F1: {class_f1[i]:.4f}\n")
                f.write(f"  Support: {np.sum(np.array(y_true) == i)}\n\n")
        
        # Generate classification report
        from sklearn.metrics import classification_report
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=[self.inv_label_map[i] for i in sorted(self.inv_label_map.keys())],
            digits=4
        )
        
        report_path = os.path.join(self.output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("Classification Report:\n")
            f.write(report)
            
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[self.inv_label_map[i] for i in sorted(self.inv_label_map.keys())],
            yticklabels=[self.inv_label_map[i] for i in sorted(self.inv_label_map.keys())]
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        cm_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        
        # Plot confidence distribution by class
        plt.figure(figsize=(12, 6))
        for label in filtered_df['true_label'].unique():
            subset = filtered_df[filtered_df['true_label'] == label]
            correct = subset[subset['true_label'] == subset['predicted_label']]['confidence']
            incorrect = subset[subset['true_label'] != subset['predicted_label']]['confidence']
            
            if len(correct) > 0:
                sns.kdeplot(correct, label=f"{label} (correct)", shade=True)
            if len(incorrect) > 0:
                sns.kdeplot(incorrect, label=f"{label} (incorrect)", linestyle='--')
                
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.title('Confidence Distribution by Class and Correctness')
        plt.legend()
        
        conf_path = os.path.join(self.output_dir, 'confidence_distribution.png')
        plt.tight_layout()
        plt.savefig(conf_path, dpi=300)
        
        print(f"Metrics saved to {metrics_path}")
        print(f"Classification report saved to {report_path}")
        print(f"Confusion matrix plot saved to {cm_path}")
        print(f"Confidence distribution plot saved to {conf_path}")
        
    def evaluate_windowed(self, audio_file, window_size=None, window_overlap=None, true_label=None):
        """Evaluate an audio file using windowing approach"""
        if window_size is None:
            window_size = self.window_size
        if window_overlap is None:
            window_overlap = self.window_overlap
            
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_file)
        duration = waveform.shape[1] / sample_rate
        
        # Create windows
        stride = window_size - window_overlap
        results = []
        
        current_time = 0
        while current_time + window_size <= duration:
            # Extract window
            start_time = current_time
            end_time = current_time + window_size
            
            # Make prediction
            pred_label, confidence = self.predict_single(audio_file, start_time, end_time)
            
            # Save result
            results.append({
                'file_path': audio_file,
                'start_time': start_time,
                'end_time': end_time,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': confidence
            })
            
            # Move to next window
            current_time += stride
        
        # Convert to dataframe
        results_df = pd.DataFrame(results)
        
        # Save results
        file_name = Path(audio_file).stem
        results_path = os.path.join(self.output_dir, f'windowed_{file_name}.csv')
        results_df.to_csv(results_path, index=False)
        
        # Plot predictions over time
        plt.figure(figsize=(15, 5))
        
        # Create a color map for labels
        labels = sorted(set(self.label_map.keys()))
        cmap = plt.cm.get_cmap('tab10', len(labels))
        label_to_color = {label: i for i, label in enumerate(labels)}
        
        # Plot segments
        for _, row in results_df.iterrows():
            color = cmap(label_to_color.get(row['predicted_label'], 0))
            plt.axvspan(
                row['start_time'], 
                row['end_time'], 
                alpha=row['confidence'], 
                color=color, 
                label=row['predicted_label']
            )
            
        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.xlabel('Time (s)')
        plt.title(f'Predictions for {file_name}')
        
        plot_path = os.path.join(self.output_dir, f'windowed_{file_name}.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        
        return results_df

if __name__ == "__main__":
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path="./output/improved_model/final",
        eval_csv_path="./csv/eval.csv",
        output_dir="./evaluation_results/improved"
    )
    
    # Evaluate from CSV
    evaluator.evaluate_from_csv()
    
    # Evaluate specific files with windowing
    test_files = [
        "test.wav",
        "test1.wav"
    ]
    
    for file in test_files:
        print(f"Evaluating {file} with windowing...")
        evaluator.evaluate_windowed(file)

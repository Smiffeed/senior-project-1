#!/usr/bin/env python3
"""
Improved training strategies for Thai profanity detection
Based on evaluation results analysis
"""

import torch
import numpy as np
from sklearn.metrics import f1_score
from transformers import TrainingArguments
import torch.nn.functional as F

class AdaptiveTrainingStrategy:
    """
    Multi-stage training strategy based on evaluation insights
    """
    
    def __init__(self, model, train_data, val_data):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        
    def stage_1_binary_pretraining(self):
        """
        Stage 1: Train binary classifier with larger windows (2.0s optimal)
        Focus on profane vs non-profane distinction
        """
        training_args = TrainingArguments(
            output_dir='./binary_pretrain',
            num_train_epochs=10,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1_binary",
            greater_is_better=True,
            # Use longer sequences for binary classification
            dataloader_num_workers=4,
        )
        
        # Binary labels: 0=none, 1=profane
        binary_train_data = self.convert_to_binary_labels(self.train_data)
        binary_val_data = self.convert_to_binary_labels(self.val_data)
        
        return training_args, binary_train_data, binary_val_data
    
    def stage_2_multiclass_finetuning(self):
        """
        Stage 2: Fine-tune for multiclass classification with shorter windows (0.3s optimal)
        Build on binary foundation
        """
        training_args = TrainingArguments(
            output_dir='./multiclass_finetune',
            num_train_epochs=15,
            per_device_train_batch_size=32,  # Smaller windows allow larger batch
            per_device_eval_batch_size=64,
            warmup_steps=200,
            weight_decay=0.005,  # Reduced for fine-tuning
            learning_rate=1e-5,  # Lower LR for fine-tuning
            logging_dir='./logs',
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=300,
            load_best_model_at_end=True,
            metric_for_best_model="f1_multiclass",
            greater_is_better=True,
        )
        
        return training_args
    
    def convert_to_binary_labels(self, data):
        """Convert multiclass labels to binary (profane/non-profane)"""
        binary_data = []
        for example in data:
            binary_label = 0 if example['label'] == 0 else 1  # 0=none, 1=profane
            binary_data.append({
                'input_values': example['input_values'],
                'label': binary_label
            })
        return binary_data

class AdvancedLossFunction:
    """
    Advanced loss functions based on evaluation insights
    """
    
    @staticmethod
    def adaptive_focal_loss(logits, labels, alpha=None, gamma=2.0, class_weights=None):
        """
        Adaptive focal loss that adjusts based on class performance
        """
        if alpha is None:
            # Adaptive alpha based on class frequency
            unique_labels, label_counts = torch.unique(labels, return_counts=True)
            total_samples = len(labels)
            alpha = torch.ones(logits.size(1), device=logits.device)
            
            for label, count in zip(unique_labels, label_counts):
                # Inverse frequency weighting
                alpha[label] = total_samples / (len(unique_labels) * count)
        
        if class_weights is not None:
            class_weights = class_weights.to(logits.device)
        
        ce_loss = F.cross_entropy(logits, labels, weight=class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Gather alpha values for each sample
        alpha_t = alpha[labels]
        
        focal_loss = alpha_t * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    @staticmethod
    def temporal_consistency_loss(logits, labels, attention_weights, lambda_consistency=0.1):
        """
        Encourage temporal consistency in predictions
        """
        base_loss = F.cross_entropy(logits, labels)
        
        # Temporal consistency regularization
        if attention_weights is not None:
            # Penalize rapid attention changes
            attention_diff = torch.diff(attention_weights, dim=-1)
            consistency_loss = torch.mean(torch.square(attention_diff))
            
            total_loss = base_loss + lambda_consistency * consistency_loss
            return total_loss
        
        return base_loss

class DataAugmentationStrategy:
    """
    Sophisticated data augmentation based on evaluation insights
    """
    
    @staticmethod
    def context_aware_augmentation(audio, labels, profanity_segments):
        """
        Augment data while preserving profanity temporal boundaries
        """
        augmented_samples = []
        
        # For each profanity segment, create context variations
        for segment_info in profanity_segments:
            start_time, end_time, label = segment_info
            
            # Extract profanity segment
            segment_samples = int(start_time * 16000), int(end_time * 16000)
            profanity_audio = audio[segment_samples[0]:segment_samples[1]]
            
            # Context window variations (different pre/post context lengths)
            context_variations = [0.1, 0.2, 0.3, 0.5]  # seconds
            
            for context_len in context_variations:
                context_samples = int(context_len * 16000)
                
                # Extract with context
                start_with_context = max(0, segment_samples[0] - context_samples)
                end_with_context = min(len(audio), segment_samples[1] + context_samples)
                
                contextualized_audio = audio[start_with_context:end_with_context]
                
                # Apply augmentations
                augmentations = [
                    DataAugmentationStrategy.add_realistic_noise(contextualized_audio),
                    DataAugmentationStrategy.time_stretch_preserve_pitch(contextualized_audio, 0.95),
                    DataAugmentationStrategy.time_stretch_preserve_pitch(contextualized_audio, 1.05),
                    DataAugmentationStrategy.volume_variation(contextualized_audio, 0.8),
                    DataAugmentationStrategy.volume_variation(contextualized_audio, 1.2),
                ]
                
                augmented_samples.extend(augmentations)
        
        return augmented_samples
    
    @staticmethod
    def add_realistic_noise(audio, noise_types=['white', 'pink', 'brown']):
        """Add realistic background noise"""
        import random
        
        noise_type = random.choice(noise_types)
        noise_level = random.uniform(0.005, 0.02)
        
        if noise_type == 'white':
            noise = np.random.normal(0, noise_level, len(audio))
        elif noise_type == 'pink':
            # Pink noise (1/f noise)
            freqs = np.fft.fftfreq(len(audio))
            freqs[0] = 1  # Avoid division by zero
            pink_noise = np.fft.ifft(np.random.normal(0, 1, len(audio)) / np.sqrt(np.abs(freqs))).real
            noise = pink_noise * noise_level
        else:  # brown noise
            # Brown noise (1/f^2 noise)
            freqs = np.fft.fftfreq(len(audio))
            freqs[0] = 1
            brown_noise = np.fft.ifft(np.random.normal(0, 1, len(audio)) / np.abs(freqs)).real
            noise = brown_noise * noise_level
        
        return audio + noise
    
    @staticmethod
    def time_stretch_preserve_pitch(audio, stretch_factor):
        """Time stretch while preserving pitch"""
        import librosa
        return librosa.effects.time_stretch(audio, rate=stretch_factor)
    
    @staticmethod
    def volume_variation(audio, volume_factor):
        """Apply volume variation"""
        return audio * volume_factor

class OptimalWindowSelector:
    """
    Dynamic window selection based on evaluation results
    """
    
    def __init__(self, evaluation_results):
        self.eval_results = evaluation_results
        
        # From your evaluation results
        self.optimal_windows = {
            'binary': 2.0,      # Best for binary classification
            'multiclass': 0.3,  # Best for multiclass classification
            'iou': 0.5,         # Balanced for IoU evaluation
        }
        
        self.optimal_strides = {
            'binary': 1.9,      # 95% overlap for binary
            'multiclass': 0.15,  # 50% overlap for multiclass
            'iou': 0.25,        # 50% overlap for IoU
        }
    
    def get_optimal_config(self, task_type='multiclass'):
        """Get optimal window/stride configuration for specific task"""
        return {
            'window_size': self.optimal_windows.get(task_type, 0.5),
            'stride_size': self.optimal_strides.get(task_type, 0.25),
        }
    
    def adaptive_windowing(self, audio_length, task_type='multiclass'):
        """
        Create adaptive windowing strategy based on audio length and task
        """
        config = self.get_optimal_config(task_type)
        window_size = config['window_size']
        stride_size = config['stride_size']
        
        # Adjust for very short or very long audio
        if audio_length < 2.0:
            # For short audio, use smaller windows
            window_size = min(window_size, audio_length * 0.8)
            stride_size = window_size * 0.5
        elif audio_length > 10.0:
            # For long audio, consider larger strides to reduce computation
            stride_size = max(stride_size, window_size * 0.3)
        
        return window_size, stride_size

def create_improved_training_pipeline():
    """
    Create comprehensive training pipeline with all improvements
    """
    pipeline_config = {
        'model_architecture': 'ThaiProfanityDetector',  # Use enhanced architecture
        'training_strategy': 'multi_stage',  # Binary pretraining + multiclass fine-tuning
        'loss_function': 'adaptive_focal_loss',
        'data_augmentation': 'context_aware',
        'window_selection': 'task_adaptive',
        'evaluation_metrics': ['f1_binary', 'f1_multiclass', 'iou_mean'],
        
        # Training parameters based on your evaluation results
        'binary_stage': {
            'window_size': 2.0,
            'stride_size': 1.9,
            'epochs': 10,
            'batch_size': 16,
            'learning_rate': 2e-5,
        },
        'multiclass_stage': {
            'window_size': 0.3,
            'stride_size': 0.15,
            'epochs': 15,
            'batch_size': 32,
            'learning_rate': 1e-5,
        }
    }
    
    return pipeline_config
#!/usr/bin/env python3
"""
Hyperparameter Optimization for Wav2Vec2 Profanity Detection Model

This script provides multiple approaches for finding optimal hyperparameters:
1. Grid Search
2. Random Search  
3. Bayesian Optimization (using Optuna)
4. Custom Progressive Search

Usage:
    python hyperparameter_optimizer.py --method grid --trials 20
    python hyperparameter_optimizer.py --method random --trials 50
    python hyperparameter_optimizer.py --method bayesian --trials 100
"""

import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Config
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import os
import json
import argparse
from itertools import product
import random
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import your existing classes and functions
import sys
sys.path.append('./scripts')
from fine_tune_wav2vec2_sen_ham_CW_max_pooling import (
    Wav2Vec2ForSpeechClassification, 
    CustomTrainer,
    SafeSavingTrainer,
    calculate_class_weights,
    prepare_dataset,
    compute_metrics,
    collate_fn,
    label_map,
    num_labels
)

# Try to import Optuna for Bayesian optimization
try:
    import optuna
    # Remove the PyTorchLightningPruningCallback import as it's not needed
    OPTUNA_AVAILABLE = True
    print("✅ Optuna is available for Bayesian optimization")
except ImportError as e:
    OPTUNA_AVAILABLE = False
    print(f"⚠️ Optuna not available: {e}")
    print("Install with: pip install optuna")

class HyperparameterOptimizer:
    def __init__(self, csv_file, model_name, base_output_dir):
        self.csv_file = csv_file
        self.model_name = model_name
        self.base_output_dir = base_output_dir
        self.results = []
        
        # Load and prepare base dataset
        self.df = pd.read_csv(csv_file)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, return_attention_mask=True, do_normalize=True
        )
        
        # Create output directory
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)
        
    def get_hyperparameter_space(self):
        """Define the hyperparameter search space"""
        return {
            # Learning rate
            'learning_rate': [1e-6, 5e-6, 1e-5, 3e-5, 5e-5, 1e-4],
            
            # Batch sizes
            'per_device_train_batch_size': [4, 8, 16, 24, 32],
            'gradient_accumulation_steps': [1, 2, 4],
            
            # Training epochs and steps
            'num_train_epochs': [20, 50, 80, 100, 150],
            'max_steps': [5000, 10000, 15000, 20000],
            
            # Regularization
            'weight_decay': [0.0, 0.01, 0.05, 0.1],
            'warmup_ratio': [0.0, 0.05, 0.1, 0.15, 0.2],
            
            # Model architecture
            'pooling_mode': ['mean', 'max', 'min'],
            
            # Optimization
            'gradient_checkpointing': [True, False],
            'fp16': [True, False],
            
            # Early stopping
            'early_stopping_patience': [5, 10, 15, 20],
            
            # Evaluation
            'eval_steps': [250, 500, 1000],
            'save_steps': [250, 500, 1000],
            
            # Dropout (if configurable)
            'hidden_dropout_prob': [0.1, 0.2, 0.3, 0.4, 0.5],
            'attention_dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    
    def get_bayesian_space(self, trial):
        """Define hyperparameter space for Bayesian optimization"""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
            'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [4, 8, 16, 32]),
            'gradient_accumulation_steps': trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 4]),
            'num_train_epochs': trial.suggest_int('num_train_epochs', 20, 150),
            'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.2),
            'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.3),
            'pooling_mode': trial.suggest_categorical('pooling_mode', ['mean', 'max', 'min']),
            'gradient_checkpointing': trial.suggest_categorical('gradient_checkpointing', [True, False]),
            'early_stopping_patience': trial.suggest_int('early_stopping_patience', 5, 25),
            'eval_steps': trial.suggest_categorical('eval_steps', [250, 500, 1000]),
            'hidden_dropout_prob': trial.suggest_float('hidden_dropout_prob', 0.1, 0.5),
        }
    
    def train_with_hyperparams(self, hyperparams, trial_name):
        """Train model with given hyperparameters"""
        print(f"\n🔄 Training {trial_name}")
        print(f"Hyperparameters: {hyperparams}")
        
        try:
            # Prepare data
            train_df, val_df = train_test_split(
                self.df, test_size=0.2, random_state=42, stratify=self.df['label']
            )
            
            # Calculate class weights
            class_weights = calculate_class_weights(train_df)
            
            # Create model config
            config = Wav2Vec2Config.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                finetuning_task="audio-classification",
                hidden_dropout_prob=hyperparams.get('hidden_dropout_prob', 0.1),
                attention_dropout=hyperparams.get('attention_dropout', 0.1),
            )
            
            # Initialize model
            model = Wav2Vec2ForSpeechClassification.from_pretrained(
                self.model_name,
                config=config,
                pooling_mode=hyperparams.get('pooling_mode', 'mean')
            )
            
            # Prepare datasets
            train_dataset = prepare_dataset(train_df, self.feature_extractor)
            val_dataset = prepare_dataset(val_df, self.feature_extractor)
            
            # Training arguments
            output_dir = os.path.join(self.base_output_dir, trial_name)
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=hyperparams.get('num_train_epochs', 50),
                max_steps=hyperparams.get('max_steps', -1),
                per_device_train_batch_size=hyperparams.get('per_device_train_batch_size', 16),
                per_device_eval_batch_size=hyperparams.get('per_device_train_batch_size', 16),
                gradient_accumulation_steps=hyperparams.get('gradient_accumulation_steps', 1),
                learning_rate=hyperparams.get('learning_rate', 3e-5),
                weight_decay=hyperparams.get('weight_decay', 0.01),
                warmup_ratio=hyperparams.get('warmup_ratio', 0.1),
                
                # Evaluation and saving
                eval_strategy="steps",
                eval_steps=hyperparams.get('eval_steps', 500),
                save_strategy="steps",
                save_steps=hyperparams.get('save_steps', 500),
                logging_steps=50,
                
                # Performance settings
                fp16=hyperparams.get('fp16', True),
                gradient_checkpointing=hyperparams.get('gradient_checkpointing', True),
                dataloader_pin_memory=True,
                
                # Model selection
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                save_total_limit=1,
                
                # Other settings
                remove_unused_columns=True,
                disable_tqdm=True,  # Reduce output noise
                report_to=None,  # Disable wandb/tensorboard
            )
            
            # Initialize trainer
            trainer = SafeSavingTrainer(
                class_weights=class_weights,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                data_collator=collate_fn,
            )
            
            # Add early stopping
            from transformers import EarlyStoppingCallback
            trainer.add_callback(EarlyStoppingCallback(
                early_stopping_patience=hyperparams.get('early_stopping_patience', 10)
            ))
            
            # Train the model
            trainer.train()
            
            # Get final evaluation
            eval_results = trainer.evaluate()
            final_accuracy = eval_results.get('eval_accuracy', 0.0)
            
            # Clean up to save memory
            del model, trainer, train_dataset, val_dataset
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return {
                'accuracy': final_accuracy,
                'hyperparams': hyperparams,
                'trial_name': trial_name,
                'eval_results': eval_results
            }
            
        except Exception as e:
            print(f"❌ Error in {trial_name}: {str(e)}")
            return {
                'accuracy': 0.0,
                'hyperparams': hyperparams,
                'trial_name': trial_name,
                'error': str(e)
            }
    
    def grid_search(self, max_trials=50):
        """Perform grid search optimization"""
        print("🔍 Starting Grid Search Optimization")
        
        space = self.get_hyperparameter_space()
        
        # Generate all combinations (limit to max_trials)
        keys = list(space.keys())
        combinations = list(product(*[space[key] for key in keys]))
        
        # Shuffle and limit
        random.shuffle(combinations)
        combinations = combinations[:max_trials]
        
        results = []
        for i, combo in enumerate(combinations):
            hyperparams = dict(zip(keys, combo))
            trial_name = f"grid_trial_{i+1:03d}"
            
            result = self.train_with_hyperparams(hyperparams, trial_name)
            results.append(result)
            
            print(f"Trial {i+1}/{len(combinations)}: Accuracy = {result['accuracy']:.4f}")
            
            # Save intermediate results
            self.save_results(results, 'grid_search_results.json')
        
        return results
    
    def random_search(self, max_trials=50):
        """Perform random search optimization"""
        print("🎲 Starting Random Search Optimization")
        
        space = self.get_hyperparameter_space()
        results = []
        
        for i in range(max_trials):
            # Randomly sample hyperparameters
            hyperparams = {}
            for key, values in space.items():
                hyperparams[key] = random.choice(values)
            
            trial_name = f"random_trial_{i+1:03d}"
            result = self.train_with_hyperparams(hyperparams, trial_name)
            results.append(result)
            
            print(f"Trial {i+1}/{max_trials}: Accuracy = {result['accuracy']:.4f}")
            
            # Save intermediate results
            self.save_results(results, 'random_search_results.json')
        
        return results
    
    def bayesian_optimization(self, max_trials=100):
        """Perform Bayesian optimization using Optuna"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization. Install with: pip install optuna")
        
        print("🧠 Starting Bayesian Optimization with Optuna")
        
        def objective(trial):
            hyperparams = self.get_bayesian_space(trial)
            trial_name = f"bayesian_trial_{trial.number:03d}"
            
            result = self.train_with_hyperparams(hyperparams, trial_name)
            accuracy = result['accuracy']
            
            # Save result
            self.results.append(result)
            self.save_results(self.results, 'bayesian_optimization_results.json')
            
            return accuracy
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=max_trials)
        
        print(f"\n🎯 Best trial: {study.best_trial.number}")
        print(f"Best accuracy: {study.best_value:.4f}")
        print(f"Best hyperparameters: {study.best_trial.params}")
        
        return self.results
    
    def progressive_search(self, stages=3):
        """Custom progressive search: coarse -> fine -> ultra-fine"""
        print("📈 Starting Progressive Search")
        
        all_results = []
        
        # Stage 1: Coarse search
        print("\n🔍 Stage 1: Coarse Search")
        coarse_space = {
            'learning_rate': [1e-5, 3e-5, 1e-4],
            'per_device_train_batch_size': [8, 16, 32],
            'num_train_epochs': [20, 50, 100],
            'weight_decay': [0.0, 0.01, 0.1],
            'pooling_mode': ['mean', 'max'],
        }
        
        stage1_results = self._search_stage(coarse_space, "coarse", max_trials=15)
        all_results.extend(stage1_results)
        
        # Find best from stage 1
        best_stage1 = max(stage1_results, key=lambda x: x['accuracy'])
        print(f"Best Stage 1 Accuracy: {best_stage1['accuracy']:.4f}")
        
        # Stage 2: Fine search around best result
        print("\n🔍 Stage 2: Fine Search")
        best_params = best_stage1['hyperparams']
        fine_space = self._get_fine_space(best_params)
        
        stage2_results = self._search_stage(fine_space, "fine", max_trials=20)
        all_results.extend(stage2_results)
        
        # Find best from stage 2
        best_stage2 = max(stage2_results, key=lambda x: x['accuracy'])
        print(f"Best Stage 2 Accuracy: {best_stage2['accuracy']:.4f}")
        
        # Stage 3: Ultra-fine search
        if stages >= 3:
            print("\n🔍 Stage 3: Ultra-Fine Search")
            ultra_fine_space = self._get_ultra_fine_space(best_stage2['hyperparams'])
            
            stage3_results = self._search_stage(ultra_fine_space, "ultra_fine", max_trials=10)
            all_results.extend(stage3_results)
        
        return all_results
    
    def _search_stage(self, space, stage_name, max_trials):
        """Helper method for progressive search stages"""
        keys = list(space.keys())
        combinations = list(product(*[space[key] for key in keys]))
        random.shuffle(combinations)
        combinations = combinations[:max_trials]
        
        results = []
        for i, combo in enumerate(combinations):
            hyperparams = dict(zip(keys, combo))
            trial_name = f"{stage_name}_trial_{i+1:03d}"
            
            result = self.train_with_hyperparams(hyperparams, trial_name)
            results.append(result)
            
            print(f"  {stage_name.title()} Trial {i+1}/{len(combinations)}: Accuracy = {result['accuracy']:.4f}")
        
        return results
    
    def _get_fine_space(self, best_params):
        """Generate fine search space around best parameters"""
        fine_space = {}
        
        # Learning rate fine tuning
        lr = best_params['learning_rate']
        fine_space['learning_rate'] = [lr*0.5, lr*0.75, lr, lr*1.25, lr*1.5]
        
        # Batch size variations
        bs = best_params['per_device_train_batch_size']
        fine_space['per_device_train_batch_size'] = [max(4, bs-8), bs, min(32, bs+8)]
        
        # Epoch fine tuning
        epochs = best_params['num_train_epochs']
        fine_space['num_train_epochs'] = [max(20, epochs-30), epochs, min(150, epochs+30)]
        
        # Weight decay fine tuning
        wd = best_params['weight_decay']
        fine_space['weight_decay'] = [max(0.0, wd-0.02), wd, min(0.2, wd+0.02)]
        
        # Keep best pooling mode
        fine_space['pooling_mode'] = [best_params['pooling_mode']]
        
        # Add warmup ratio variations
        fine_space['warmup_ratio'] = [0.05, 0.1, 0.15]
        
        return fine_space
    
    def _get_ultra_fine_space(self, best_params):
        """Generate ultra-fine search space"""
        ultra_fine_space = {}
        
        # Very fine learning rate tuning
        lr = best_params['learning_rate']
        ultra_fine_space['learning_rate'] = [lr*0.9, lr*0.95, lr, lr*1.05, lr*1.1]
        
        # Keep other best parameters mostly fixed
        ultra_fine_space['per_device_train_batch_size'] = [best_params['per_device_train_batch_size']]
        ultra_fine_space['pooling_mode'] = [best_params['pooling_mode']]
        
        # Fine tune regularization
        wd = best_params['weight_decay']
        ultra_fine_space['weight_decay'] = [max(0.0, wd-0.01), wd, min(0.2, wd+0.01)]
        
        # Fine tune training length
        epochs = best_params['num_train_epochs']
        ultra_fine_space['num_train_epochs'] = [max(20, epochs-10), epochs, min(150, epochs+10)]
        
        return ultra_fine_space
    
    def save_results(self, results, filename):
        """Save results to JSON file"""
        filepath = os.path.join(self.base_output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def plot_results(self, results, method_name):
        """Plot optimization results"""
        if not results:
            return
        
        # Extract accuracies
        accuracies = [r['accuracy'] for r in results if 'accuracy' in r]
        trials = list(range(1, len(accuracies) + 1))
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Accuracy over trials
        ax1.plot(trials, accuracies, 'bo-', alpha=0.7)
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'{method_name} - Accuracy over Trials')
        ax1.grid(True, alpha=0.3)
        
        # Add best accuracy line
        best_acc = max(accuracies)
        ax1.axhline(y=best_acc, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_acc:.4f}')
        ax1.legend()
        
        # Plot 2: Accuracy distribution
        ax2.hist(accuracies, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Accuracy')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{method_name} - Accuracy Distribution')
        ax2.axvline(x=best_acc, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_acc:.4f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_output_dir, f'{method_name.lower()}_results.png'), dpi=300)
        plt.close()
        
        print(f"📊 Results plot saved: {method_name.lower()}_results.png")
    
    def analyze_results(self, results, method_name):
        """Analyze and summarize optimization results"""
        if not results:
            print("No results to analyze")
            return
        
        # Filter out failed trials
        valid_results = [r for r in results if 'accuracy' in r and r['accuracy'] > 0]
        
        if not valid_results:
            print("No valid results found")
            return
        
        # Find best result
        best_result = max(valid_results, key=lambda x: x['accuracy'])
        
        # Calculate statistics
        accuracies = [r['accuracy'] for r in valid_results]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"\n📊 {method_name} Results Analysis")
        print("="*50)
        print(f"Total trials: {len(results)}")
        print(f"Valid trials: {len(valid_results)}")
        print(f"Failed trials: {len(results) - len(valid_results)}")
        print(f"\nAccuracy Statistics:")
        print(f"  Best: {best_result['accuracy']:.4f}")
        print(f"  Mean: {mean_acc:.4f}")
        print(f"  Std:  {std_acc:.4f}")
        print(f"  Min:  {min(accuracies):.4f}")
        print(f"  Max:  {max(accuracies):.4f}")
        
        print(f"\n🏆 Best Hyperparameters:")
        for key, value in best_result['hyperparams'].items():
            print(f"  {key}: {value}")
        
        # Save best hyperparameters
        best_config_path = os.path.join(self.base_output_dir, f'best_hyperparams_{method_name.lower()}.json')
        with open(best_config_path, 'w') as f:
            json.dump(best_result['hyperparams'], f, indent=2)
        
        print(f"\n💾 Best hyperparameters saved to: {best_config_path}")
        
        return best_result

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for Wav2Vec2')
    parser.add_argument('--method', type=str, choices=['grid', 'random', 'bayesian', 'progressive'], 
                       default='random', help='Optimization method')
    parser.add_argument('--trials', type=int, default=30, help='Number of trials')
    parser.add_argument('--csv_file', type=str, default='./csv/balanced_train.csv', 
                       help='Training data CSV file')
    parser.add_argument('--model_name', type=str, default='airesearch/wav2vec2-large-xlsr-53-th',
                       help='Base model name')
    parser.add_argument('--output_dir', type=str, default='./hyperparameter_optimization',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Hyperparameter Optimization")
    print(f"Method: {args.method}")
    print(f"Max trials: {args.trials}")
    print(f"Data: {args.csv_file}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        csv_file=args.csv_file,
        model_name=args.model_name,
        base_output_dir=args.output_dir
    )
    
    # Run optimization
    start_time = datetime.now()
    
    if args.method == 'grid':
        results = optimizer.grid_search(max_trials=args.trials)
    elif args.method == 'random':
        results = optimizer.random_search(max_trials=args.trials)
    elif args.method == 'bayesian':
        results = optimizer.bayesian_optimization(max_trials=args.trials)
    elif args.method == 'progressive':
        results = optimizer.progressive_search()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n⏱️ Optimization completed in: {duration}")
    
    # Analyze and visualize results
    optimizer.analyze_results(results, args.method.title())
    optimizer.plot_results(results, args.method.title())
    
    print(f"\n✅ Hyperparameter optimization complete!")
    print(f"📁 Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()

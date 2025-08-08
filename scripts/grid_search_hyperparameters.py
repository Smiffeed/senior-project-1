import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Config
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from torch.nn.utils.rnn import pad_sequence
from transformers import EarlyStoppingCallback
import librosa
from sklearn.model_selection import KFold
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import json
import itertools
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import from the main training script
import sys
sys.path.append('.')
from fine_tune_wav2vec2_sen_ham_CW import (
    label_map, num_labels, calculate_class_weights, CustomTrainer, SafeSavingTrainer,
    compute_metrics, load_dataset, preprocess_audio, prepare_dataset, collate_fn
)

# Set environment variables
os.environ["TRANSFORMERS_VERBOSITY"] = "warning"
os.environ["DATASETS_VERBOSITY"] = "warning"

class GridSearchTrainer:
    def __init__(self, csv_file, model_name, base_output_dir):
        self.csv_file = csv_file
        self.model_name = model_name
        self.base_output_dir = base_output_dir
        self.results = []
        
        # Load and prepare data once
        print("Loading and preparing data...")
        self.df = load_dataset(csv_file)
        
        # Split into train/val/test
        train_val_df, self.test_df = train_test_split(
            self.df, test_size=0.15, random_state=42, stratify=self.df['label']
        )
        self.train_df, self.val_df = train_test_split(
            train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['label']
        )
        
        print(f"Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")
        
        # Prepare feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            return_attention_mask=True,
            do_normalize=True,
        )
        
        # Prepare datasets
        print("Preparing datasets...")
        self.train_dataset = prepare_dataset(self.train_df, self.feature_extractor)
        self.val_dataset = prepare_dataset(self.val_df, self.feature_extractor)
        self.test_dataset = prepare_dataset(self.test_df, self.feature_extractor)
        
        print(f"Dataset sizes - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        
        # Calculate class weights
        train_labels = [ex['label'] for ex in self.train_dataset]
        self.class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        self.class_weights = torch.FloatTensor(self.class_weights)
        
    def define_hyperparameter_grid(self):
        """Define the hyperparameter search space"""
        param_grid = {
            # Core training parameters
            'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
            'per_device_train_batch_size': [4, 8, 16],
            'gradient_accumulation_steps': [1, 2, 4],
            'num_train_epochs': [50, 100, 150],
            'max_steps': [2000, 3000, 5000],
            
            # Regularization parameters
            'weight_decay': [0.0, 0.01, 0.1],
            'warmup_ratio': [0.05, 0.1, 0.15],
            'attention_dropout': [0.1, 0.2, 0.3],
            'hidden_dropout': [0.3, 0.4, 0.5],
            
            # Training strategy parameters
            'save_steps': [250, 500, 1000],
            'eval_steps': [125, 250, 500],
            'early_stopping_patience': [10, 15, 20],
            
            # Advanced parameters
            'lr_scheduler_type': ['linear', 'cosine', 'cosine_with_restarts'],
            'fp16': [True, False],
            'gradient_checkpointing': [True, False],
            
            # Focal loss parameters
            'gamma': [2.0, 3.0, 4.0],
            'alpha': [0.5, 0.75, 0.9]
        }
        return param_grid
        
    def create_smart_grid(self, max_combinations=50):
        """Create a smart subset of hyperparameter combinations"""
        param_grid = self.define_hyperparameter_grid()
        
        # Define high-priority combinations based on your previous best results
        priority_combinations = [
            {
                'learning_rate': 3e-5,
                'per_device_train_batch_size': 4,
                'gradient_accumulation_steps': 2,
                'num_train_epochs': 150,
                'max_steps': 5000,
                'weight_decay': 0.0,
                'warmup_ratio': 0.05,
                'attention_dropout': 0.3,
                'hidden_dropout': 0.5,
                'save_steps': 500,
                'eval_steps': 250,
                'early_stopping_patience': 20,
                'lr_scheduler_type': 'cosine',
                'fp16': True,
                'gradient_checkpointing': False,
                'gamma': 3.0,
                'alpha': 0.75
            }
        ]
        
        # Create variations around the best known configuration
        variations = []
        base_config = priority_combinations[0]
        
        # Learning rate variations
        for lr in [1e-5, 2e-5, 3e-5, 5e-5]:
            config = base_config.copy()
            config['learning_rate'] = lr
            variations.append(config)
        
        # Batch size variations
        for bs in [4, 8, 16]:
            config = base_config.copy()
            config['per_device_train_batch_size'] = bs
            if bs > 4:
                config['gradient_accumulation_steps'] = max(1, config['gradient_accumulation_steps'] // 2)
            variations.append(config)
        
        # Dropout variations
        for att_drop, hid_drop in [(0.1, 0.3), (0.2, 0.4), (0.3, 0.5)]:
            config = base_config.copy()
            config['attention_dropout'] = att_drop
            config['hidden_dropout'] = hid_drop
            variations.append(config)
        
        # Scheduler variations
        for scheduler in ['linear', 'cosine', 'cosine_with_restarts']:
            config = base_config.copy()
            config['lr_scheduler_type'] = scheduler
            variations.append(config)
        
        # Focal loss variations
        for gamma, alpha in [(2.0, 0.5), (3.0, 0.75), (4.0, 0.9)]:
            config = base_config.copy()
            config['gamma'] = gamma
            config['alpha'] = alpha
            variations.append(config)
        
        # Weight decay variations
        for wd in [0.0, 0.01, 0.1]:
            config = base_config.copy()
            config['weight_decay'] = wd
            variations.append(config)
        
        # Remove duplicates
        unique_variations = []
        seen = set()
        for config in priority_combinations + variations:
            config_key = tuple(sorted(config.items()))
            if config_key not in seen:
                unique_variations.append(config)
                seen.add(config_key)
        
        return unique_variations[:max_combinations]
    
    def train_single_configuration(self, config, config_id):
        """Train model with a single hyperparameter configuration"""
        print(f"\n{'='*60}")
        print(f"Training Configuration {config_id + 1}")
        print(f"{'='*60}")
        print(f"Config: {json.dumps(config, indent=2)}")
        
        try:
            # Create output directory for this configuration
            output_dir = os.path.join(self.base_output_dir, f'config_{config_id + 1:03d}')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save configuration
            with open(os.path.join(output_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            
            # Create model configuration
            model_config = Wav2Vec2Config.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                finetuning_task="audio-classification",
                attention_dropout=config['attention_dropout'],
                hidden_dropout=config['hidden_dropout']
            )
            
            # Initialize model
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.model_name,
                config=model_config
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=config['num_train_epochs'],
                max_steps=config['max_steps'],
                per_device_train_batch_size=config['per_device_train_batch_size'],
                per_device_eval_batch_size=config['per_device_train_batch_size'],
                gradient_accumulation_steps=config['gradient_accumulation_steps'],
                learning_rate=config['learning_rate'],
                weight_decay=config['weight_decay'],
                warmup_ratio=config['warmup_ratio'],
                save_steps=config.get('save_steps', 500),
                eval_steps=config['eval_steps'],
                logging_steps=config.get('logging_steps', 50),
                save_strategy=config.get('save_strategy', "steps"),
                eval_strategy="steps",
                load_best_model_at_end=config.get('load_best_model_at_end', True),
                save_total_limit=config.get('save_total_limit', 3),
                metric_for_best_model="eval_f1_macro",
                greater_is_better=True,
                lr_scheduler_type=config['lr_scheduler_type'],
                fp16=config['fp16'],
                bf16=False,
                gradient_checkpointing=config['gradient_checkpointing'],
                dataloader_pin_memory=True,
                logging_dir=f"{output_dir}/logs",
                report_to=None,  # Disable wandb logging
                push_to_hub=False,
                remove_unused_columns=True,
                dataloader_num_workers=0,
                dataloader_drop_last=True,
                adam_epsilon=1e-6,
                max_grad_norm=1.0,
            )
            
            # Initialize trainer
            trainer = SafeSavingTrainer(
                class_weights=self.class_weights,
                use_focal_loss=True,
                gamma=config['gamma'],
                alpha=config['alpha'],
                model=model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=config['early_stopping_patience'])],
                data_collator=collate_fn,
            )
            
            # Train model
            print(f"Starting training for configuration {config_id + 1}...")
            train_result = trainer.train()
            
            # Evaluate on validation set
            print(f"Evaluating configuration {config_id + 1}...")
            val_results = trainer.evaluate()
            
            # Evaluate on test set
            test_results = self.evaluate_on_test_set(model, self.feature_extractor)
            
            # Save results
            result = {
                'config_id': config_id + 1,
                'config': config,
                'train_loss': train_result.training_loss,
                'val_accuracy': val_results.get('eval_accuracy', 0),
                'val_f1_macro': val_results.get('eval_f1_macro', 0),
                'val_f1_weighted': val_results.get('eval_f1_weighted', 0),
                'val_profanity_accuracy': val_results.get('eval_profanity_accuracy', 0),
                'val_profanity_f1': val_results.get('eval_profanity_f1', 0),
                'test_accuracy': test_results.get('accuracy', 0),
                'test_f1_macro': test_results.get('f1_macro', 0),
                'test_f1_weighted': test_results.get('f1_weighted', 0),
                'output_dir': output_dir,
                'training_time': train_result.metrics.get('train_runtime', 0),
                'total_steps': train_result.global_step,
            }
            
            # Save individual result
            with open(os.path.join(output_dir, 'results.json'), 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"Configuration {config_id + 1} Results:")
            print(f"  Val F1 Macro: {result['val_f1_macro']:.4f}")
            print(f"  Val Accuracy: {result['val_accuracy']:.4f}")
            print(f"  Test F1 Macro: {result['test_f1_macro']:.4f}")
            print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
            
            # Clean up model files to save disk space (keep only results.json and config.json)
            self.cleanup_model_files(output_dir)
            
            return result
            
        except Exception as e:
            print(f"Error training configuration {config_id + 1}: {e}")
            return {
                'config_id': config_id + 1,
                'config': config,
                'error': str(e),
                'val_f1_macro': 0,
                'val_accuracy': 0,
                'test_f1_macro': 0,
                'test_accuracy': 0,
            }
    
    def cleanup_model_files(self, output_dir):
        """Clean up model files to save disk space, keeping only results and config"""
        import shutil
        
        try:
            # List of files/folders to keep
            keep_files = {'results.json', 'config.json'}
            
            # Remove everything except the files we want to keep
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if item not in keep_files:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        
            print(f"  🧹 Cleaned up model files in {output_dir}")
                        
        except Exception as e:
            print(f"  ⚠️  Warning: Could not clean up {output_dir}: {e}")
    
    def evaluate_on_test_set(self, model, feature_extractor):
        """Evaluate model on test set"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for example in self.test_dataset:
                try:
                    input_values = torch.tensor(example['input_values']).unsqueeze(0).to(device)
                    outputs = model(input_values=input_values)
                    pred = torch.argmax(outputs.logits, dim=-1).cpu().item()
                    
                    predictions.append(pred)
                    true_labels.append(example['label'])
                    
                except Exception as e:
                    print(f"Error evaluating test example: {e}")
                    continue
        
        if len(predictions) == 0:
            return {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0}
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(true_labels, predictions)
        f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
    
    def run_grid_search(self, max_configurations=20):
        """Run the complete grid search"""
        print(f"Starting Grid Search with up to {max_configurations} configurations...")
        
        # Create smart grid
        configurations = self.create_smart_grid(max_configurations)
        print(f"Generated {len(configurations)} configurations to test")
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.base_output_dir, f'grid_search_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save all configurations
        with open(os.path.join(results_dir, 'all_configurations.json'), 'w') as f:
            json.dump(configurations, f, indent=2)
        
        # Train each configuration
        all_results = []
        for i, config in enumerate(configurations):
            result = self.train_single_configuration(config, i)
            all_results.append(result)
            
            # Save intermediate results
            with open(os.path.join(results_dir, 'intermediate_results.json'), 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Print progress
            print(f"\nProgress: {i + 1}/{len(configurations)} configurations completed")
            
        # Analyze and save final results
        final_results = self.analyze_results(all_results, results_dir)
        return final_results
    
    def analyze_results(self, results, results_dir):
        """Analyze and visualize grid search results"""
        print(f"\n{'='*60}")
        print("GRID SEARCH ANALYSIS")
        print(f"{'='*60}")
        
        # Filter out failed runs
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]
        
        print(f"Successful runs: {len(successful_results)}")
        print(f"Failed runs: {len(failed_results)}")
        
        if len(successful_results) == 0:
            print("No successful runs to analyze!")
            return results
        
        # Sort by validation F1 macro score
        successful_results.sort(key=lambda x: x['val_f1_macro'], reverse=True)
        
        # Print top 10 results
        print(f"\nTOP 10 CONFIGURATIONS:")
        print("-" * 100)
        for i, result in enumerate(successful_results[:10]):
            print(f"Rank {i+1:2d}: Config {result['config_id']:3d} | "
                  f"Val F1: {result['val_f1_macro']:.4f} | "
                  f"Val Acc: {result['val_accuracy']:.4f} | "
                  f"Test F1: {result['test_f1_macro']:.4f} | "
                  f"Test Acc: {result['test_accuracy']:.4f}")
        
        # Best configuration details
        best_config = successful_results[0]
        print(f"\n{'='*60}")
        print("BEST CONFIGURATION DETAILS:")
        print(f"{'='*60}")
        print(f"Configuration ID: {best_config['config_id']}")
        print(f"Validation F1 Macro: {best_config['val_f1_macro']:.4f}")
        print(f"Validation Accuracy: {best_config['val_accuracy']:.4f}")
        print(f"Test F1 Macro: {best_config['test_f1_macro']:.4f}")
        print(f"Test Accuracy: {best_config['test_accuracy']:.4f}")
        print(f"Training Time: {best_config.get('training_time', 0):.2f} seconds")
        print("\nHyperparameters:")
        for key, value in best_config['config'].items():
            print(f"  {key}: {value}")
        
        # Save detailed results
        with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
            json.dump(successful_results, f, indent=2)
        
        # Create analysis plots
        self.create_analysis_plots(successful_results, results_dir)
        
        # Generate hyperparameter importance analysis
        self.analyze_hyperparameter_importance(successful_results, results_dir)
        
        return {
            'best_config': best_config,
            'all_results': successful_results,
            'failed_results': failed_results,
            'results_dir': results_dir
        }
    
    def create_analysis_plots(self, results, results_dir):
        """Create visualization plots for the grid search results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('default')
            
            # 1. Performance comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Top 20 configurations
            top_results = results[:20]
            config_ids = [r['config_id'] for r in top_results]
            val_f1 = [r['val_f1_macro'] for r in top_results]
            test_f1 = [r['test_f1_macro'] for r in top_results]
            val_acc = [r['val_accuracy'] for r in top_results]
            test_acc = [r['test_accuracy'] for r in top_results]
            
            # F1 Scores
            axes[0, 0].bar(range(len(config_ids)), val_f1, alpha=0.7, label='Validation F1')
            axes[0, 0].bar(range(len(config_ids)), test_f1, alpha=0.7, label='Test F1')
            axes[0, 0].set_title('F1 Macro Scores (Top 20 Configs)')
            axes[0, 0].set_xlabel('Configuration Rank')
            axes[0, 0].set_ylabel('F1 Score')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy
            axes[0, 1].bar(range(len(config_ids)), val_acc, alpha=0.7, label='Validation Acc')
            axes[0, 1].bar(range(len(config_ids)), test_acc, alpha=0.7, label='Test Acc')
            axes[0, 1].set_title('Accuracy Scores (Top 20 Configs)')
            axes[0, 1].set_xlabel('Configuration Rank')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Validation vs Test F1 correlation
            axes[1, 0].scatter(val_f1, test_f1, alpha=0.6)
            axes[1, 0].plot([min(val_f1), max(val_f1)], [min(val_f1), max(val_f1)], 'r--', alpha=0.8)
            axes[1, 0].set_xlabel('Validation F1 Macro')
            axes[1, 0].set_ylabel('Test F1 Macro')
            axes[1, 0].set_title('Validation vs Test F1 Correlation')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Training time vs performance
            training_times = [r.get('training_time', 0) for r in top_results]
            if any(t > 0 for t in training_times):
                axes[1, 1].scatter(training_times, val_f1, alpha=0.6)
                axes[1, 1].set_xlabel('Training Time (seconds)')
                axes[1, 1].set_ylabel('Validation F1 Macro')
                axes[1, 1].set_title('Training Time vs Performance')
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Training time data not available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Analysis plots saved to {results_dir}")
            
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def analyze_hyperparameter_importance(self, results, results_dir):
        """Analyze the importance of different hyperparameters"""
        try:
            # Group results by hyperparameter values
            param_performance = {}
            
            for result in results:
                config = result['config']
                val_f1 = result['val_f1_macro']
                
                for param, value in config.items():
                    if param not in param_performance:
                        param_performance[param] = {}
                    if value not in param_performance[param]:
                        param_performance[param][value] = []
                    param_performance[param][value].append(val_f1)
            
            # Calculate average performance for each parameter value
            param_analysis = {}
            for param, values in param_performance.items():
                param_analysis[param] = {}
                for value, performances in values.items():
                    param_analysis[param][value] = {
                        'mean': np.mean(performances),
                        'std': np.std(performances),
                        'count': len(performances)
                    }
            
            # Save parameter analysis
            with open(os.path.join(results_dir, 'hyperparameter_analysis.json'), 'w') as f:
                json.dump(param_analysis, f, indent=2, default=str)
            
            # Print key insights
            print(f"\n{'='*60}")
            print("HYPERPARAMETER INSIGHTS:")
            print(f"{'='*60}")
            
            for param, values in param_analysis.items():
                if len(values) > 1:  # Only analyze parameters with multiple values
                    best_value = max(values.items(), key=lambda x: x[1]['mean'])
                    worst_value = min(values.items(), key=lambda x: x[1]['mean'])
                    
                    print(f"\n{param}:")
                    print(f"  Best value: {best_value[0]} (F1: {best_value[1]['mean']:.4f} ± {best_value[1]['std']:.4f})")
                    print(f"  Worst value: {worst_value[0]} (F1: {worst_value[1]['mean']:.4f} ± {worst_value[1]['std']:.4f})")
                    print(f"  Improvement: {(best_value[1]['mean'] - worst_value[1]['mean']):.4f}")
            
        except Exception as e:
            print(f"Error analyzing hyperparameter importance: {e}")


def main():
    # Configuration
    csv_file = './csv/train_with_augmentation.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    base_output_dir = './models/grid_search_results'
    
    # Create base directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Initialize grid search
    print("Initializing Grid Search...")
    grid_search = GridSearchTrainer(csv_file, model_name, base_output_dir)
    
    # Run grid search
    results = grid_search.run_grid_search(max_configurations=25)  # Adjust this number based on your resources
    
    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETED!")
    print(f"{'='*60}")
    print(f"Results saved in: {results['results_dir']}")
    print(f"Best configuration achieved Val F1: {results['best_config']['val_f1_macro']:.4f}")
    print(f"Best configuration achieved Test F1: {results['best_config']['test_f1_macro']:.4f}")


if __name__ == "__main__":
    main()

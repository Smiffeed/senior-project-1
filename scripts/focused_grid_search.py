import torch
import json
import numpy as np
from datetime import datetime
import os
import sys
sys.path.append('.')

# Import the main grid search class
from grid_search_hyperparameters import GridSearchTrainer

class FocusedGridSearch(GridSearchTrainer):
    """
    A focused grid search that tests small variations around known good configurations
    """
    
    def create_focused_grid(self, max_combinations=15):
        """Create a focused grid around your best known configuration"""
        
        # Your current best configuration
        base_config = {
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
        
        configurations = [base_config.copy()]  # Include base config
        
        # 1. Learning rate fine-tuning
        for lr in [2.5e-5, 3.5e-5, 4e-5]:
            config = base_config.copy()
            config['learning_rate'] = lr
            configurations.append(config)
        
        # 2. Batch size + gradient accumulation optimization
        batch_configs = [
            {'per_device_train_batch_size': 8, 'gradient_accumulation_steps': 1},
            {'per_device_train_batch_size': 2, 'gradient_accumulation_steps': 4},
            {'per_device_train_batch_size': 16, 'gradient_accumulation_steps': 1},
        ]
        for batch_config in batch_configs:
            config = base_config.copy()
            config.update(batch_config)
            configurations.append(config)
        
        # 3. Dropout fine-tuning
        dropout_configs = [
            {'attention_dropout': 0.25, 'hidden_dropout': 0.45},
            {'attention_dropout': 0.35, 'hidden_dropout': 0.55},
            {'attention_dropout': 0.2, 'hidden_dropout': 0.4},
        ]
        for dropout_config in dropout_configs:
            config = base_config.copy()
            config.update(dropout_config)
            configurations.append(config)
        
        # 4. Training length optimization
        length_configs = [
            {'num_train_epochs': 120, 'max_steps': 4000},
            {'num_train_epochs': 180, 'max_steps': 6000},
            {'num_train_epochs': 100, 'max_steps': 3500},
        ]
        for length_config in length_configs:
            config = base_config.copy()
            config.update(length_config)
            configurations.append(config)
        
        # 5. Warmup ratio variations
        for warmup in [0.03, 0.07, 0.1]:
            config = base_config.copy()
            config['warmup_ratio'] = warmup
            configurations.append(config)
        
        # 6. Focal loss fine-tuning
        focal_configs = [
            {'gamma': 2.5, 'alpha': 0.7},
            {'gamma': 3.5, 'alpha': 0.8},
            {'gamma': 2.0, 'alpha': 0.6},
        ]
        for focal_config in focal_configs:
            config = base_config.copy()
            config.update(focal_config)
            configurations.append(config)
        
        return configurations[:max_combinations]
    
    def run_focused_search(self, max_configurations=15):
        """Run focused grid search"""
        print(f"Starting Focused Grid Search with {max_configurations} configurations...")
        
        # Create focused grid
        configurations = self.create_focused_grid(max_configurations)
        print(f"Testing {len(configurations)} focused configurations")
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.base_output_dir, f'focused_search_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save all configurations
        with open(os.path.join(results_dir, 'focused_configurations.json'), 'w') as f:
            json.dump(configurations, f, indent=2)
        
        # Train each configuration
        all_results = []
        for i, config in enumerate(configurations):
            print(f"\n{'='*80}")
            print(f"FOCUSED SEARCH: Configuration {i+1}/{len(configurations)}")
            print(f"{'='*80}")
            
            result = self.train_single_configuration(config, i)
            all_results.append(result)
            
            # Save intermediate results
            with open(os.path.join(results_dir, 'focused_intermediate_results.json'), 'w') as f:
                json.dump(all_results, f, indent=2)
        
        # Analyze results
        final_results = self.analyze_results(all_results, results_dir)
        
        # Additional focused analysis
        self.compare_with_baseline(final_results, results_dir)
        
        return final_results
    
    def compare_with_baseline(self, results, results_dir):
        """Compare results with the baseline configuration"""
        successful_results = results['all_results']
        baseline_result = successful_results[0]  # First config is the baseline
        
        print(f"\n{'='*60}")
        print("COMPARISON WITH BASELINE:")
        print(f"{'='*60}")
        print(f"Baseline (Config 1): Val F1 = {baseline_result['val_f1_macro']:.4f}")
        
        improvements = []
        for result in successful_results[1:]:
            improvement = result['val_f1_macro'] - baseline_result['val_f1_macro']
            if improvement > 0:
                improvements.append((result, improvement))
        
        if improvements:
            improvements.sort(key=lambda x: x[1], reverse=True)
            print(f"\nConfigurations better than baseline:")
            for result, improvement in improvements:
                print(f"  Config {result['config_id']}: +{improvement:.4f} F1 improvement")
                print(f"    Val F1: {result['val_f1_macro']:.4f}, Test F1: {result['test_f1_macro']:.4f}")
        else:
            print("No configurations improved upon the baseline.")
        
        # Save comparison
        comparison_data = {
            'baseline': baseline_result,
            'improvements': [{'result': r, 'improvement': i} for r, i in improvements]
        }
        with open(os.path.join(results_dir, 'baseline_comparison.json'), 'w') as f:
            json.dump(comparison_data, f, indent=2)


def main():
    # Configuration
    csv_file = './csv/train_with_augmentation.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    base_output_dir = './models/focused_search_results'
    
    # Create base directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Initialize focused search
    print("Initializing Focused Grid Search...")
    focused_search = FocusedGridSearch(csv_file, model_name, base_output_dir)
    
    # Run focused search
    results = focused_search.run_focused_search(max_configurations=15)
    
    print(f"\n{'='*60}")
    print("FOCUSED GRID SEARCH COMPLETED!")
    print(f"{'='*60}")
    print(f"Results saved in: {results['results_dir']}")
    print(f"Best configuration achieved Val F1: {results['best_config']['val_f1_macro']:.4f}")
    print(f"Best configuration achieved Test F1: {results['best_config']['test_f1_macro']:.4f}")
    
    # Print the best configuration
    print(f"\nBEST HYPERPARAMETERS:")
    for key, value in results['best_config']['config'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

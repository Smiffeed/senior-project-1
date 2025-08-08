import torch
import json
import numpy as np
from datetime import datetime
import os
import sys
sys.path.append('.')

# Import the main grid search class
from grid_search_hyperparameters import GridSearchTrainer

class QuickGridSearch(GridSearchTrainer):
    """
    A quick grid search for testing just the most important hyperparameters
    """
    
    def create_quick_grid(self):
        """Create a small grid with only the most important parameters"""
        
        # Base configuration (optimized for minimal disk usage)
        base_config = {
            'learning_rate': 3e-5,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 2,
            'num_train_epochs': 50,   # Reduced further to save space and time
            'max_steps': 1500,        # Reduced further to save space and time
            'weight_decay': 0.0,
            'warmup_ratio': 0.05,
            'attention_dropout': 0.3,
            'hidden_dropout': 0.5,
            'save_steps': 10000,      # Save much less frequently (only at end)
            'eval_steps': 300,        # Evaluate less frequently
            'early_stopping_patience': 10,  # Stop earlier to save time/space
            'lr_scheduler_type': 'cosine',
            'fp16': True,
            'gradient_checkpointing': False,
            'gamma': 3.0,
            'alpha': 0.75,
            'save_strategy': 'no',    # Don't save intermediate checkpoints
            'logging_steps': 100,     # Log less frequently
            'save_total_limit': 1,    # Keep only 1 checkpoint
            'load_best_model_at_end': False,  # Don't keep best model to save space
        }
        
        configurations = []
        
        # Reduce the number of configurations to save space and time
        # Test only the most critical parameters
        
        # Test 3 learning rates instead of 4
        learning_rates = [2e-5, 3e-5, 5e-5]
        for lr in learning_rates:
            config = base_config.copy()
            config['learning_rate'] = lr
            configurations.append(config)
        
        # Test only 2 batch sizes instead of 3
        batch_sizes = [4, 8]
        for bs in batch_sizes:
            config = base_config.copy()
            config['per_device_train_batch_size'] = bs
            # Adjust gradient accumulation to keep effective batch size similar
            if bs == 4:
                config['gradient_accumulation_steps'] = 2
            else:  # bs == 8
                config['gradient_accumulation_steps'] = 1
            configurations.append(config)
        
        # Test only 2 dropout combinations instead of 3
        dropout_configs = [
            {'attention_dropout': 0.2, 'hidden_dropout': 0.4},
            {'attention_dropout': 0.4, 'hidden_dropout': 0.6},
        ]
        for dropout_config in dropout_configs:
            config = base_config.copy()
            config.update(dropout_config)
            configurations.append(config)
        
        # Test only 2 focal loss settings instead of 3
        focal_configs = [
            {'gamma': 2.0, 'alpha': 0.6},
            {'gamma': 4.0, 'alpha': 0.8},
        ]
        for focal_config in focal_configs:
            config = base_config.copy()
            config.update(focal_config)
            configurations.append(config)
        
        return configurations
    
    def run_quick_search(self):
        """Run quick grid search"""
        print("Starting Quick Grid Search...")
        
        # Create quick grid
        configurations = self.create_quick_grid()
        print(f"Testing {len(configurations)} configurations (reduced training time)")
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.base_output_dir, f'quick_search_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save all configurations
        with open(os.path.join(results_dir, 'quick_configurations.json'), 'w') as f:
            json.dump(configurations, f, indent=2)
        
        # Train each configuration
        all_results = []
        for i, config in enumerate(configurations):
            print(f"\n{'='*60}")
            print(f"QUICK SEARCH: Configuration {i+1}/{len(configurations)}")
            print(f"Key params: LR={config['learning_rate']}, BS={config['per_device_train_batch_size']}, "
                  f"Attn_Drop={config['attention_dropout']}, Gamma={config['gamma']}")
            print(f"{'='*60}")
            
            result = self.train_single_configuration(config, i)
            all_results.append(result)
            
            # Save intermediate results
            with open(os.path.join(results_dir, 'quick_intermediate_results.json'), 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Quick progress report
            if 'error' not in result:
                print(f"✅ Config {i+1} completed: Val F1 = {result['val_f1_macro']:.4f}")
            else:
                print(f"❌ Config {i+1} failed: {result.get('error', 'Unknown error')}")
        
        # Analyze results
        final_results = self.analyze_results(all_results, results_dir)
        
        # Quick summary
        self.print_quick_summary(final_results)
        
        return final_results
    
    def print_quick_summary(self, results):
        """Print a quick summary of the most important findings"""
        successful_results = results['all_results']
        
        if not successful_results:
            print("No successful configurations!")
            return
        
        print(f"\n{'='*60}")
        print("QUICK SEARCH SUMMARY:")
        print(f"{'='*60}")
        
        # Find best by different criteria
        best_val_f1 = max(successful_results, key=lambda x: x['val_f1_macro'])
        best_test_f1 = max(successful_results, key=lambda x: x['test_f1_macro'])
        
        print(f"🏆 Best Validation F1: {best_val_f1['val_f1_macro']:.4f}")
        print(f"   Learning Rate: {best_val_f1['config']['learning_rate']}")
        print(f"   Batch Size: {best_val_f1['config']['per_device_train_batch_size']}")
        print(f"   Attention Dropout: {best_val_f1['config']['attention_dropout']}")
        print(f"   Gamma: {best_val_f1['config']['gamma']}")
        
        print(f"\n🎯 Best Test F1: {best_test_f1['test_f1_macro']:.4f}")
        print(f"   Learning Rate: {best_test_f1['config']['learning_rate']}")
        print(f"   Batch Size: {best_test_f1['config']['per_device_train_batch_size']}")
        print(f"   Attention Dropout: {best_test_f1['config']['attention_dropout']}")
        print(f"   Gamma: {best_test_f1['config']['gamma']}")
        
        # Parameter analysis
        print(f"\n📊 PARAMETER INSIGHTS:")
        
        # Learning rate analysis
        lr_results = {}
        for result in successful_results:
            lr = result['config']['learning_rate']
            if lr not in lr_results:
                lr_results[lr] = []
            lr_results[lr].append(result['val_f1_macro'])
        
        print("Learning Rate Performance:")
        for lr, f1_scores in sorted(lr_results.items()):
            avg_f1 = np.mean(f1_scores)
            print(f"  {lr}: {avg_f1:.4f} avg F1 ({len(f1_scores)} configs)")
        
        # Batch size analysis
        bs_results = {}
        for result in successful_results:
            bs = result['config']['per_device_train_batch_size']
            if bs not in bs_results:
                bs_results[bs] = []
            bs_results[bs].append(result['val_f1_macro'])
        
        print("Batch Size Performance:")
        for bs, f1_scores in sorted(bs_results.items()):
            avg_f1 = np.mean(f1_scores)
            print(f"  {bs}: {avg_f1:.4f} avg F1 ({len(f1_scores)} configs)")


def main():
    # Configuration
    csv_file = './csv/train_with_augmentation.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    base_output_dir = './models/quick_search_results'
    
    # Create base directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Initialize quick search
    print("Initializing Quick Grid Search...")
    print("This will test key hyperparameters with reduced training time for faster results.")
    
    quick_search = QuickGridSearch(csv_file, model_name, base_output_dir)
    
    # Run quick search
    results = quick_search.run_quick_search()
    
    print(f"\n{'='*60}")
    print("QUICK GRID SEARCH COMPLETED!")
    print(f"{'='*60}")
    print(f"Results saved in: {results['results_dir']}")
    print(f"Best configuration: Val F1 = {results['best_config']['val_f1_macro']:.4f}")
    
    # Save the best config for easy copy-paste
    best_config_file = os.path.join(results['results_dir'], 'best_config_for_main_training.json')
    
    # Scale back up the training parameters for full training
    full_training_config = results['best_config']['config'].copy()
    full_training_config['num_train_epochs'] = 150
    full_training_config['max_steps'] = 5000
    full_training_config['save_steps'] = 500
    full_training_config['eval_steps'] = 250
    full_training_config['early_stopping_patience'] = 20
    
    with open(best_config_file, 'w') as f:
        json.dump(full_training_config, f, indent=2)
    
    print(f"📄 Best config for full training saved to: {best_config_file}")
    print("You can now use these hyperparameters for your full training run!")


if __name__ == "__main__":
    main()

import torch
import json
import numpy as np
from datetime import datetime
import os
import sys
import shutil
sys.path.append('.')

# Import the main grid search class
from grid_search_hyperparameters import GridSearchTrainer

class MinimalGridSearch(GridSearchTrainer):
    """
    Ultra-minimal grid search for systems with limited disk space
    """
    
    def create_minimal_grid(self):
        """Create the smallest possible grid for disk-constrained systems"""
        
        # Base configuration optimized for minimal resource usage
        base_config = {
            'learning_rate': 3e-5,
            'per_device_train_batch_size': 8,  # Larger batch to train faster
            'gradient_accumulation_steps': 1,   # Less accumulation
            'num_train_epochs': 30,            # Very short training
            'max_steps': 800,                  # Very few steps
            'weight_decay': 0.0,
            'warmup_ratio': 0.05,
            'attention_dropout': 0.3,
            'hidden_dropout': 0.5,
            'save_steps': 10000,               # Never save during training
            'eval_steps': 400,                 # Evaluate only twice
            'early_stopping_patience': 5,     # Stop very early
            'lr_scheduler_type': 'cosine',
            'fp16': True,
            'gradient_checkpointing': False,
            'gamma': 3.0,
            'alpha': 0.75,
            'save_strategy': 'no',             # Never save checkpoints
            'logging_steps': 200,              # Log very little
            'save_total_limit': 1,
            'load_best_model_at_end': False,   # Don't keep model in memory
        }
        
        # Test only 4 critical configurations
        configurations = []
        
        # Config 1: Current best (baseline)
        configurations.append(base_config.copy())
        
        # Config 2: Higher learning rate
        config2 = base_config.copy()
        config2['learning_rate'] = 5e-5
        configurations.append(config2)
        
        # Config 3: Different dropout
        config3 = base_config.copy()
        config3['attention_dropout'] = 0.2
        config3['hidden_dropout'] = 0.4
        configurations.append(config3)
        
        # Config 4: Different focal loss
        config4 = base_config.copy()
        config4['gamma'] = 2.0
        config4['alpha'] = 0.6
        configurations.append(config4)
        
        return configurations
    
    def run_minimal_search(self):
        """Run ultra-minimal grid search with aggressive disk cleanup"""
        print("Starting Minimal Grid Search (Disk-Space Optimized)...")
        print("⚠️  This will use minimal disk space and delete models after each config")
        
        # Create configurations
        configurations = self.create_minimal_grid()
        print(f"Testing only {len(configurations)} configurations")
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.base_output_dir, f'minimal_search_{timestamp}')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save configurations
        with open(os.path.join(results_dir, 'minimal_configurations.json'), 'w') as f:
            json.dump(configurations, f, indent=2)
        
        # Train each configuration
        all_results = []
        for i, config in enumerate(configurations):
            print(f"\n{'='*50}")
            print(f"MINIMAL SEARCH: Config {i+1}/{len(configurations)}")
            print(f"LR={config['learning_rate']}, BS={config['per_device_train_batch_size']}")
            print(f"Steps={config['max_steps']}, Dropout={config['attention_dropout']}")
            print(f"{'='*50}")
            
            # Train the configuration
            result = self.train_single_configuration(config, i)
            all_results.append(result)
            
            # Immediately clean up everything to save space
            output_dir = result.get('output_dir')
            if output_dir and os.path.exists(output_dir):
                try:
                    # Keep only the results.json, delete everything else
                    results_file = os.path.join(output_dir, 'results.json')
                    config_file = os.path.join(output_dir, 'config.json')
                    
                    # Copy results to main directory
                    shutil.copy(results_file, os.path.join(results_dir, f'config_{i+1}_results.json'))
                    shutil.copy(config_file, os.path.join(results_dir, f'config_{i+1}_config.json'))
                    
                    # Delete the entire output directory
                    shutil.rmtree(output_dir)
                    print(f"  🗑️  Deleted {output_dir} to save space")
                except Exception as e:
                    print(f"  ⚠️  Could not delete {output_dir}: {e}")
            
            # Save progress
            with open(os.path.join(results_dir, 'minimal_progress.json'), 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Quick status
            if 'error' not in result:
                print(f"✅ Config {i+1}: Val F1 = {result['val_f1_macro']:.4f}")
            else:
                print(f"❌ Config {i+1}: FAILED")
        
        # Analyze results
        successful_results = [r for r in all_results if 'error' not in r]
        
        if successful_results:
            # Find best configuration
            best_config = max(successful_results, key=lambda x: x['val_f1_macro'])
            
            # Save final results
            final_results = {
                'timestamp': timestamp,
                'total_configs': len(configurations),
                'successful_configs': len(successful_results),
                'best_config': best_config,
                'all_results': successful_results,
                'results_dir': results_dir
            }
            
            with open(os.path.join(results_dir, 'minimal_final_results.json'), 'w') as f:
                json.dump(final_results, f, indent=2)
            
            # Print summary
            print(f"\n{'='*50}")
            print("MINIMAL SEARCH COMPLETED!")
            print(f"{'='*50}")
            print(f"🏆 Best Config: Val F1 = {best_config['val_f1_macro']:.4f}")
            print(f"   Learning Rate: {best_config['config']['learning_rate']}")
            print(f"   Batch Size: {best_config['config']['per_device_train_batch_size']}")
            print(f"   Attention Dropout: {best_config['config']['attention_dropout']}")
            print(f"   Gamma: {best_config['config']['gamma']}")
            print(f"📁 Results saved in: {results_dir}")
            print(f"💾 Disk usage minimized by deleting all model files")
            
            return final_results
        else:
            print("❌ No successful configurations!")
            return {'error': 'All configurations failed', 'results_dir': results_dir}


def main():
    # Configuration
    csv_file = './csv/train_with_augmentation.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    base_output_dir = './models/minimal_search_results'
    
    # Create base directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Check available disk space
    import psutil
    disk_usage = psutil.disk_usage('.')
    free_gb = disk_usage.free / (1024**3)
    print(f"💾 Available disk space: {free_gb:.1f} GB")
    
    if free_gb < 5:
        print("⚠️  WARNING: Less than 5GB free space!")
        print("This script will use minimal resources and clean up aggressively.")
    
    # Initialize minimal search
    print("Initializing Minimal Grid Search...")
    print("This uses the least disk space possible and deletes models immediately.")
    
    minimal_search = MinimalGridSearch(csv_file, model_name, base_output_dir)
    
    # Run minimal search
    results = minimal_search.run_minimal_search()
    
    if 'error' not in results:
        print(f"\n🎉 SUCCESS! Best configuration found with Val F1 = {results['best_config']['val_f1_macro']:.4f}")
        print("You can now apply these hyperparameters to your main training script.")
    else:
        print(f"\n❌ Search failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Experiment runner for advanced Thai profanity detection model.
This script allows you to easily run different experimental configurations
and compare their results.
"""

import os
import json
import argparse
import sys
from datetime import datetime
import shutil

# Add the scripts directory to the path
sys.path.append('./scripts')

from ultimate_model_training import main as train_main, ADVANCED_CONFIG
from comprehensive_evaluation import main as eval_main

class ExperimentRunner:
    """Manages and runs different experimental configurations."""
    
    def __init__(self, base_config_path='./config/advanced_training_config.json'):
        self.base_config_path = base_config_path
        self.experiments_dir = './experiments'
        os.makedirs(self.experiments_dir, exist_ok=True)
        
        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = json.load(f)
    
    def create_experiment_config(self, experiment_name, modifications):
        """Create a new experiment configuration with specified modifications."""
        # Deep copy base config
        experiment_config = json.loads(json.dumps(self.base_config))
        
        # Apply modifications
        for key_path, value in modifications.items():
            self._set_nested_dict_value(experiment_config, key_path, value)
        
        # Create experiment directory
        exp_dir = os.path.join(self.experiments_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save experiment config
        config_path = os.path.join(exp_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(experiment_config, f, indent=2)
        
        # Update output directory
        experiment_config['output_config']['base_output_dir'] = os.path.join(exp_dir, 'models')
        
        return experiment_config, exp_dir
    
    def _set_nested_dict_value(self, d, key_path, value):
        """Set a value in a nested dictionary using dot notation."""
        keys = key_path.split('.')
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = value
    
    def run_experiment(self, experiment_name, config, skip_training=False, skip_evaluation=False):
        """Run a complete experiment."""
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENT: {experiment_name}")
        print(f"{'='*60}")
        
        exp_dir = os.path.join(self.experiments_dir, experiment_name)
        
        # Save experiment metadata
        metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': config
        }
        
        with open(os.path.join(exp_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update global config for training
        global ADVANCED_CONFIG
        if 'advanced_features' in config:
            ADVANCED_CONFIG.update(config['advanced_features'])
        
        results = {}
        
        try:
            # Training phase
            if not skip_training:
                print(f"\n--- Training Phase ---")
                
                # Update training parameters
                if 'training_config' in config:
                    # You would update your training script's global variables here
                    pass
                
                # Run training (this would need to be adapted to use the config)
                # For now, we'll simulate this
                print("Training with configuration:")
                for section, values in config.get('advanced_features', {}).items():
                    print(f"  {section}: {values}")
                
                # In a real implementation, you would:
                # train_main()  # But you'd need to modify it to accept config
                print("Training completed (simulated)")
                
                results['training'] = {'status': 'completed', 'message': 'Training phase finished'}
            
            # Evaluation phase
            if not skip_evaluation:
                print(f"\n--- Evaluation Phase ---")
                
                # Run evaluation
                # eval_main()  # You'd need to modify this to use the experiment directory
                print("Evaluation completed (simulated)")
                
                results['evaluation'] = {'status': 'completed', 'message': 'Evaluation phase finished'}
            
            # Mark experiment as successful
            metadata['end_time'] = datetime.now().isoformat()
            metadata['status'] = 'completed'
            metadata['results'] = results
            
        except Exception as e:
            print(f"Experiment failed with error: {str(e)}")
            metadata['end_time'] = datetime.now().isoformat()
            metadata['status'] = 'failed'
            metadata['error'] = str(e)
            results['error'] = str(e)
        
        # Save final metadata
        with open(os.path.join(exp_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return results
    
    def run_ablation_study(self):
        """Run ablation study to understand contribution of each component."""
        print("\n🔬 RUNNING ABLATION STUDY")
        print("="*60)
        
        ablation_experiments = {
            'baseline': {
                'advanced_features.use_spectral_augmentation': False,
                'advanced_features.use_contrastive_learning': False,
                'advanced_features.use_adversarial_training': False,
                'advanced_features.use_ensemble': False,
                'advanced_features.use_progressive_training': False,
                'advanced_features.use_focal_loss': False
            },
            
            'with_spectral_aug': {
                'advanced_features.use_spectral_augmentation': True,
                'advanced_features.use_contrastive_learning': False,
                'advanced_features.use_adversarial_training': False,
                'advanced_features.use_ensemble': False,
                'advanced_features.use_progressive_training': False,
                'advanced_features.use_focal_loss': False
            },
            
            'with_contrastive': {
                'advanced_features.use_spectral_augmentation': True,
                'advanced_features.use_contrastive_learning': True,
                'advanced_features.use_adversarial_training': False,
                'advanced_features.use_ensemble': False,
                'advanced_features.use_progressive_training': False,
                'advanced_features.use_focal_loss': False
            },
            
            'with_adversarial': {
                'advanced_features.use_spectral_augmentation': True,
                'advanced_features.use_contrastive_learning': True,
                'advanced_features.use_adversarial_training': True,
                'advanced_features.use_ensemble': False,
                'advanced_features.use_progressive_training': False,
                'advanced_features.use_focal_loss': False
            },
            
            'with_focal_loss': {
                'advanced_features.use_spectral_augmentation': True,
                'advanced_features.use_contrastive_learning': True,
                'advanced_features.use_adversarial_training': True,
                'advanced_features.use_ensemble': False,
                'advanced_features.use_progressive_training': False,
                'advanced_features.use_focal_loss': True
            },
            
            'full_advanced': {
                'advanced_features.use_spectral_augmentation': True,
                'advanced_features.use_contrastive_learning': True,
                'advanced_features.use_adversarial_training': True,
                'advanced_features.use_ensemble': True,
                'advanced_features.use_progressive_training': True,
                'advanced_features.use_focal_loss': True
            }
        }
        
        results = {}
        
        for exp_name, modifications in ablation_experiments.items():
            print(f"\n--- Running {exp_name} ---")
            config, exp_dir = self.create_experiment_config(f"ablation_{exp_name}", modifications)
            result = self.run_experiment(f"ablation_{exp_name}", config)
            results[exp_name] = result
        
        # Save ablation study summary
        summary_path = os.path.join(self.experiments_dir, 'ablation_study_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Ablation study completed. Results saved to {summary_path}")
        return results
    
    def run_hyperparameter_search(self):
        """Run hyperparameter search experiments."""
        print("\n🎛️ RUNNING HYPERPARAMETER SEARCH")
        print("="*60)
        
        # Define hyperparameter grid
        hp_experiments = {
            'lr_high': {
                'training_config.learning_rate': 5e-5,
                'loss_weights.contrastive_weight': 0.1
            },
            'lr_low': {
                'training_config.learning_rate': 1e-5,
                'loss_weights.contrastive_weight': 0.1
            },
            'contrastive_high': {
                'training_config.learning_rate': 3e-5,
                'loss_weights.contrastive_weight': 0.2
            },
            'contrastive_low': {
                'training_config.learning_rate': 3e-5,
                'loss_weights.contrastive_weight': 0.05
            },
            'focal_gamma_high': {
                'loss_weights.focal_loss_gamma': 3.0,
                'loss_weights.focal_loss_alpha': 1.0
            },
            'focal_gamma_low': {
                'loss_weights.focal_loss_gamma': 1.0,
                'loss_weights.focal_loss_alpha': 1.0
            }
        }
        
        results = {}
        
        for exp_name, modifications in hp_experiments.items():
            print(f"\n--- Running hyperparameter experiment: {exp_name} ---")
            config, exp_dir = self.create_experiment_config(f"hp_{exp_name}", modifications)
            result = self.run_experiment(f"hp_{exp_name}", config)
            results[exp_name] = result
        
        # Save hyperparameter search summary
        summary_path = os.path.join(self.experiments_dir, 'hyperparameter_search_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Hyperparameter search completed. Results saved to {summary_path}")
        return results
    
    def list_experiments(self):
        """List all completed experiments."""
        print("\n📋 EXPERIMENT HISTORY")
        print("="*60)
        
        if not os.path.exists(self.experiments_dir):
            print("No experiments found.")
            return
        
        experiments = []
        for exp_name in os.listdir(self.experiments_dir):
            exp_path = os.path.join(self.experiments_dir, exp_name)
            if os.path.isdir(exp_path):
                metadata_path = os.path.join(exp_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    experiments.append({
                        'name': exp_name,
                        'status': metadata.get('status', 'unknown'),
                        'start_time': metadata.get('start_time', 'unknown'),
                        'end_time': metadata.get('end_time', 'unknown')
                    })
        
        if not experiments:
            print("No completed experiments found.")
            return
        
        # Sort by start time
        experiments.sort(key=lambda x: x['start_time'])
        
        print(f"{'Name':<30} {'Status':<15} {'Start Time':<20} {'End Time':<20}")
        print("-" * 85)
        
        for exp in experiments:
            start_time = exp['start_time'][:19] if exp['start_time'] != 'unknown' else 'unknown'
            end_time = exp['end_time'][:19] if exp['end_time'] != 'unknown' else 'unknown'
            print(f"{exp['name']:<30} {exp['status']:<15} {start_time:<20} {end_time:<20}")
    
    def compare_experiments(self, experiment_names):
        """Compare results from multiple experiments."""
        print("\n📊 EXPERIMENT COMPARISON")
        print("="*60)
        
        comparison_data = {}
        
        for exp_name in experiment_names:
            exp_path = os.path.join(self.experiments_dir, exp_name)
            metadata_path = os.path.join(exp_path, 'metadata.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                comparison_data[exp_name] = metadata
            else:
                print(f"⚠️ Experiment {exp_name} not found.")
        
        if not comparison_data:
            print("No valid experiments to compare.")
            return
        
        # Create comparison summary
        summary = {
            'experiments': list(comparison_data.keys()),
            'comparison_time': datetime.now().isoformat(),
            'details': comparison_data
        }
        
        # Save comparison
        comparison_path = os.path.join(self.experiments_dir, 'experiment_comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Comparison saved to {comparison_path}")
        
        # Print summary
        for exp_name, data in comparison_data.items():
            print(f"\n{exp_name}:")
            print(f"  Status: {data.get('status', 'unknown')}")
            if 'results' in data:
                print(f"  Results: {data['results']}")

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Run advanced Thai profanity detection experiments')
    parser.add_argument('--action', choices=['ablation', 'hyperparameter', 'custom', 'list', 'compare'], 
                       required=True, help='Type of experiment to run')
    parser.add_argument('--name', help='Name for custom experiment')
    parser.add_argument('--config-file', help='Custom configuration file')
    parser.add_argument('--modifications', help='JSON string of configuration modifications')
    parser.add_argument('--experiments', nargs='+', help='Experiment names to compare')
    parser.add_argument('--skip-training', action='store_true', help='Skip training phase')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip evaluation phase')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    if args.action == 'ablation':
        runner.run_ablation_study()
    
    elif args.action == 'hyperparameter':
        runner.run_hyperparameter_search()
    
    elif args.action == 'custom':
        if not args.name:
            print("❌ Custom experiments require a --name")
            return
        
        modifications = {}
        if args.modifications:
            try:
                modifications = json.loads(args.modifications)
            except json.JSONDecodeError:
                print("❌ Invalid JSON in --modifications")
                return
        
        config, exp_dir = runner.create_experiment_config(args.name, modifications)
        runner.run_experiment(args.name, config, args.skip_training, args.skip_evaluation)
    
    elif args.action == 'list':
        runner.list_experiments()
    
    elif args.action == 'compare':
        if not args.experiments:
            print("❌ Comparison requires --experiments")
            return
        runner.compare_experiments(args.experiments)

if __name__ == "__main__":
    main()

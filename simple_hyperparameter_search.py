#!/usr/bin/env python3
"""
Simple Hyperparameter Search for Wav2Vec2 Model

This script provides a lightweight approach to finding good hyperparameters
by testing the most impactful parameters first.

Usage:
    python simple_hyperparameter_search.py
    python simple_hyperparameter_search.py --quick
    python simple_hyperparameter_search.py --comprehensive
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Key hyperparameters to optimize (most impactful ones)
QUICK_SEARCH_SPACE = {
    'learning_rate': [1e-5, 3e-5, 5e-5],
    'batch_size': [8, 16, 24],
    'epochs': [30, 60, 100],
    'pooling_mode': ['mean', 'max'],
}

COMPREHENSIVE_SEARCH_SPACE = {
    'learning_rate': [1e-6, 5e-6, 1e-5, 3e-5, 5e-5, 8e-5],
    'batch_size': [4, 8, 16, 24, 32],
    'epochs': [20, 40, 60, 80, 100, 120],
    'weight_decay': [0.0, 0.01, 0.05, 0.1],
    'warmup_ratio': [0.0, 0.1, 0.2],
    'pooling_mode': ['mean', 'max', 'min'],
    'gradient_accumulation_steps': [1, 2, 4],
}

def create_training_script(hyperparams, trial_name, base_output_dir):
    """
    Create a training script with specific hyperparameters
    """
    script_content = f'''#!/usr/bin/env python3
"""
Auto-generated training script for hyperparameter trial: {trial_name}
Generated at: {datetime.now()}
"""

import sys
import os
sys.path.append('./scripts')

from fine_tune_wav2vec2_sen_ham_CW_max_pooling import *

def main():
    # Hyperparameters for this trial
    hyperparams = {json.dumps(hyperparams, indent=4)}
    
    print(f"🔄 Running trial: {trial_name}")
    print(f"Hyperparameters: {{hyperparams}}")
    
    # Configuration
    csv_file = './csv/balanced_train.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    output_dir = './{base_output_dir}/{trial_name}'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Train the model with these hyperparameters
        result = train_wav2vec2_model_with_hyperparams(
            csv_file=csv_file,
            model_name=model_name,
            output_dir=output_dir,
            hyperparams=hyperparams
        )
        
        # Save results
        with open(f'{{output_dir}}/results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ Trial {trial_name} completed successfully!")
        print(f"Accuracy: {{result.get('accuracy', 'N/A')}}")
        
        return result
        
    except Exception as e:
        print(f"❌ Trial {trial_name} failed: {{str(e)}}")
        error_result = {{
            'trial_name': '{trial_name}',
            'hyperparams': hyperparams,
            'error': str(e),
            'accuracy': 0.0
        }}
        
        with open(f'{{output_dir}}/error.json', 'w') as f:
            json.dump(error_result, f, indent=2)
        
        return error_result

def train_wav2vec2_model_with_hyperparams(csv_file, model_name, output_dir, hyperparams):
    """
    Modified training function that accepts hyperparameters
    """
    # Load dataset
    df = load_dataset(csv_file)
    
    # Handle small classes
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        if count < 2:
            row_to_duplicate = df[df['label'] == label].iloc[0]
            df = pd.concat([df, pd.DataFrame([row_to_duplicate])], ignore_index=True)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_df)
    
    # Load model components
    config = Wav2Vec2Config.from_pretrained(
        model_name,
        num_labels=num_labels,
        finetuning_task="audio-classification"
    )
    
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_name,
        config=config,
        pooling_mode=hyperparams.get('pooling_mode', 'mean')
    )
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        return_attention_mask=True,
        do_normalize=True,
    )
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_df, feature_extractor)
    val_dataset = prepare_dataset(val_df, feature_extractor)
    
    # Training arguments with hyperparameters
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hyperparams.get('epochs', 60),
        per_device_train_batch_size=hyperparams.get('batch_size', 16),
        per_device_eval_batch_size=hyperparams.get('batch_size', 16),
        gradient_accumulation_steps=hyperparams.get('gradient_accumulation_steps', 1),
        learning_rate=hyperparams.get('learning_rate', 3e-5),
        weight_decay=hyperparams.get('weight_decay', 0.01),
        warmup_ratio=hyperparams.get('warmup_ratio', 0.1),
        
        # Fixed settings for faster training
        save_strategy="steps",
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        logging_steps=100,
        
        # Performance settings
        fp16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        
        # Model selection
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        
        # Reduce noise
        disable_tqdm=True,
        report_to=None,
        remove_unused_columns=True,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    # Save model
    trainer.save_model(output_dir)
    feature_extractor.save_pretrained(output_dir)
    
    # Return results
    return {{
        'trial_name': '{trial_name}',
        'hyperparams': hyperparams,
        'accuracy': eval_results.get('eval_accuracy', 0.0),
        'eval_results': eval_results,
        'model_path': output_dir
    }}

if __name__ == "__main__":
    main()
'''
    
    return script_content

def run_hyperparameter_search(search_space, search_type, max_trials=None):
    """
    Run hyperparameter search
    """
    print(f"🔍 Starting {search_type} hyperparameter search")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"hyperparameter_search_{search_type}_{timestamp}"
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate all parameter combinations
    from itertools import product
    
    param_names = list(search_space.keys())
    param_values = list(search_space.values())
    all_combinations = list(product(*param_values))
    
    # Limit trials if specified
    if max_trials and max_trials < len(all_combinations):
        import random
        random.shuffle(all_combinations)
        all_combinations = all_combinations[:max_trials]
    
    print(f"Total combinations to test: {len(all_combinations)}")
    
    results = []
    
    for i, combination in enumerate(all_combinations):
        # Create hyperparameter dict
        hyperparams = dict(zip(param_names, combination))
        trial_name = f"trial_{i+1:03d}"
        
        print(f"\n📊 Trial {i+1}/{len(all_combinations)}: {trial_name}")
        print(f"Parameters: {hyperparams}")
        
        # Create and run training script
        script_content = create_training_script(hyperparams, trial_name, base_output_dir)
        script_path = f"{base_output_dir}/{trial_name}_train.py"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Execute the script
        try:
            import subprocess
            import sys
            
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                # Load results
                result_file = f"{base_output_dir}/{trial_name}/results.json"
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        trial_result = json.load(f)
                    print(f"✅ Accuracy: {trial_result.get('accuracy', 0):.4f}")
                else:
                    trial_result = {'error': 'No results file found', 'accuracy': 0.0}
                    print("⚠️ No results file found")
            else:
                trial_result = {'error': result.stderr, 'accuracy': 0.0}
                print(f"❌ Training failed: {result.stderr[:200]}")
            
            trial_result['hyperparams'] = hyperparams
            trial_result['trial_name'] = trial_name
            results.append(trial_result)
            
        except subprocess.TimeoutExpired:
            print("⏱️ Training timed out")
            results.append({
                'hyperparams': hyperparams,
                'trial_name': trial_name,
                'error': 'Timeout',
                'accuracy': 0.0
            })
        except Exception as e:
            print(f"❌ Error running trial: {e}")
            results.append({
                'hyperparams': hyperparams,
                'trial_name': trial_name,
                'error': str(e),
                'accuracy': 0.0
            })
        
        # Save intermediate results
        with open(f"{base_output_dir}/all_results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    # Analyze results
    analyze_and_visualize_results(results, base_output_dir, search_type)
    
    return results, base_output_dir

def analyze_and_visualize_results(results, output_dir, search_type):
    """
    Analyze results and create visualizations
    """
    print(f"\n📊 Analyzing {search_type} search results...")
    
    # Filter valid results
    valid_results = [r for r in results if r.get('accuracy', 0) > 0]
    
    if not valid_results:
        print("❌ No valid results found!")
        return
    
    # Find best result
    best_result = max(valid_results, key=lambda x: x['accuracy'])
    
    # Calculate statistics
    accuracies = [r['accuracy'] for r in valid_results]
    
    print(f"\n🏆 Best Result:")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  Trial: {best_result['trial_name']}")
    print(f"  Hyperparameters:")
    for key, value in best_result['hyperparams'].items():
        print(f"    {key}: {value}")
    
    print(f"\n📈 Statistics:")
    print(f"  Total trials: {len(results)}")
    print(f"  Successful trials: {len(valid_results)}")
    print(f"  Success rate: {len(valid_results)/len(results)*100:.1f}%")
    print(f"  Mean accuracy: {np.mean(accuracies):.4f}")
    print(f"  Std accuracy: {np.std(accuracies):.4f}")
    
    # Save best hyperparameters
    best_config = {
        'best_hyperparams': best_result['hyperparams'],
        'best_accuracy': best_result['accuracy'],
        'search_type': search_type,
        'statistics': {
            'total_trials': len(results),
            'successful_trials': len(valid_results),
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
        }
    }
    
    with open(f"{output_dir}/best_hyperparameters.json", 'w') as f:
        json.dump(best_config, f, indent=2)
    
    # Create visualizations
    create_visualizations(valid_results, output_dir, search_type)
    
    print(f"\n💾 Results saved in: {output_dir}")
    print(f"📊 Best hyperparameters: {output_dir}/best_hyperparameters.json")

def create_visualizations(results, output_dir, search_type):
    """
    Create result visualizations
    """
    if not results:
        return
    
    # Extract data
    trial_numbers = list(range(1, len(results) + 1))
    accuracies = [r['accuracy'] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{search_type.title()} Hyperparameter Search Results', fontsize=16)
    
    # Plot 1: Accuracy over trials
    axes[0, 0].plot(trial_numbers, accuracies, 'bo-', alpha=0.7)
    axes[0, 0].set_xlabel('Trial Number')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Over Trials')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Highlight best result
    best_idx = np.argmax(accuracies)
    axes[0, 0].plot(trial_numbers[best_idx], accuracies[best_idx], 'ro', markersize=10, label=f'Best: {accuracies[best_idx]:.4f}')
    axes[0, 0].legend()
    
    # Plot 2: Accuracy distribution
    axes[0, 1].hist(accuracies, bins=15, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Accuracy')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Accuracy Distribution')
    axes[0, 1].axvline(x=np.mean(accuracies), color='r', linestyle='--', label=f'Mean: {np.mean(accuracies):.4f}')
    axes[0, 1].legend()
    
    # Plot 3: Learning rate vs Accuracy (if learning_rate is in hyperparams)
    if 'learning_rate' in results[0]['hyperparams']:
        learning_rates = [r['hyperparams']['learning_rate'] for r in results]
        axes[1, 0].scatter(learning_rates, accuracies, alpha=0.7)
        axes[1, 0].set_xlabel('Learning Rate')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Learning Rate vs Accuracy')
        axes[1, 0].set_xscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Batch size vs Accuracy (if batch_size is in hyperparams)
    if 'batch_size' in results[0]['hyperparams']:
        batch_sizes = [r['hyperparams']['batch_size'] for r in results]
        axes[1, 1].scatter(batch_sizes, accuracies, alpha=0.7)
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Batch Size vs Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hyperparameter_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Visualization saved: {output_dir}/hyperparameter_analysis.png")

def main():
    parser = argparse.ArgumentParser(description='Simple Hyperparameter Search')
    parser.add_argument('--mode', choices=['quick', 'comprehensive', 'custom'], default='quick',
                       help='Search mode: quick (few trials), comprehensive (many trials), or custom')
    parser.add_argument('--max_trials', type=int, help='Maximum number of trials (overrides mode default)')
    
    args = parser.parse_args()
    
    print("🚀 Simple Hyperparameter Search for Wav2Vec2")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'quick':
        search_space = QUICK_SEARCH_SPACE
        max_trials = args.max_trials or 20
    elif args.mode == 'comprehensive':
        search_space = COMPREHENSIVE_SEARCH_SPACE
        max_trials = args.max_trials or 50
    else:  # custom
        print("Custom mode - modify QUICK_SEARCH_SPACE in the script")
        search_space = QUICK_SEARCH_SPACE
        max_trials = args.max_trials or 15
    
    print(f"Search space: {search_space}")
    print(f"Max trials: {max_trials}")
    
    # Run search
    start_time = datetime.now()
    results, output_dir = run_hyperparameter_search(search_space, args.mode, max_trials)
    end_time = datetime.now()
    
    print(f"\n⏱️ Search completed in: {end_time - start_time}")
    print(f"📁 Results directory: {output_dir}")
    
    # Print final summary
    valid_results = [r for r in results if r.get('accuracy', 0) > 0]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['accuracy'])
        print(f"\n🎯 FINAL BEST RESULT:")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        print(f"Hyperparameters: {best_result['hyperparams']}")
    else:
        print("❌ No successful trials found!")

if __name__ == "__main__":
    main()

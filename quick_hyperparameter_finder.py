#!/usr/bin/env python3
"""
Quick Start: Find Good Hyperparameters for Your Wav2Vec2 Model

This script will help you quickly find better hyperparameters than the defaults.
It focuses on the most impactful parameters and runs fast trials.

Usage:
    python quick_hyperparameter_finder.py

Expected runtime: 30 minutes to 2 hours (depending on your GPU)
"""

import json
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# Most impactful hyperparameters to test
SEARCH_COMBINATIONS = [
    # Format: (learning_rate, batch_size, epochs, pooling_mode)
    
    # Conservative learning rates (safe choices)
    (1e-5, 16, 60, 'mean'),
    (3e-5, 16, 60, 'mean'),
    (5e-5, 16, 60, 'mean'),
    
    # Test max pooling (often better for detection tasks)
    (3e-5, 16, 60, 'max'),
    (5e-5, 16, 60, 'max'),
    
    # Different batch sizes
    (3e-5, 8, 60, 'mean'),
    (3e-5, 24, 60, 'mean'),
    
    # Different training lengths
    (3e-5, 16, 40, 'mean'),
    (3e-5, 16, 80, 'mean'),
    
    # Higher learning rates (aggressive but might work)
    (8e-5, 16, 60, 'mean'),
    (1e-4, 16, 40, 'mean'),
    
    # Memory-efficient options
    (3e-5, 8, 60, 'max'),
    (5e-5, 8, 80, 'max'),
]

def create_training_script(learning_rate, batch_size, epochs, pooling_mode, trial_name):
    """Create a standalone training script for this hyperparameter combination"""
    
    script_content = f'''#!/usr/bin/env python3
"""
Training script for {trial_name}
Learning Rate: {learning_rate}
Batch Size: {batch_size}
Epochs: {epochs}
Pooling: {pooling_mode}
"""

import sys
import os
import json
sys.path.append('./scripts')

# Import your training functions
from fine_tune_wav2vec2_sen_ham_CW_max_pooling import *

def main():
    print(f"🔄 Running {trial_name}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Pooling Mode: {pooling_mode}")
    
    # Configuration
    csv_file = './csv/balanced_train.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    output_dir = './quick_hyperparameter_search/{trial_name}'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load dataset
        df = load_dataset(csv_file)
        
        # Handle small classes (duplicate if needed)
        class_counts = df['label'].value_counts()
        for label, count in class_counts.items():
            if count < 2:
                row_to_duplicate = df[df['label'] == label].iloc[0]
                df = pd.concat([df, pd.DataFrame([row_to_duplicate])], ignore_index=True)
        
        # Split data
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        # Calculate class weights
        class_weights = calculate_class_weights(train_df)
        
        # Create model config
        config = Wav2Vec2Config.from_pretrained(
            model_name,
            num_labels=num_labels,
            finetuning_task="audio-classification"
        )
        
        # Initialize model with specified pooling
        model = Wav2Vec2ForSpeechClassification.from_pretrained(
            model_name,
            config=config,
            pooling_mode='{pooling_mode}'
        )
        
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            return_attention_mask=True,
            do_normalize=True,
        )
        
        # Prepare datasets
        train_dataset = prepare_dataset(train_df, feature_extractor)
        val_dataset = prepare_dataset(val_df, feature_extractor)
        
        # Training arguments with our hyperparameters
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs={epochs},
            per_device_train_batch_size={batch_size},
            per_device_eval_batch_size={batch_size},
            gradient_accumulation_steps=1,
            learning_rate={learning_rate},
            weight_decay=0.01,
            warmup_ratio=0.1,
            
            # Evaluation settings
            save_strategy="steps",
            save_steps=500,
            eval_strategy="steps", 
            eval_steps=500,
            logging_steps=50,
            
            # Performance settings
            fp16=True,
            gradient_checkpointing=True,
            dataloader_pin_memory=True,
            
            # Model selection
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=1,
            
            # Reduce output noise
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
        
        # Train the model
        trainer.train()
        
        # Final evaluation
        eval_results = trainer.evaluate()
        final_accuracy = eval_results.get('eval_accuracy', 0.0)
        
        # Save the model
        trainer.save_model(output_dir)
        feature_extractor.save_pretrained(output_dir)
        
        # Save results
        results = {{
            'trial_name': '{trial_name}',
            'hyperparameters': {{
                'learning_rate': {learning_rate},
                'batch_size': {batch_size},
                'epochs': {epochs},
                'pooling_mode': '{pooling_mode}',
            }},
            'accuracy': final_accuracy,
            'eval_results': eval_results,
            'model_path': output_dir
        }}
        
        with open(f'{{output_dir}}/results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ {trial_name} completed!")
        print(f"Final Accuracy: {{final_accuracy:.4f}}")
        print(f"Results saved to: {{output_dir}}/results.json")
        
        return results
        
    except Exception as e:
        print(f"❌ {trial_name} failed: {{str(e)}}")
        error_results = {{
            'trial_name': '{trial_name}',
            'hyperparameters': {{
                'learning_rate': {learning_rate},
                'batch_size': {batch_size},
                'epochs': {epochs},
                'pooling_mode': '{pooling_mode}',
            }},
            'accuracy': 0.0,
            'error': str(e)
        }}
        
        with open(f'{{output_dir}}/error.json', 'w') as f:
            json.dump(error_results, f, indent=2)
        
        return error_results

if __name__ == "__main__":
    main()
'''
    
    return script_content

def run_quick_search():
    """Run the quick hyperparameter search"""
    
    print("🚀 Quick Hyperparameter Finder for Wav2Vec2")
    print("=" * 50)
    print(f"Will test {len(SEARCH_COMBINATIONS)} combinations")
    print("Expected time: 30 minutes to 2 hours")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"quick_hyperparameter_search_{timestamp}"
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, (lr, bs, epochs, pooling) in enumerate(SEARCH_COMBINATIONS):
        trial_name = f"trial_{i+1:02d}"
        
        print(f"\n📊 Trial {i+1}/{len(SEARCH_COMBINATIONS)}: {trial_name}")
        print(f"   Learning Rate: {lr}")
        print(f"   Batch Size: {bs}")
        print(f"   Epochs: {epochs}")
        print(f"   Pooling: {pooling}")
        
        # Create training script
        script_content = create_training_script(lr, bs, epochs, pooling, trial_name)
        script_path = f"{base_dir}/{trial_name}_train.py"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Run the training
        try:
            print("   🔄 Training started...")
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, timeout=7200)  # 2 hour timeout
            
            if result.returncode == 0:
                # Load results
                result_file = f"{base_dir}/{trial_name}/results.json"
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        trial_result = json.load(f)
                    
                    accuracy = trial_result.get('accuracy', 0)
                    print(f"   ✅ Completed! Accuracy: {accuracy:.4f}")
                    results.append(trial_result)
                else:
                    print("   ⚠️ Warning: No results file found")
                    results.append({
                        'trial_name': trial_name,
                        'accuracy': 0.0,
                        'error': 'No results file'
                    })
            else:
                print(f"   ❌ Training failed")
                print(f"   Error: {result.stderr[:200]}...")
                results.append({
                    'trial_name': trial_name,
                    'accuracy': 0.0,
                    'error': result.stderr[:500]
                })
                
        except subprocess.TimeoutExpired:
            print("   ⏱️ Training timed out (2 hours)")
            results.append({
                'trial_name': trial_name,
                'accuracy': 0.0,
                'error': 'Timeout after 2 hours'
            })
        
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            results.append({
                'trial_name': trial_name,
                'accuracy': 0.0,
                'error': str(e)
            })
        
        # Save intermediate results
        with open(f"{base_dir}/all_results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    # Analyze results
    analyze_results(results, base_dir)
    
    return results, base_dir

def analyze_results(results, base_dir):
    """Analyze and summarize the results"""
    
    print("\n" + "=" * 60)
    print("📊 QUICK HYPERPARAMETER SEARCH RESULTS")
    print("=" * 60)
    
    # Filter successful results
    successful_results = [r for r in results if r.get('accuracy', 0) > 0]
    
    if not successful_results:
        print("❌ No successful trials found!")
        print("Check your data file path and GPU memory availability")
        return
    
    # Sort by accuracy
    successful_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"\\n📈 SUMMARY:")
    print(f"   Total trials: {len(results)}")
    print(f"   Successful: {len(successful_results)}")
    print(f"   Failed: {len(results) - len(successful_results)}")
    
    # Show top 3 results
    print(f"\\n🏆 TOP 3 RESULTS:")
    for i, result in enumerate(successful_results[:3]):
        print(f"\\n   {i+1}. {result['trial_name']} - Accuracy: {result['accuracy']:.4f}")
        if 'hyperparameters' in result:
            hp = result['hyperparameters']
            print(f"      Learning Rate: {hp.get('learning_rate', 'N/A')}")
            print(f"      Batch Size: {hp.get('batch_size', 'N/A')}")
            print(f"      Epochs: {hp.get('epochs', 'N/A')}")
            print(f"      Pooling: {hp.get('pooling_mode', 'N/A')}")
    
    # Best result details
    best_result = successful_results[0]
    print(f"\\n🎯 BEST HYPERPARAMETERS:")
    if 'hyperparameters' in best_result:
        hp = best_result['hyperparameters']
        print(f"   Learning Rate: {hp.get('learning_rate', 'N/A')}")
        print(f"   Batch Size: {hp.get('batch_size', 'N/A')}")
        print(f"   Epochs: {hp.get('epochs', 'N/A')}")
        print(f"   Pooling Mode: {hp.get('pooling_mode', 'N/A')}")
        print(f"   Final Accuracy: {best_result['accuracy']:.4f}")
    
    # Save best hyperparameters
    best_hp_file = f"{base_dir}/best_hyperparameters.json"
    with open(best_hp_file, 'w') as f:
        json.dump({
            'best_result': best_result,
            'top_3_results': successful_results[:3],
            'summary': {
                'total_trials': len(results),
                'successful_trials': len(successful_results),
                'best_accuracy': best_result['accuracy']
            }
        }, f, indent=2)
    
    print(f"\\n💾 Results saved to:")
    print(f"   All results: {base_dir}/all_results.json")
    print(f"   Best hyperparameters: {best_hp_file}")
    
    # Recommendations
    print(f"\\n💡 RECOMMENDATIONS:")
    
    best_accuracy = best_result['accuracy']
    if best_accuracy > 0.9:
        print("   🎉 Excellent results! Your model is performing very well.")
        print("   Consider running comprehensive search for final optimization.")
    elif best_accuracy > 0.85:
        print("   👍 Good results! You can probably improve further.")
        print("   Try running more comprehensive hyperparameter search.")
    elif best_accuracy > 0.8:
        print("   🔄 Decent results, but there's room for improvement.")
        print("   Consider data quality, augmentation, or longer training.")
    else:
        print("   ⚠️ Results are below expectations.")
        print("   Check your data quality and consider different approaches.")
    
    if 'hyperparameters' in best_result:
        hp = best_result['hyperparameters']
        
        # Learning rate advice
        lr = hp.get('learning_rate', 0)
        if lr >= 8e-5:
            print("   📈 Best learning rate is high - model might benefit from aggressive learning")
        elif lr <= 1e-5:
            print("   📉 Best learning rate is low - consider more conservative training")
        
        # Pooling advice
        if hp.get('pooling_mode') == 'max':
            print("   🎯 Max pooling worked best - good for detection tasks")
        
        # Batch size advice
        bs = hp.get('batch_size', 0)
        if bs <= 8:
            print("   💾 Small batch size worked best - try gradient accumulation for stability")
        elif bs >= 24:
            print("   🚀 Large batch size worked best - your GPU can handle more intensive training")

def main():
    print("Quick Hyperparameter Finder")
    print("This will help you find good hyperparameters quickly!")
    print()
    
    # Check if required files exist
    required_files = [
        './csv/balanced_train.csv',
        './scripts/fine_tune_wav2vec2_sen_ham_CW_max_pooling.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   {f}")
        print("\\nPlease ensure these files exist before running.")
        return
    
    print("✅ All required files found!")
    print()
    
    # Ask for confirmation
    response = input("Ready to start hyperparameter search? This may take 30min-2hours (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Search cancelled.")
        return
    
    # Run the search
    start_time = datetime.now()
    results, output_dir = run_quick_search()
    end_time = datetime.now()
    
    print(f"\\n⏱️ Total time: {end_time - start_time}")
    print(f"📁 Results directory: {output_dir}")
    
    # Final message
    print("\\n🎉 Quick hyperparameter search completed!")
    print("Use the best hyperparameters for your final model training.")

if __name__ == "__main__":
    main()

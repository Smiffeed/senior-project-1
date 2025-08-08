"""
Manual Hyperparameter Testing Script

This script allows you to manually test specific hyperparameter combinations
for your Wav2Vec2 model. Modify the parameter sets below to test different configurations.
"""

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Config
from transformers import TrainingArguments
from transformers import EarlyStoppingCallback
import os
import json
from datetime import datetime

# Import your existing functions
from scripts.fine_tune_wav2vec2_sen_ham_CW import (
    load_dataset, prepare_dataset, SafeSavingTrainer, compute_metrics,
    collate_fn, label_map, num_labels
)

def test_hyperparameters(params, csv_file, model_name, output_dir, test_name):
    """
    Test a specific set of hyperparameters
    
    Args:
        params: Dictionary of hyperparameters
        csv_file: Path to training CSV
        model_name: Pre-trained model name
        output_dir: Output directory for this test
        test_name: Name for this test
    """
    print(f"\n{'='*60}")
    print(f"TESTING: {test_name}")
    print(f"{'='*60}")
    
    # Print parameters
    print("Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Load and prepare data
    df = load_dataset(csv_file)
    
    # Split data: 60% train, 20% val, 20% test
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.25, random_state=42, stratify=train_val_df['label']
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples") 
    print(f"  Test: {len(test_df)} samples")
    
    # Prepare feature extractor and datasets
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name, return_attention_mask=True, do_normalize=True
    )
    
    train_dataset = prepare_dataset(train_df, feature_extractor)
    val_dataset = prepare_dataset(val_df, feature_extractor)
    test_dataset = prepare_dataset(test_df, feature_extractor)
    
    # Calculate class weights
    train_labels = [ex['label'] for ex in train_dataset]
    class_weights = torch.FloatTensor(
        compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    )
    
    # Load model
    config = Wav2Vec2Config.from_pretrained(
        model_name, num_labels=num_labels, finetuning_task="audio-classification"
    )
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, config=config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=params.get('num_epochs', 100),
        per_device_train_batch_size=params.get('batch_size', 8),
        per_device_eval_batch_size=params.get('batch_size', 8),
        gradient_accumulation_steps=params.get('gradient_accumulation_steps', 3),
        learning_rate=params.get('learning_rate', 5e-5),
        weight_decay=params.get('weight_decay', 0.005),
        warmup_ratio=params.get('warmup_ratio', 0.15),
        lr_scheduler_type=params.get('lr_scheduler_type', 'cosine'),
        eval_steps=params.get('eval_steps', 200),
        save_steps=params.get('save_steps', 200),
        logging_steps=params.get('logging_steps', 50),
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        fp16=True,
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        remove_unused_columns=True,
        save_total_limit=3,
        adam_epsilon=params.get('adam_epsilon', 1e-6),
        max_grad_norm=params.get('max_grad_norm', 1.0),
    )
    
    # Initialize trainer
    trainer = SafeSavingTrainer(
        class_weights=class_weights,
        use_focal_loss=params.get('use_focal_loss', True),
        gamma=params.get('focal_gamma', 3.0),
        alpha=params.get('focal_alpha', 0.75),
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=params.get('early_stopping_patience', 15))]
    )
    
    try:
        # Train the model
        print("\nStarting training...")
        trainer.train()
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_results = trainer.evaluate()
        
        # Evaluate on test set
        print("Evaluating on test set...")
        trainer.args.eval_dataset = test_dataset
        test_results = trainer.evaluate()
        
        # Save model
        trainer.save_model(output_dir)
        feature_extractor.save_pretrained(output_dir)
        
        # Compile results
        results = {
            'test_name': test_name,
            'parameters': params,
            'validation_results': {
                'accuracy': val_results.get('eval_accuracy', 0.0),
                'f1_macro': val_results.get('eval_f1_macro', 0.0),
                'f1_weighted': val_results.get('eval_f1_weighted', 0.0),
                'profanity_f1': val_results.get('eval_profanity_f1', 0.0),
            },
            'test_results': {
                'accuracy': test_results.get('eval_accuracy', 0.0),
                'f1_macro': test_results.get('eval_f1_macro', 0.0),
                'f1_weighted': test_results.get('eval_f1_weighted', 0.0),
                'profanity_f1': test_results.get('eval_profanity_f1', 0.0),
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Print results
        print(f"\n{'='*40}")
        print("RESULTS")
        print(f"{'='*40}")
        print("Validation Results:")
        for key, value in results['validation_results'].items():
            print(f"  {key}: {value:.4f}")
        
        print("\nTest Results:")
        for key, value in results['test_results'].items():
            print(f"  {key}: {value:.4f}")
        
        # Save results to file
        results_file = os.path.join(output_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None

def main():
    """
    Main function to test different hyperparameter configurations
    """
    csv_file = './csv/train_with_augmentation.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    base_output_dir = './models/manual_hyperparameter_tests'
    
    # Create base directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Define different parameter sets to test
    # You can modify these or add more configurations
    
    # Configuration 1: Conservative (good starting point)
    config_1 = {
        'learning_rate': 3e-5,
        'batch_size': 8,
        'gradient_accumulation_steps': 3,
        'num_epochs': 80,
        'weight_decay': 0.005,
        'warmup_ratio': 0.15,
        'focal_gamma': 3.0,
        'focal_alpha': 0.75,
        'use_focal_loss': True,
        'early_stopping_patience': 15,
        'lr_scheduler_type': 'cosine',
        'eval_steps': 200
    }
    
    # Configuration 2: Higher learning rate
    config_2 = {
        'learning_rate': 5e-5,
        'batch_size': 8,
        'gradient_accumulation_steps': 3,
        'num_epochs': 100,
        'weight_decay': 0.01,
        'warmup_ratio': 0.2,
        'focal_gamma': 4.0,
        'focal_alpha': 0.8,
        'use_focal_loss': True,
        'early_stopping_patience': 12,
        'lr_scheduler_type': 'cosine',
        'eval_steps': 150
    }
    
    # Configuration 3: Smaller batch, more accumulation
    config_3 = {
        'learning_rate': 4e-5,
        'batch_size': 6,
        'gradient_accumulation_steps': 4,
        'num_epochs': 120,
        'weight_decay': 0.008,
        'warmup_ratio': 0.1,
        'focal_gamma': 2.5,
        'focal_alpha': 0.7,
        'use_focal_loss': True,
        'early_stopping_patience': 20,
        'lr_scheduler_type': 'linear',
        'eval_steps': 250
    }
    
    # Configuration 4: No focal loss (baseline)
    config_4 = {
        'learning_rate': 3e-5,
        'batch_size': 8,
        'gradient_accumulation_steps': 3,
        'num_epochs': 80,
        'weight_decay': 0.005,
        'warmup_ratio': 0.15,
        'focal_gamma': 2.0,
        'focal_alpha': 0.5,
        'use_focal_loss': False,  # No focal loss
        'early_stopping_patience': 15,
        'lr_scheduler_type': 'cosine',
        'eval_steps': 200
    }
    
    # List of configurations to test
    configurations = [
        (config_1, "Conservative_Setup"),
        (config_2, "Higher_LR_Strong_Focal"),
        (config_3, "Small_Batch_Linear_LR"),
        (config_4, "No_Focal_Loss_Baseline")
    ]
    
    # Test each configuration
    all_results = []
    
    for config, name in configurations:
        print(f"\n{'#'*80}")
        print(f"TESTING CONFIGURATION: {name}")
        print(f"{'#'*80}")
        
        output_dir = os.path.join(base_output_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        
        results = test_hyperparameters(config, csv_file, model_name, output_dir, name)
        
        if results:
            all_results.append(results)
            print(f"✅ {name} completed successfully")
        else:
            print(f"❌ {name} failed")
    
    # Compare all results
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}")
    
    if all_results:
        # Sort by test F1 score
        all_results.sort(key=lambda x: x['test_results']['f1_macro'], reverse=True)
        
        print("Ranking by Test F1 Score:")
        for i, result in enumerate(all_results):
            print(f"{i+1}. {result['test_name']}")
            print(f"   Test F1: {result['test_results']['f1_macro']:.4f}")
            print(f"   Test Accuracy: {result['test_results']['accuracy']:.4f}")
            print(f"   Profanity F1: {result['test_results']['profanity_f1']:.4f}")
            print()
        
        # Save comparison
        comparison_file = os.path.join(base_output_dir, 'comparison.json')
        with open(comparison_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"🏆 WINNER: {all_results[0]['test_name']}")
        print(f"Best Test F1 Score: {all_results[0]['test_results']['f1_macro']:.4f}")
        print(f"\nAll results saved to: {base_output_dir}")
    
    else:
        print("❌ No successful configurations")

if __name__ == "__main__":
    main()

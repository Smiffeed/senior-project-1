import torch
import optuna
import pandas as pd
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Config
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score
import tempfile
import shutil
import os
from collections import Counter
import json
from datetime import datetime

# Import your existing functions
from scripts.fine_tune_wav2vec2_sen_ham_CW import (
    load_dataset, prepare_dataset, SafeSavingTrainer, compute_metrics,
    collate_fn, label_map, num_labels, calculate_class_weights
)

class HyperparameterOptimizer:
    def __init__(self, csv_file, model_name, base_output_dir, n_trials=50, n_folds=3):
        """
        Initialize hyperparameter optimizer
        
        Args:
            csv_file: Path to training CSV
            model_name: Pre-trained model name
            base_output_dir: Base directory for saving models
            n_trials: Number of hyperparameter combinations to try
            n_folds: Number of cross-validation folds
        """
        self.csv_file = csv_file
        self.model_name = model_name
        self.base_output_dir = base_output_dir
        self.n_trials = n_trials
        self.n_folds = n_folds
        
        # Load and prepare data once
        self.df = load_dataset(csv_file)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, return_attention_mask=True, do_normalize=True
        )
        
        # Split data: 60% train, 20% val, 20% test
        train_val_df, self.test_df = train_test_split(
            self.df, test_size=0.2, random_state=42, stratify=self.df['label']
        )
        self.train_df, self.val_df = train_test_split(
            train_val_df, test_size=0.25, random_state=42, stratify=train_val_df['label']  # 0.25 * 0.8 = 0.2
        )
        
        print(f"Dataset split:")
        print(f"  Train: {len(self.train_df)} samples")
        print(f"  Validation: {len(self.val_df)} samples") 
        print(f"  Test: {len(self.test_df)} samples")
        
        # Prepare datasets
        self.train_dataset = prepare_dataset(self.train_df, self.feature_extractor)
        self.val_dataset = prepare_dataset(self.val_df, self.feature_extractor)
        self.test_dataset = prepare_dataset(self.test_df, self.feature_extractor)
        
        # Track best results
        self.best_score = 0.0
        self.best_params = None
        self.trial_results = []

    def suggest_hyperparameters(self, trial):
        """
        Suggest hyperparameters for optimization
        """
        # Learning rate - most important for fine-tuning
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True)
        
        # Batch size and gradient accumulation
        batch_size = trial.suggest_categorical('batch_size', [4, 6, 8, 12, 16])
        gradient_accumulation_steps = trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 3, 4])
        
        # Training epochs
        num_epochs = trial.suggest_int('num_epochs', 20, 200)
        
        # Regularization
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.3)
        
        # Focal loss parameters
        use_focal_loss = trial.suggest_categorical('use_focal_loss', [True, False])
        focal_gamma = trial.suggest_float('focal_gamma', 1.0, 5.0) if use_focal_loss else 2.0
        focal_alpha = trial.suggest_float('focal_alpha', 0.25, 0.95) if use_focal_loss else 0.75
        
        # Early stopping
        early_stopping_patience = trial.suggest_int('early_stopping_patience', 5, 25)
        
        # Scheduler
        lr_scheduler_type = trial.suggest_categorical('lr_scheduler_type', 
            ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant_with_warmup'])
        
        # Save frequency
        eval_steps = trial.suggest_int('eval_steps', 50, 500)
        
        return {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'num_epochs': num_epochs,
            'weight_decay': weight_decay,
            'warmup_ratio': warmup_ratio,
            'use_focal_loss': use_focal_loss,
            'focal_gamma': focal_gamma,
            'focal_alpha': focal_alpha,
            'early_stopping_patience': early_stopping_patience,
            'lr_scheduler_type': lr_scheduler_type,
            'eval_steps': eval_steps
        }

    def objective(self, trial):
        """
        Objective function for hyperparameter optimization
        """
        params = self.suggest_hyperparameters(trial)
        
        print(f"\n{'='*60}")
        print(f"TRIAL {trial.number + 1}/{self.n_trials}")
        print(f"{'='*60}")
        print("Testing parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        try:
            # Perform cross-validation with these hyperparameters
            cv_scores = self.cross_validate(params, trial.number)
            
            # Calculate mean F1 score across folds
            mean_f1 = np.mean([score['f1_macro'] for score in cv_scores])
            std_f1 = np.std([score['f1_macro'] for score in cv_scores])
            
            print(f"\nCross-validation results:")
            print(f"  Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
            
            # Store results
            result = {
                'trial': trial.number,
                'params': params,
                'cv_scores': cv_scores,
                'mean_f1': mean_f1,
                'std_f1': std_f1,
                'timestamp': datetime.now().isoformat()
            }
            self.trial_results.append(result)
            
            # Update best if this is better
            if mean_f1 > self.best_score:
                self.best_score = mean_f1
                self.best_params = params.copy()
                print(f"🎉 NEW BEST SCORE: {mean_f1:.4f}")
            
            return mean_f1
            
        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {e}")
            return 0.0

    def cross_validate(self, params, trial_num):
        """
        Perform cross-validation with given hyperparameters
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train_dataset)):
            print(f"\n  Fold {fold + 1}/{self.n_folds}")
            
            # Create train and validation subsets
            train_subset = self.train_dataset.select(train_idx)
            val_subset = self.train_dataset.select(val_idx)
            
            # Train model with these hyperparameters
            score = self.train_and_evaluate(train_subset, val_subset, params, trial_num, fold)
            fold_scores.append(score)
            
            print(f"    Fold {fold + 1} F1: {score['f1_macro']:.4f}")
        
        return fold_scores

    def train_and_evaluate(self, train_dataset, val_dataset, params, trial_num, fold):
        """
        Train model with specific hyperparameters and evaluate
        """
        # Create temporary output directory
        output_dir = os.path.join(self.base_output_dir, f"trial_{trial_num}_fold_{fold}")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Calculate class weights
            train_labels = [ex['label'] for ex in train_dataset]
            class_weights = torch.FloatTensor(
                compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
            )
            
            # Load model
            config = Wav2Vec2Config.from_pretrained(
                self.model_name, num_labels=num_labels, finetuning_task="audio-classification"
            )
            model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name, config=config)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=params['num_epochs'],
                per_device_train_batch_size=params['batch_size'],
                per_device_eval_batch_size=params['batch_size'],
                gradient_accumulation_steps=params['gradient_accumulation_steps'],
                learning_rate=params['learning_rate'],
                weight_decay=params['weight_decay'],
                warmup_ratio=params['warmup_ratio'],
                lr_scheduler_type=params['lr_scheduler_type'],
                eval_steps=params['eval_steps'],
                save_steps=params['eval_steps'],
                logging_steps=params['eval_steps'] // 2,
                eval_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1_macro",
                greater_is_better=True,
                fp16=True,
                dataloader_num_workers=0,
                dataloader_drop_last=True,
                remove_unused_columns=True,
                save_total_limit=1,  # Only keep best checkpoint
                disable_tqdm=True,   # Reduce output noise
            )
            
            # Initialize trainer
            trainer = SafeSavingTrainer(
                class_weights=class_weights,
                use_focal_loss=params['use_focal_loss'],
                gamma=params['focal_gamma'],
                alpha=params['focal_alpha'],
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                data_collator=collate_fn,
            )
            
            # Train
            trainer.train()
            
            # Evaluate on validation set
            eval_results = trainer.evaluate()
            
            return {
                'accuracy': eval_results.get('eval_accuracy', 0.0),
                'f1_macro': eval_results.get('eval_f1_macro', 0.0),
                'f1_weighted': eval_results.get('eval_f1_weighted', 0.0),
                'profanity_f1': eval_results.get('eval_profanity_f1', 0.0)
            }
            
        except Exception as e:
            print(f"    Error in fold {fold}: {e}")
            return {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0, 'profanity_f1': 0.0}
        
        finally:
            # Clean up to save disk space
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)

    def optimize(self):
        """
        Run hyperparameter optimization
        """
        print(f"🚀 Starting hyperparameter optimization with {self.n_trials} trials")
        print(f"Using {self.n_folds}-fold cross-validation")
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Save results
        self.save_results(study)
        
        return study

    def save_results(self, study):
        """
        Save optimization results
        """
        results_dir = os.path.join(self.base_output_dir, "optimization_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save best parameters
        best_params_file = os.path.join(results_dir, "best_hyperparameters.json")
        with open(best_params_file, 'w') as f:
            json.dump({
                'best_score': self.best_score,
                'best_params': self.best_params,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Save all trial results
        all_results_file = os.path.join(results_dir, "all_trials.json")
        with open(all_results_file, 'w') as f:
            json.dump(self.trial_results, f, indent=2)
        
        # Create summary report
        self.create_summary_report(study, results_dir)
        
        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Best F1 Score: {self.best_score:.4f}")
        print("Best Parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"\nResults saved to: {results_dir}")

    def create_summary_report(self, study, results_dir):
        """
        Create a detailed summary report
        """
        report_file = os.path.join(results_dir, "optimization_summary.txt")
        
        with open(report_file, 'w') as f:
            f.write("HYPERPARAMETER OPTIMIZATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: {self.csv_file}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Total trials: {len(self.trial_results)}\n")
            f.write(f"Cross-validation folds: {self.n_folds}\n\n")
            
            f.write("BEST RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Best F1 Score: {self.best_score:.4f}\n\n")
            
            f.write("Best Hyperparameters:\n")
            for key, value in self.best_params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Top 5 trials
            sorted_trials = sorted(self.trial_results, key=lambda x: x['mean_f1'], reverse=True)
            f.write("TOP 5 TRIALS:\n")
            f.write("-" * 20 + "\n")
            for i, trial in enumerate(sorted_trials[:5]):
                f.write(f"{i+1}. Trial {trial['trial']}: F1 = {trial['mean_f1']:.4f} ± {trial['std_f1']:.4f}\n")
            f.write("\n")
            
            # Parameter importance (if available)
            try:
                importance = optuna.importance.get_param_importances(study)
                f.write("PARAMETER IMPORTANCE:\n")
                f.write("-" * 20 + "\n")
                for param, imp in importance.items():
                    f.write(f"  {param}: {imp:.4f}\n")
            except:
                pass

    def train_final_model(self):
        """
        Train final model with best hyperparameters on full training data
        """
        if self.best_params is None:
            raise ValueError("No best parameters found. Run optimization first.")
        
        print(f"\n{'='*60}")
        print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
        print(f"{'='*60}")
        
        # Use full training data (train + val)
        full_train_df = pd.concat([self.train_df, self.val_df], ignore_index=True)
        full_train_dataset = prepare_dataset(full_train_df, self.feature_extractor)
        
        output_dir = os.path.join(self.base_output_dir, "best_model_final")
        
        # Calculate class weights
        train_labels = [ex['label'] for ex in full_train_dataset]
        class_weights = torch.FloatTensor(
            compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        )
        
        # Load model
        config = Wav2Vec2Config.from_pretrained(
            self.model_name, num_labels=num_labels, finetuning_task="audio-classification"
        )
        model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name, config=config)
        
        # Training arguments with best hyperparameters
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.best_params['num_epochs'],
            per_device_train_batch_size=self.best_params['batch_size'],
            per_device_eval_batch_size=self.best_params['batch_size'],
            gradient_accumulation_steps=self.best_params['gradient_accumulation_steps'],
            learning_rate=self.best_params['learning_rate'],
            weight_decay=self.best_params['weight_decay'],
            warmup_ratio=self.best_params['warmup_ratio'],
            lr_scheduler_type=self.best_params['lr_scheduler_type'],
            eval_steps=self.best_params['eval_steps'],
            save_steps=self.best_params['eval_steps'],
            logging_steps=self.best_params['eval_steps'] // 2,
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
        )
        
        # Initialize trainer
        trainer = SafeSavingTrainer(
            class_weights=class_weights,
            use_focal_loss=self.best_params['use_focal_loss'],
            gamma=self.best_params['focal_gamma'],
            alpha=self.best_params['focal_alpha'],
            model=model,
            args=training_args,
            train_dataset=full_train_dataset,
            eval_dataset=self.test_dataset,  # Use test set for final evaluation
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )
        
        # Train
        trainer.train()
        
        # Final evaluation on test set
        test_results = trainer.evaluate()
        
        # Save model and feature extractor
        trainer.save_model(output_dir)
        self.feature_extractor.save_pretrained(output_dir)
        
        print(f"\nFINAL MODEL PERFORMANCE ON TEST SET:")
        print(f"  Accuracy: {test_results.get('eval_accuracy', 0.0):.4f}")
        print(f"  F1 Macro: {test_results.get('eval_f1_macro', 0.0):.4f}")
        print(f"  F1 Weighted: {test_results.get('eval_f1_weighted', 0.0):.4f}")
        print(f"  Profanity F1: {test_results.get('eval_profanity_f1', 0.0):.4f}")
        print(f"\nFinal model saved to: {output_dir}")
        
        return output_dir, test_results


def run_hyperparameter_optimization():
    """
    Main function to run hyperparameter optimization
    """
    # Configuration
    csv_file = './csv/train_with_augmentation.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    base_output_dir = './models/hyperparameter_optimization'
    n_trials = 30  # Adjust based on your computational budget
    n_folds = 3    # 3-fold CV for faster optimization
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        csv_file=csv_file,
        model_name=model_name,
        base_output_dir=base_output_dir,
        n_trials=n_trials,
        n_folds=n_folds
    )
    
    # Run optimization
    study = optimizer.optimize()
    
    # Train final model with best hyperparameters
    final_model_path, test_results = optimizer.train_final_model()
    
    return optimizer, study, final_model_path, test_results


if __name__ == "__main__":
    # Import required sklearn function
    from sklearn.utils.class_weight import compute_class_weight
    
    # Run optimization
    optimizer, study, final_model_path, test_results = run_hyperparameter_optimization()
    
    print(f"\n🎉 HYPERPARAMETER OPTIMIZATION COMPLETE!")
    print(f"Best model saved to: {final_model_path}")
    print(f"Test set F1 Score: {test_results.get('eval_f1_macro', 0.0):.4f}")

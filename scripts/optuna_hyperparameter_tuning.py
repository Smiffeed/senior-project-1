import torch
import optuna
import json
import numpy as np
from datetime import datetime
import os
import sys
import sqlite3
sys.path.append('.')

# Import from the main training script
from fine_tune_wav2vec2_sen_ham_CW import *

class OptunaHyperparameterTuning:
    """
    Hyperparameter optimization using Optuna (more efficient than grid search)
    """
    
    def __init__(self, csv_file, model_name, base_output_dir):
        self.csv_file = csv_file
        self.model_name = model_name
        self.base_output_dir = base_output_dir
        
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
        
        # Calculate class weights
        train_labels = [ex['label'] for ex in self.train_dataset]
        self.class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        self.class_weights = torch.FloatTensor(self.class_weights)
        
        print(f"Dataset sizes - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
    
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        
        # Suggest hyperparameters
        config = {
            # Core parameters
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True),
            'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [2, 4, 8, 16]),
            'gradient_accumulation_steps': trial.suggest_categorical('gradient_accumulation_steps', [1, 2, 4]),
            'num_train_epochs': trial.suggest_int('num_train_epochs', 50, 150),
            'max_steps': trial.suggest_int('max_steps', 2000, 6000),
            
            # Regularization
            'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.1),
            'warmup_ratio': trial.suggest_float('warmup_ratio', 0.03, 0.15),
            'attention_dropout': trial.suggest_float('attention_dropout', 0.1, 0.4),
            'hidden_dropout': trial.suggest_float('hidden_dropout', 0.3, 0.6),
            
            # Training strategy
            'lr_scheduler_type': trial.suggest_categorical('lr_scheduler_type', ['linear', 'cosine', 'cosine_with_restarts']),
            'early_stopping_patience': trial.suggest_int('early_stopping_patience', 10, 25),
            
            # Focal loss
            'gamma': trial.suggest_float('gamma', 1.5, 5.0),
            'alpha': trial.suggest_float('alpha', 0.5, 0.9),
            
            # Fixed parameters for faster training during optimization
            'save_steps': 300,
            'eval_steps': 150,
            'fp16': True,
            'gradient_checkpointing': False,
        }
        
        try:
            # Create output directory
            output_dir = os.path.join(self.base_output_dir, f'trial_{trial.number}')
            os.makedirs(output_dir, exist_ok=True)
            
            # Save trial config
            with open(os.path.join(output_dir, 'trial_config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            
            # Train model
            val_f1_macro = self.train_and_evaluate(config, output_dir, trial)
            
            # Report intermediate value for pruning
            trial.report(val_f1_macro, step=0)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            return val_f1_macro
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0  # Return low score for failed trials
    
    def train_and_evaluate(self, config, output_dir, trial):
        """Train model and return validation F1 score"""
        
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
            save_steps=config['save_steps'],
            eval_steps=config['eval_steps'],
            logging_steps=50,
            save_strategy="steps",
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            lr_scheduler_type=config['lr_scheduler_type'],
            fp16=config['fp16'],
            gradient_checkpointing=config['gradient_checkpointing'],
            dataloader_pin_memory=True,
            save_total_limit=1,  # Keep only best model to save space
            logging_dir=None,    # Disable logging to save space
            report_to=None,      # Disable wandb
            push_to_hub=False,
            remove_unused_columns=True,
            dataloader_num_workers=0,
            dataloader_drop_last=True,
        )
        
        # Custom callback to report intermediate results to Optuna
        class OptunaCallback:
            def __init__(self, trial):
                self.trial = trial
                
            def on_evaluate(self, args, state, control, model=None, **kwargs):
                # Report current validation score
                current_score = state.log_history[-1].get('eval_f1_macro', 0.0)
                self.trial.report(current_score, step=state.global_step)
                
                # Check if trial should be pruned
                if self.trial.should_prune():
                    control.should_training_stop = True
        
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
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=config['early_stopping_patience']),
                OptunaCallback(trial)
            ],
            data_collator=collate_fn,
        )
        
        # Train model
        trainer.train()
        
        # Get final validation results
        eval_results = trainer.evaluate()
        val_f1_macro = eval_results.get('eval_f1_macro', 0.0)
        
        print(f"Trial {trial.number}: Val F1 = {val_f1_macro:.4f}")
        
        return val_f1_macro
    
    def run_optimization(self, n_trials=50, timeout=None):
        """Run Optuna optimization"""
        
        # Create study
        study_name = f"wav2vec2_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        storage_path = os.path.join(self.base_output_dir, f"{study_name}.db")
        
        # Create study with pruning
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=f'sqlite:///{storage_path}',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
        
        print(f"Starting Optuna optimization with {n_trials} trials...")
        print(f"Study database: {storage_path}")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Print results
        print(f"\n{'='*60}")
        print("OPTUNA OPTIMIZATION COMPLETED!")
        print(f"{'='*60}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.4f}")
        print(f"Best params:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Save results
        results_dir = os.path.join(self.base_output_dir, f'optuna_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save best parameters
        with open(os.path.join(results_dir, 'best_params.json'), 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        # Save study results
        trials_df = study.trials_dataframe()
        trials_df.to_csv(os.path.join(results_dir, 'all_trials.csv'), index=False)
        
        # Create visualization plots
        try:
            import matplotlib.pyplot as plt
            import optuna.visualization as vis
            
            # Optimization history
            fig = vis.matplotlib.plot_optimization_history(study)
            fig.savefig(os.path.join(results_dir, 'optimization_history.png'))
            plt.close(fig)
            
            # Parameter importances
            fig = vis.matplotlib.plot_param_importances(study)
            fig.savefig(os.path.join(results_dir, 'param_importances.png'))
            plt.close(fig)
            
            # Slice plot for top parameters
            important_params = list(study.best_params.keys())[:6]  # Top 6 parameters
            if len(important_params) > 0:
                fig = vis.matplotlib.plot_slice(study, params=important_params)
                fig.savefig(os.path.join(results_dir, 'parameter_effects.png'))
                plt.close(fig)
                
        except Exception as e:
            print(f"Could not create visualization plots: {e}")
        
        print(f"Results saved to: {results_dir}")
        
        # Prepare config for full training
        full_training_config = study.best_params.copy()
        full_training_config.update({
            'save_steps': 500,
            'eval_steps': 250,
            'early_stopping_patience': 20,
        })
        
        with open(os.path.join(results_dir, 'best_config_for_full_training.json'), 'w') as f:
            json.dump(full_training_config, f, indent=2)
        
        return {
            'study': study,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'results_dir': results_dir
        }


def main():
    # Configuration
    csv_file = './csv/train_with_augmentation.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    base_output_dir = './models/optuna_optimization'
    
    # Create base directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Initialize optimization
    print("Initializing Optuna Hyperparameter Optimization...")
    print("Optuna is more efficient than grid search and will automatically")
    print("focus on promising parameter regions and prune bad trials.")
    
    optimizer = OptunaHyperparameterTuning(csv_file, model_name, base_output_dir)
    
    # Run optimization
    results = optimizer.run_optimization(
        n_trials=30,      # Number of trials to run
        timeout=3600*6   # 6 hours timeout (optional)
    )
    
    print(f"\n🎉 OPTIMIZATION COMPLETED!")
    print(f"📊 Best F1 Score: {results['best_value']:.4f}")
    print(f"📁 Results saved in: {results['results_dir']}")
    print(f"⚙️  Best hyperparameters ready for full training!")


if __name__ == "__main__":
    main()

import torch
import pandas as pd
import numpy as np
from itertools import product
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Config
from transformers import TrainingArguments
import shutil

# Import your existing functions
from scripts.fine_tune_wav2vec2_sen_ham_CW import (
    load_dataset, prepare_dataset, SafeSavingTrainer, compute_metrics,
    collate_fn, label_map, num_labels
)

class SimpleGridSearch:
    def __init__(self, csv_file, model_name, base_output_dir):
        """
        Simple grid search for hyperparameter optimization
        """
        self.csv_file = csv_file
        self.model_name = model_name
        self.base_output_dir = base_output_dir
        
        # Load and prepare data
        self.df = load_dataset(csv_file)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, return_attention_mask=True, do_normalize=True
        )
        
        # Split data
        train_val_df, self.test_df = train_test_split(
            self.df, test_size=0.2, random_state=42, stratify=self.df['label']
        )
        self.train_df, self.val_df = train_test_split(
            train_val_df, test_size=0.25, random_state=42, stratify=train_val_df['label']
        )
        
        # Prepare datasets
        self.train_dataset = prepare_dataset(self.train_df, self.feature_extractor)
        self.val_dataset = prepare_dataset(self.val_df, self.feature_extractor)
        
        self.results = []
        self.best_score = 0.0
        self.best_params = None

    def define_search_space(self):
        """
        Define hyperparameter search space - smaller for computational efficiency
        """
        return {
            'learning_rate': [1e-5, 3e-5, 5e-5],
            'batch_size': [6, 8, 12],
            'num_epochs': [50, 100, 150],
            'weight_decay': [0.001, 0.005, 0.01],
            'warmup_ratio': [0.1, 0.15, 0.2],
            'focal_gamma': [2.0, 3.0, 4.0],
            'focal_alpha': [0.5, 0.75, 0.9],
            'gradient_accumulation_steps': [2, 3, 4]
        }

    def search(self, max_combinations=20):
        """
        Perform grid search with limited combinations
        """
        search_space = self.define_search_space()
        
        # Generate all combinations
        keys = list(search_space.keys())
        values = list(search_space.values())
        all_combinations = list(product(*values))
        
        print(f"Total possible combinations: {len(all_combinations)}")
        print(f"Testing {min(max_combinations, len(all_combinations))} combinations")
        
        # Randomly sample combinations if too many
        if len(all_combinations) > max_combinations:
            import random
            random.seed(42)
            selected_combinations = random.sample(all_combinations, max_combinations)
        else:
            selected_combinations = all_combinations
        
        # Test each combination
        for i, combination in enumerate(selected_combinations):
            params = dict(zip(keys, combination))
            
            print(f"\n{'='*50}")
            print(f"COMBINATION {i+1}/{len(selected_combinations)}")
            print(f"{'='*50}")
            print("Parameters:")
            for key, value in params.items():
                print(f"  {key}: {value}")
            
            try:
                score = self.evaluate_params(params, i)
                
                result = {
                    'combination': i,
                    'params': params,
                    'f1_score': score,
                    'timestamp': datetime.now().isoformat()
                }
                self.results.append(result)
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                    print(f"🎉 NEW BEST SCORE: {score:.4f}")
                
                print(f"Score: {score:.4f}")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                continue
        
        self.save_results()
        return self.best_params, self.best_score

    def evaluate_params(self, params, combination_id):
        """
        Evaluate a single parameter combination
        """
        output_dir = os.path.join(self.base_output_dir, f"combination_{combination_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Calculate class weights
            train_labels = [ex['label'] for ex in self.train_dataset]
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
                eval_steps=200,
                save_steps=200,
                logging_steps=100,
                eval_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1_macro",
                greater_is_better=True,
                fp16=True,
                dataloader_num_workers=0,
                dataloader_drop_last=True,
                remove_unused_columns=True,
                save_total_limit=1,
                disable_tqdm=True,
                lr_scheduler_type="cosine",
            )
            
            # Initialize trainer
            trainer = SafeSavingTrainer(
                class_weights=class_weights,
                use_focal_loss=True,
                gamma=params['focal_gamma'],
                alpha=params['focal_alpha'],
                model=model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                compute_metrics=compute_metrics,
                data_collator=collate_fn,
            )
            
            # Train
            trainer.train()
            
            # Evaluate
            eval_results = trainer.evaluate()
            f1_score = eval_results.get('eval_f1_macro', 0.0)
            
            return f1_score
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return 0.0
        
        finally:
            # Clean up
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)

    def save_results(self):
        """
        Save search results
        """
        results_dir = os.path.join(self.base_output_dir, "grid_search_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save best parameters
        best_file = os.path.join(results_dir, "best_params.json")
        with open(best_file, 'w') as f:
            json.dump({
                'best_score': self.best_score,
                'best_params': self.best_params,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Save all results
        all_results_file = os.path.join(results_dir, "all_results.json")
        with open(all_results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary
        self.create_summary(results_dir)
        
        print(f"\n{'='*50}")
        print("GRID SEARCH COMPLETE!")
        print(f"{'='*50}")
        print(f"Best F1 Score: {self.best_score:.4f}")
        print("Best Parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"\nResults saved to: {results_dir}")

    def create_summary(self, results_dir):
        """
        Create summary report
        """
        summary_file = os.path.join(results_dir, "summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("GRID SEARCH SUMMARY\n")
            f.write("=" * 30 + "\n\n")
            
            f.write(f"Total combinations tested: {len(self.results)}\n")
            f.write(f"Best F1 Score: {self.best_score:.4f}\n\n")
            
            f.write("Best Parameters:\n")
            for key, value in self.best_params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Top 5 results
            sorted_results = sorted(self.results, key=lambda x: x['f1_score'], reverse=True)
            f.write("Top 5 Results:\n")
            for i, result in enumerate(sorted_results[:5]):
                f.write(f"{i+1}. F1={result['f1_score']:.4f}\n")
                for key, value in result['params'].items():
                    f.write(f"   {key}: {value}\n")
                f.write("\n")

def run_simple_grid_search():
    """
    Run simple grid search
    """
    csv_file = './csv/train_with_augmentation.csv'
    model_name = "airesearch/wav2vec2-large-xlsr-53-th"
    base_output_dir = './models/grid_search'
    
    # Create searcher
    searcher = SimpleGridSearch(csv_file, model_name, base_output_dir)
    
    # Run search
    best_params, best_score = searcher.search(max_combinations=15)  # Test 15 combinations
    
    return best_params, best_score

if __name__ == "__main__":
    best_params, best_score = run_simple_grid_search()
    print(f"\nFinal best F1 score: {best_score:.4f}")
    print("Final best parameters:", best_params)

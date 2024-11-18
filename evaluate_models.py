import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from datasets import Dataset
import pandas as pd

# Import necessary functions from your existing files
from fine_tune_wav2vec2_multi import (
    load_dataset,
    prepare_dataset,
    evaluate_all_folds
)

def evaluate_model(model, feature_extractor, test_data):
    predictions = []
    labels = []
    
    for item in test_data:
        # Get input values directly from the prepared dataset
        inputs = {
            'input_values': torch.tensor(item['input_values']).unsqueeze(0),
            'attention_mask': torch.tensor(item['attention_mask']).unsqueeze(0)
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1)
            predictions.append(pred.item())
            labels.append(item['label'])
    
    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)
    
    # Calculate per-class metrics
    class_names = ['none', 'เย็ดแม่', 'กู', 'มึง', 'เหี้ย']
    per_class_metrics = {}
    for i, name in enumerate(class_names):
        class_preds = [p == i for p in predictions]
        class_labels = [l == i for l in labels]
        true_pos = sum(p and l for p, l in zip(class_preds, class_labels))
        total = sum(class_labels)
        if total > 0:
            per_class_metrics[name] = true_pos / total
    
    return {
        'accuracy': accuracy,
        'per_class_metrics': per_class_metrics
    }

def evaluate_all_folds(test_data, num_folds=5, base_dir='./models/fine_tuned_wav2vec2'):
    fold_performances = {}
    
    for fold in range(1, num_folds + 1):
        model_path = f'{base_dir}_fold_{fold}'
        
        # Load model and feature extractor for this fold
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        
        # Evaluate model
        model.eval()
        results = evaluate_model(model, feature_extractor, test_data)
        
        fold_performances[fold] = {
            'accuracy': results['accuracy'],
            'per_class_metrics': results['per_class_metrics'],
            'model_path': model_path
        }
        
        print(f"\nFold {fold} Performance:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print("Per-class metrics:", results['per_class_metrics'])
    
    # Find best performing fold
    best_fold = max(fold_performances.items(), 
                   key=lambda x: x[1]['accuracy'])
    
    print(f"\nBest performing model is Fold {best_fold[0]}")
    print(f"Path: {best_fold[1]['model_path']}")
    print(f"Accuracy: {best_fold[1]['accuracy']:.4f}")
    
    return best_fold[1]['model_path']

if __name__ == "__main__":
    # Load your test dataset
    test_csv = 'test_dataset.csv'  # Your test dataset file
    df = load_dataset(test_csv)
    
    # Initialize feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "airesearch/wav2vec2-large-xlsr-53-th",
        return_attention_mask=True,
        do_normalize=True,
    )
    
    # Prepare test dataset
    test_dataset = prepare_dataset(df, feature_extractor)
    
    # Evaluate all folds and get best model path
    best_model_path = evaluate_all_folds(test_dataset)
    
    print(f"\nYou should use this model for testing: {best_model_path}")

import os
import re
from pathlib import Path

def extract_word_level_metrics(file_path):
    """Extract word-level precision, recall, and F1 from note.txt file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract balanced accuracy
        balanced_acc_match = re.search(r'Balanced Accuracy:\s*([\d.]+)', content)
        balanced_acc = float(balanced_acc_match.group(1)) * 100 if balanced_acc_match else 0
        
        # Extract binary accuracy from confusion matrix section
        accuracy_match = re.search(r'Simple Binary Accuracy:\s*([\d.]+)', content)
        binary_acc = float(accuracy_match.group(1)) * 100 if accuracy_match else 0
        
        # Extract word-level metrics
        word_precision_match = re.search(r'Word-Level Precision:\s*([\d.]+)', content)
        word_precision = float(word_precision_match.group(1)) * 100 if word_precision_match else 0
        
        word_recall_match = re.search(r'Word-Level Recall:\s*([\d.]+)', content)
        word_recall = float(word_recall_match.group(1)) * 100 if word_recall_match else 0
        
        word_f1_match = re.search(r'Word-Level F1-Score:\s*([\d.]+)', content)
        word_f1 = float(word_f1_match.group(1)) * 100 if word_f1_match else 0
        
        return word_precision, word_recall, word_f1, binary_acc, balanced_acc
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, 0, 0, 0, 0

def main():
    base_path = Path("evaluation_results/4_classes")
    results = []
    
    # Define the window/stride combinations we want
    configurations = [
        # 0.3s window
        ("0.3", "0.125"), ("0.3", "0.2"), ("0.3", "0.25"), ("0.3", "0.3"),
        # 0.4s window  
        ("0.4", "0.2"), ("0.4", "0.25"), ("0.4", "0.3"), ("0.4", "0.35"), ("0.4", "0.4"),
        # 0.5s window
        ("0.5", "0.25"), ("0.5", "0.3"), ("0.5", "0.35"), ("0.5", "0.4"), ("0.5", "0.45"), ("0.5", "0.5"),
        # 0.6s window
        ("0.6", "0.3"), ("0.6", "0.35"), ("0.6", "0.4"), ("0.6", "0.45"), ("0.6", "0.5"), ("0.6", "0.55"), ("0.6", "0.6"),
        # 0.7s window
        ("0.7", "0.35"), ("0.7", "0.4"), ("0.7", "0.45"), ("0.7", "0.5"), ("0.7", "0.55"), ("0.7", "0.6"), ("0.7", "0.65"), ("0.7", "0.7"),
        # 0.8s window
        ("0.8", "0.4"), ("0.8", "0.45"), ("0.8", "0.5"), ("0.8", "0.55"), ("0.8", "0.6"), ("0.8", "0.65"), ("0.8", "0.7"), ("0.8", "0.75"), ("0.8", "0.8"),
        # 0.9s window
        ("0.9", "0.45"), ("0.9", "0.5"), ("0.9", "0.55"), ("0.9", "0.6"), ("0.9", "0.65"), ("0.9", "0.7"), ("0.9", "0.75"), ("0.9", "0.8"), ("0.9", "0.85"), ("0.9", "0.9"),
        # 1.0s window
        ("1.0", "0.5"), ("1.0", "0.55"), ("1.0", "0.6"), ("1.0", "0.65"), ("1.0", "0.7"), ("1.0", "0.75"), ("1.0", "0.8"), ("1.0", "0.85"), ("1.0", "0.9"), ("1.0", "1.0"),
    ]
    
    for window, stride in configurations:
        folder_path = base_path / f"window_{window}s" / f"stride_{stride}s" / "note.txt"
        
        if folder_path.exists():
            word_precision, word_recall, word_f1, binary_acc, balanced_acc = extract_word_level_metrics(folder_path)
            results.append((window, stride, word_precision, word_recall, word_f1, binary_acc, balanced_acc))
            print(f"{window} & {stride} & {word_precision:.2f} & {word_recall:.2f} & {word_f1:.2f} & {binary_acc:.2f} & {balanced_acc:.2f} \\\\")
        else:
            print(f"Missing: {folder_path}")
    
    print(f"\nTotal configurations found: {len(results)}")

if __name__ == "__main__":
    main()

import os
import re
from pathlib import Path

def extract_original_gt_metrics(file_path):
    """Extract Original GT-based word-level metrics from note.txt file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract balanced accuracy
        balanced_acc_match = re.search(r'Balanced Binary Accuracy:\s*([\d.]+)', content)
        balanced_acc = float(balanced_acc_match.group(1)) * 100 if balanced_acc_match else 0
        
        # Extract binary accuracy from comprehensive summary section
        binary_acc_match = re.search(r'Simple Binary Accuracy:\s*([\d.]+)', content)
        binary_acc = float(binary_acc_match.group(1)) * 100 if binary_acc_match else 0
        
        # Extract Original GT-based word-level metrics
        # Look for "Word-level F1 (vs original GT): X.XXXX ⭐ (Most realistic)"
        original_gt_f1_match = re.search(r'Word-level F1 \(vs original GT\):\s*([\d.]+)', content)
        original_gt_f1 = float(original_gt_f1_match.group(1)) * 100 if original_gt_f1_match else 0
        
        # Look for precision and recall in the "=== Word-Level vs Original Ground Truth Accuracy ===" section
        precision_vs_original_match = re.search(r'Precision vs Original GT:\s*([\d.]+)', content)
        precision_vs_original = float(precision_vs_original_match.group(1)) * 100 if precision_vs_original_match else 0
        
        recall_vs_original_match = re.search(r'Recall vs Original GT:\s*([\d.]+)', content)
        recall_vs_original = float(recall_vs_original_match.group(1)) * 100 if recall_vs_original_match else 0
        
        return precision_vs_original, recall_vs_original, original_gt_f1, binary_acc, balanced_acc
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
        
        if folder_path.exists() and folder_path.stat().st_size > 0:  # Check if file exists and is not empty
            precision_original_gt, recall_original_gt, f1_original_gt, binary_acc, balanced_acc = extract_original_gt_metrics(folder_path)
            results.append((window, stride, precision_original_gt, recall_original_gt, f1_original_gt, binary_acc, balanced_acc))
            print(f"{window} & {stride} & {precision_original_gt:.2f} & {recall_original_gt:.2f} & {f1_original_gt:.2f} & {binary_acc:.2f} & {balanced_acc:.2f} \\\\")
        else:
            print(f"Missing or empty: {folder_path}")
    
    print(f"\nTotal configurations found: {len(results)}")

if __name__ == "__main__":
    main()

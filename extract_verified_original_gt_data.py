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
        # Look for precision and recall in the "=== Word-Level vs Original Ground Truth Accuracy ===" section
        precision_vs_original_match = re.search(r'Precision vs Original GT:\s*([\d.]+)', content)
        precision_vs_original = float(precision_vs_original_match.group(1)) * 100 if precision_vs_original_match else 0
        
        recall_vs_original_match = re.search(r'Recall vs Original GT:\s*([\d.]+)', content)
        recall_vs_original = float(recall_vs_original_match.group(1)) * 100 if recall_vs_original_match else 0
        
        f1_vs_original_match = re.search(r'F1-Score vs Original GT:\s*([\d.]+)', content)
        f1_vs_original = float(f1_vs_original_match.group(1)) * 100 if f1_vs_original_match else 0
        
        return precision_vs_original, recall_vs_original, f1_vs_original, binary_acc, balanced_acc
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, 0, 0, 0, 0

def extract_all_correct_data():
    """Extract correct data from all actual configurations"""
    base_path = Path("evaluation_results/4_classes")
    
    # Get actual configurations
    actual_configs = [
        # 0.3s window - actual strides
        ("0.3", "0.15"), ("0.3", "0.2"), ("0.3", "0.25"), ("0.3", "0.3"),
        # 0.4s window - actual strides  
        ("0.4", "0.2"), ("0.4", "0.25"), ("0.4", "0.3"), ("0.4", "0.35"), ("0.4", "0.4"),
        # 0.5s window - actual strides
        ("0.5", "0.25"), ("0.5", "0.3"), ("0.5", "0.35"), ("0.5", "0.4"), ("0.5", "0.45"), ("0.5", "0.5"),
        # 0.6s window - actual strides
        ("0.6", "0.3"), ("0.6", "0.35"), ("0.6", "0.4"), ("0.6", "0.45"), ("0.6", "0.5"), ("0.6", "0.55"), ("0.6", "0.6"),
        # 0.7s window - actual strides
        ("0.7", "0.35"), ("0.7", "0.4"), ("0.7", "0.45"), ("0.7", "0.5"), ("0.7", "0.55"), ("0.7", "0.6"), ("0.7", "0.65"), ("0.7", "0.7"),
        # 0.8s window - actual strides (excluding 0.75s which is empty)
        ("0.8", "0.4"), ("0.8", "0.45"), ("0.8", "0.5"), ("0.8", "0.55"), ("0.8", "0.6"), ("0.8", "0.65"), ("0.8", "0.7"), ("0.8", "0.8"),
        # 0.9s window - actual strides
        ("0.9", "0.45"), ("0.9", "0.5"), ("0.9", "0.55"), ("0.9", "0.6"), ("0.9", "0.65"), ("0.9", "0.7"), ("0.9", "0.75"), ("0.9", "0.8"), ("0.9", "0.85"), ("0.9", "0.9"),
        # 1.0s window - actual strides (excluding 0.95s which has no note.txt)
        ("1.0", "0.5"), ("1.0", "0.55"), ("1.0", "0.6"), ("1.0", "0.65"), ("1.0", "0.7"), ("1.0", "0.75"), ("1.0", "0.8"), ("1.0", "0.85"), ("1.0", "0.9"), ("1.0", "1.0"),
    ]
    
    print("=== EXTRACTING ORIGINAL GT METRICS FROM ALL ACTUAL CONFIGURATIONS ===\n")
    
    results = []
    successful_extractions = 0
    
    for window, stride in actual_configs:
        folder_path = base_path / f"window_{window}s" / f"stride_{stride}s" / "note.txt"
        
        if folder_path.exists() and folder_path.stat().st_size > 0:
            precision_original_gt, recall_original_gt, f1_original_gt, binary_acc, balanced_acc = extract_original_gt_metrics(folder_path)
            
            if precision_original_gt > 0 or recall_original_gt > 0 or f1_original_gt > 0:  # Valid data
                results.append((window, stride, precision_original_gt, recall_original_gt, f1_original_gt, binary_acc, balanced_acc))
                print(f"{window} & {stride} & {precision_original_gt:.2f} & {recall_original_gt:.2f} & {f1_original_gt:.2f} & {binary_acc:.2f} & {balanced_acc:.2f} \\\\")
                successful_extractions += 1
            else:
                print(f"❌ {window}s/{stride}s: Could not extract Original GT metrics")
        else:
            print(f"❌ {window}s/{stride}s: Missing or empty file")
    
    print(f"\n=== SUMMARY ===")
    print(f"Attempted configurations: {len(actual_configs)}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {len(actual_configs) - successful_extractions}")
    
    return results

if __name__ == "__main__":
    extract_all_correct_data()

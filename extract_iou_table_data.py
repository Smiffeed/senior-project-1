import os
import re
from pathlib import Path

def extract_iou_metrics(file_path):
    """Extract IoU-based metrics from note.txt file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract balanced accuracy
        balanced_acc_match = re.search(r'Balanced Binary Accuracy:\s*([\d.]+)', content)
        balanced_acc = float(balanced_acc_match.group(1)) * 100 if balanced_acc_match else 0
        
        # Extract binary accuracy from comprehensive summary section
        binary_acc_match = re.search(r'Simple Binary Accuracy:\s*([\d.]+)', content)
        binary_acc = float(binary_acc_match.group(1)) * 100 if binary_acc_match else 0
        
        # Extract IoU-based metrics from the IoU section
        iou_precision_match = re.search(r'IoU-based Precision:\s*([\d.]+)', content)
        iou_precision = float(iou_precision_match.group(1)) * 100 if iou_precision_match else 0
        
        iou_recall_match = re.search(r'IoU-based Recall:\s*([\d.]+)', content)
        iou_recall = float(iou_recall_match.group(1)) * 100 if iou_recall_match else 0
        
        iou_f1_match = re.search(r'IoU-based F1-Score:\s*([\d.]+)', content)
        iou_f1 = float(iou_f1_match.group(1)) * 100 if iou_f1_match else 0
        
        return iou_precision, iou_recall, iou_f1, binary_acc, balanced_acc
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, 0, 0, 0, 0

def get_actual_configurations():
    """Get the actual window/stride configurations that exist"""
    base_path = Path("evaluation_results/4_classes")
    configurations = []
    
    # Get all window directories
    for window_dir in sorted(base_path.iterdir()):
        if window_dir.is_dir() and window_dir.name.startswith("window_"):
            window = window_dir.name.replace("window_", "").replace("s", "")
            
            # Get all stride directories for this window
            for stride_dir in sorted(window_dir.iterdir()):
                if stride_dir.is_dir() and stride_dir.name.startswith("stride_"):
                    stride = stride_dir.name.replace("stride_", "").replace("s", "")
                    
                    note_file = stride_dir / "note.txt"
                    if note_file.exists() and note_file.stat().st_size > 0:
                        configurations.append((window, stride))
    
    return configurations

def main():
    configurations = get_actual_configurations()
    results = []
    
    print("Extracting IoU-based metrics from all configurations...")
    print("Format: Window & Stride & IoU Precision & IoU Recall & IoU F1 & Binary Acc & Balanced Acc")
    print()
    
    for window, stride in configurations:
        folder_path = Path("evaluation_results/4_classes") / f"window_{window}s" / f"stride_{stride}s" / "note.txt"
        
        iou_precision, iou_recall, iou_f1, binary_acc, balanced_acc = extract_iou_metrics(folder_path)
        results.append((window, stride, iou_precision, iou_recall, iou_f1, binary_acc, balanced_acc))
        print(f"{window} & {stride} & {iou_precision:.2f} & {iou_recall:.2f} & {iou_f1:.2f} & {binary_acc:.2f} & {balanced_acc:.2f} \\\\")
    
    print(f"\nTotal configurations found: {len(results)}")
    
    # Group by window size for table formatting
    print("\n=== GROUPED BY WINDOW SIZE FOR TABLE ===")
    current_window = None
    for window, stride, iou_precision, iou_recall, iou_f1, binary_acc, balanced_acc in results:
        if window != current_window:
            if current_window is not None:
                print("        \\hline")
            print(f"        \\multicolumn{{7}}{{|c|}}{{\\textbf{{{window}s Window Configurations}}}} \\\\")
            print("        \\hline")
            current_window = window
        
        print(f"        {window} & {stride} & {iou_precision:.2f} & {iou_recall:.2f} & {iou_f1:.2f} & {binary_acc:.2f} & {balanced_acc:.2f} \\\\")

if __name__ == "__main__":
    main()

import os
import re
import csv

def extract_binary_metrics(note_file_path):
    """Extract binary classification metrics from a note.txt file"""
    try:
        with open(note_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract window and stride from path
        path_parts = note_file_path.replace('\\', '/').split('/')
        window_folder = [part for part in path_parts if part.startswith('window_')][0]
        stride_folder = [part for part in path_parts if part.startswith('stride_')][0]
        
        window = window_folder.replace('window_', '').replace('s', '')
        stride = stride_folder.replace('stride_', '').replace('s', '')
        
        # Extract Simple Binary Accuracy
        simple_accuracy_match = re.search(r'Simple Binary Accuracy: ([\d.]+)', content)
        simple_accuracy = float(simple_accuracy_match.group(1)) if simple_accuracy_match else None
        
        # Extract Balanced Binary Accuracy
        balanced_accuracy_match = re.search(r'Balanced Binary Accuracy: ([\d.]+)', content)
        balanced_accuracy = float(balanced_accuracy_match.group(1)) if balanced_accuracy_match else None
        
        # Extract binary classification metrics (looking for the second set which seems more accurate)
        # Find the section with "Detailed Counts:"
        detailed_section = re.search(r'Detailed Counts:(.*?)🎯', content, re.DOTALL)
        if detailed_section:
            detailed_text = detailed_section.group(1)
            
            # Extract TP, FP, TN, FN
            tp_match = re.search(r'True Positives: (\d+)', detailed_text)
            fp_match = re.search(r'False Positives: (\d+)', detailed_text)
            tn_match = re.search(r'True Negatives: (\d+)', detailed_text)
            fn_match = re.search(r'False Negatives: (\d+)', detailed_text)
            
            if all([tp_match, fp_match, tn_match, fn_match]):
                tp = int(tp_match.group(1))
                fp = int(fp_match.group(1))
                tn = int(tn_match.group(1))
                fn = int(fn_match.group(1))
                
                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                
                return {
                    'window': window,
                    'stride': stride,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'balanced_accuracy': balanced_accuracy,
                    'simple_accuracy': simple_accuracy
                }
    
    except Exception as e:
        print(f"Error processing {note_file_path}: {e}")
    
    return None

def main():
    # Base directory
    base_dir = r"c:\Users\muldi\Documents\Playground\University\senior-project-1\evaluation_results\4_classes"
    
    results = []
    
    # Walk through all directories
    for root, dirs, files in os.walk(base_dir):
        if 'note.txt' in files:
            note_path = os.path.join(root, 'note.txt')
            metrics = extract_binary_metrics(note_path)
            if metrics:
                results.append(metrics)
    
    # Sort by window then stride
    results.sort(key=lambda x: (float(x['window']), float(x['stride'])))
    
    # Write to CSV
    output_file = r"c:\Users\muldi\Documents\Playground\University\senior-project-1\binary_classification_metrics.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['window', 'stride', 'accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy', 'simple_accuracy'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Binary classification metrics extracted to: {output_file}")
    print(f"Total configurations processed: {len(results)}")
    
    # Print first few results for verification
    print("\nFirst 5 results:")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. Window {result['window']}s, Stride {result['stride']}s:")
        print(f"   Accuracy: {result['accuracy']:.4f}, Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, F1: {result['f1']:.4f}")

if __name__ == "__main__":
    main()

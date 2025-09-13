import os
import re

def calculate_and_add_balanced_accuracy(root_dir):
    """
    Scans for note.txt files, calculates balanced accuracy from the
    'Binary Classification Metrics' section, and adds or replaces it in that section.
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == 'note.txt':
                file_path = os.path.join(subdir, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Regex to find the specific metrics section
                    header_pattern = r"=== Binary Classification Metrics \(Threshold: [\d.]+\) ==="
                    section_pattern = re.compile(rf"({header_pattern}\s*\n)(.*?)(\n\n|\Z)", re.DOTALL)
                    section_match = section_pattern.search(content)

                    if not section_match:
                        print(f"Warning: Could not find 'Binary Classification Metrics' section in {file_path}. Skipping.")
                        continue
                    
                    full_header = section_match.group(1)
                    metrics_text = section_match.group(2)
                    
                    # Extract the four core values from the metrics text
                    tp_match = re.search(r"^\s*True Positives: (\d+)", metrics_text, re.MULTILINE)
                    fn_match = re.search(r"^\s*False Negatives: (\d+)", metrics_text, re.MULTILINE)
                    tn_match = re.search(r"^\s*True Negatives: (\d+)", metrics_text, re.MULTILINE)
                    fp_match = re.search(r"^\s*False Positives: (\d+)", metrics_text, re.MULTILINE)

                    if not all([tp_match, fn_match, tn_match, fp_match]):
                        print(f"Warning: Could not extract all confusion matrix values from {file_path}. Skipping.")
                        continue

                    tp = int(tp_match.group(1))
                    fn = int(fn_match.group(1))
                    tn = int(tn_match.group(1))
                    fp = int(fp_match.group(1))

                    # Calculate Balanced Accuracy
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
                    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
                    balanced_accuracy = (tpr + tnr) / 2
                    new_balanced_accuracy_line = f"Balanced Accuracy: {balanced_accuracy:.4f}"

                    # Check if 'Balanced Accuracy' already exists and replace it, otherwise insert it.
                    if "Balanced Accuracy:" in metrics_text:
                        # Replace the existing line
                        new_metrics_text = re.sub(
                            r"^\s*Balanced Accuracy: [\d.]+",
                            new_balanced_accuracy_line,
                            metrics_text,
                            flags=re.MULTILINE
                        )
                        action = "REPLACED"
                    else:
                        # Insert the new line after the 'Accuracy:' line
                        new_metrics_text = re.sub(
                            r"(^\s*Accuracy: [\d.]+)",
                            f"\\1\n{new_balanced_accuracy_line}",
                            metrics_text,
                            flags=re.MULTILINE
                        )
                        action = "ADDED"

                    # Reconstruct the full content with the updated metrics section
                    new_content = content.replace(metrics_text, new_metrics_text)

                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Successfully calculated and {action} balanced accuracy in {file_path}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == '__main__':
    search_directory = 'evaluation_results/eval_window'
    print(f"Starting to calculate and update 'note.txt' files in '{search_directory}'...")
    calculate_and_add_balanced_accuracy(search_directory)
    print("Update process complete.")

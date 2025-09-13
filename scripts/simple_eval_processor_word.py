#!/usr/bin/env python3
"""
Simple eval_percent processor with clean output filtering - Word Level Version
Uses evaluate_advanced_models_word.py for word-level evaluation with merging
"""

import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

def filter_evaluation_output(raw_output):
    """Filter the raw output to keep only important classification results"""
    lines = raw_output.split('\n')
    filtered_lines = []
    
    # Skip processing progress and warnings
    skip_patterns = [
        "Processed ",
        "UserWarning:",
        "warnings.warn",
        "torchaudio._backend",
        "TorchCodec",
        "C:\\Users\\",
        "Using binary classification threshold:",
        "Using CSV file:",
        "📊 This script will generate:",
        "💡 Try different options:",
        "Loaded HuggingFace",
        "Using device:",
        "Using ADVANCED preprocessing",
        "Loading evaluation data",
        "✅ Loaded",
        "📏 Window configuration:",
        "env\\lib\\site-packages"
    ]
    
    # Important sections to keep
    keep_sections = [
        "=== Classification Report",
        "              precision    recall  f1-score   support",
        "        เย็ด",
        "          กู",
        "         มึง", 
        "       เหี้ย",
        "    accuracy",
        "   macro avg",
        "weighted avg",
        "Overall Accuracy",
        "Total Profanity Windows",
        "Correct Predictions:",
        "Missed Profanity",
        "=== Binary Profanity Detection Metrics",
        "Using threshold:",
        "Balanced Accuracy:",
        "=== Creating Binary Confusion Matrix",
        "✅ Binary confusion matrix saved",
        "=== Binary Classification Metrics",
        "True Negatives:",
        "False Positives:",
        "False Negatives:",
        "True Positives:",
        "Accuracy:",
        "Precision:",
        "Recall:",
        "F1-Score:",
        "F1-score:",
        "Detailed Counts:",
        "Total Windows:",
        "🎯 BINARY CLASSIFICATION ACCURACY:",
        "Simple Binary Accuracy:",
        "Balanced Binary Accuracy:",
        "(Balanced accounts for class imbalance:",
        "Class Distribution:",
        "Profanity samples:",
        "None samples:",
        "=== ROC/AUC",
        "AUC Score:",
        "Current threshold:",
        "✅ ROC curve saved",
        "=== DET Curve",
        "✅ DET curve saved",
        "=== MULTI-CLASS PROFANITY",
        "Total Profanity Samples:",
        "Correct Profanity Classifications:",
        "Simple Profanity Classification Accuracy:",
        "Per-Class Profanity Accuracy:",
        "Balanced Profanity Classification Accuracy:",
        "============================================================",
        "NORMALIZED WORD-LEVEL EVALUATION",
        "=== Word-Level Detection Metrics",
        "Total True Words:",
        "Total Predicted Words:",
        "Correct Word Detections:",
        "Word-Level Precision:",
        "Word-Level Recall:",
        "Word-Level F1-Score:",
        "=== Word-Level vs Original Ground Truth",
        "Precision vs Original GT:",
        "Recall vs Original GT:",
        "F1-Score vs Original GT:",
        "🎯 ACCURACY COMPARISON:",
        "Windowed-based Word F1:",
        "Original GT-based Word F1:",
        "Difference:",
        "=== Class-wise Performance vs Original Ground Truth",
        "=== Comparison: Window vs Word Level",
        "Window-level F1:",
        "Word-level F1:",
        "Original GT Word F1:",
        "Window vs Word difference:",
        "Word vs Original GT difference:",
        "📊 Original GT-based metrics",
        "======================================================================",
        "🎯 COMPREHENSIVE ACCURACY SUMMARY",
        "1️⃣ BINARY CLASSIFICATION ACCURACY",
        "2️⃣ MULTI-CLASS CLASSIFICATION ACCURACY",
        "3️⃣ WORD-LEVEL ACCURACY",
        "📊 KEY INSIGHTS:",
        "• Binary detection handles",
        "• Profanity classification handles", 
        "• Word-level performance",
        "• Data imbalance:",
        "• Current threshold:",
        "🎯 THRESHOLD GUIDANCE:",
        "• Lower threshold",
        "• Higher threshold", 
        "• Balanced threshold",
        "• Use ROC curve",
        "Word-level F1 (vs windowed GT):",
        "Word-level F1 (vs original GT):",
        "⭐ (Most realistic)",
        "🎯 THRESHOLD GUIDANCE:",
        "=== Detailed Word/Instance Statistics",
        "Total Label Instances",
        "Total Real Words",
        "Reduction ratio:",
        "=== Breakdown by Class",
        "Format:",
        "📊 SUMMARY COMPARISON:",
        "Original Ground Truth:",
        "Windowed Ground Truth:",
        "Predicted Instances:",
        "Merged Words:",
        # Add patterns to capture the actual breakdown data lines
        "เย็ด:",
        "กู:",
        "มึง:",
        "เหี้ย:",
        "หี:",
        "แตด:",
        "ควย:",
        "สวะ:",
        "(Showing original ground truth",
        "(Showing windowed ground truth",
        "(Could not load",
        "→",  # Arrow character used in breakdown format
        "predicted →",
        "words",
        "profanity instances",
        "profanity words",
        "=== Prediction Statistics",
        "=== Advanced Preprocessing Benefits",
        "✅ Enhanced noise reduction",
        "✅ RMS normalization",
        "✅ Dynamic range compression",
        "✅ High-pass filtering",
        "✅ Consistent preprocessing pipeline",
        "=== Generating Word-Level Confusion Matrix",
        "Creating word-level confusion matrix",
        "Note: Only showing detected words",
        "✅ Word-level confusion matrix saved",
        "=== Generating Window-Level Confusion Matrix",
        "✅ Window-level confusion matrix saved"
    ]
    
    for line in lines:
        # Skip unwanted lines
        should_skip = any(pattern in line for pattern in skip_patterns)
        if should_skip:
            continue
            
        # For evaluation output, be less aggressive - keep most content
        # Only skip empty lines in excessive amounts
        if line.strip():
            filtered_lines.append(line)
        elif filtered_lines and filtered_lines[-1].strip():  # Only add empty line if previous line wasn't empty
            filtered_lines.append("")
    
    return '\n'.join(filtered_lines)

def process_single_file(csv_file, model_path, threshold, output_base_dir):
    """Process a single CSV file and organize outputs with clean note.txt"""
    
    # Parse filename to get window and stride info
    csv_path = Path(csv_file)
    window_dir = csv_path.parent.name  # e.g., "window_0.3s"
    stride_file = csv_path.stem        # e.g., "stride_10.0%"
    
    # Create output directory
    output_dir = Path(output_base_dir) / window_dir / stride_file
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing: {csv_file}")
    print(f"Output: {output_dir}")
    
    # Use unique plots directory for parallel processing, or default for single processing
    worker_plots_dir = os.environ.get('WORKER_PLOTS_DIR', './plots')
    plots_dir = Path(worker_plots_dir)
    
    # Clean plots directory
    if plots_dir.exists():
        for file in plots_dir.glob("*.png"):
            file.unlink()
    else:
        plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create note.txt file
    note_file = output_dir / "note.txt"
    
    # Run evaluation using evaluate_advanced_models_word.py
    python_exe = "C:/Users/muldi/Documents/Playground/University/senior-project-1/env/Scripts/python.exe"
    cmd = [
        python_exe,
        "scripts/evaluate_advanced_models_word.py",
        "--csv_file", str(csv_file),
        "--model_path", str(model_path),
        "--threshold", str(threshold)
    ]
    
    print(f"Running word-level evaluation and capturing clean output...")
    print(f"Clean output will be saved to: {note_file}")
    
    # Write header to note file
    header = f"""Word-Level Evaluation Results
=============================
CSV File: {csv_file}
Model Path: {model_path}
Threshold: {threshold}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Evaluation Type: Word-Level with Window Merging (No IoU)
Command: {' '.join(cmd)}

{'='*80}

"""
    
    try:
        # Capture all output to a temporary location first
        print("Running word-level evaluation (this may take a while)...")
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            cwd=os.getcwd(),
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )
        
        if result.returncode == 0:
            print(f"Word-level evaluation completed successfully")
            
            # Filter the output to keep only important parts
            raw_output = result.stdout
            filtered_output = filter_evaluation_output(raw_output)
            
            # Write the clean output to note.txt
            with open(note_file, 'w', encoding='utf-8') as f:
                f.write(header)
                f.write(filtered_output)
                f.write(f"\n\n=== Word-Level Evaluation completed successfully ===\n")
                f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # Move plot files
                moved_files = []
                if plots_dir.exists():
                    for plot_file in plots_dir.glob("*.png"):
                        dest_file = output_dir / plot_file.name
                        shutil.move(str(plot_file), str(dest_file))
                        moved_files.append(plot_file.name)
                        print(f"Moved: {plot_file.name}")
                
                # Add file organization summary
                f.write(f"\n=== File Organization Summary ===\n")
                f.write(f"Output Directory: {output_dir}\n")
                f.write(f"Moved Plot Files: {len(moved_files)}\n")
                for file in moved_files:
                    f.write(f"  - {file}\n")
                
                f.write(f"\n=== Word-Level Evaluation Features ===\n")
                f.write(f"✅ Window-level processing with advanced preprocessing\n")
                f.write(f"✅ Word merging from consecutive windows\n")
                f.write(f"✅ Word-level vs original ground truth comparison\n")
                f.write(f"✅ Comprehensive accuracy metrics\n")
                f.write(f"✅ Binary and multi-class classification\n")
                f.write(f"✅ No IoU processing (simplified evaluation)\n")
                
                print(f"Clean word-level evaluation output saved to: {note_file}")
                print(f"Moved {len(moved_files)} plot files")
                return True
        else:
            print(f"Word-level evaluation failed with code: {result.returncode}")
            with open(note_file, 'w', encoding='utf-8') as f:
                f.write(header)
                f.write(f"=== Word-Level Evaluation failed with return code: {result.returncode} ===\n")
                f.write(result.stdout)
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Word-level evaluation failed: {e}")
        with open(note_file, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(f"=== ERROR ===\n{str(e)}\n")
        return False
    except KeyboardInterrupt:
        print(f"Word-level evaluation interrupted by user")
        with open(note_file, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(f"=== INTERRUPTED BY USER ===\n")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        with open(note_file, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write(f"=== UNEXPECTED ERROR ===\n{str(e)}\n")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple eval_percent processor with clean output - Word Level")
    parser.add_argument("--csv_file", required=True, help="CSV file to process")
    parser.add_argument("--model_path", default="./models/4_classes_max_steps", help="Model path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold")
    parser.add_argument("--output_dir", default="./evaluation_results/eval_percent", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Simple eval_percent processor (clean output) - Word Level")
    print(f"File: {args.csv_file}")
    print(f"Model: {args.model_path}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {args.output_dir}")
    print(f"Evaluation: Word-level with window merging (no IoU)")
    
    success = process_single_file(
        args.csv_file,
        args.model_path, 
        args.threshold,
        args.output_dir
    )
    
    if success:
        print(f"Word-level processing completed successfully!")
    else:
        print(f"Word-level processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

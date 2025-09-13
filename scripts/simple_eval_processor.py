#!/usr/bin/env python3
"""
Simple eval_percent processor without complex output capturing
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

def process_single_file(csv_file, model_path, threshold, output_base_dir):
    """Process a single CSV file and organize outputs"""
    
    # Parse filename to get window and stride info
    csv_path = Path(csv_file)
    window_dir = csv_path.parent.name  # e.g., "window_0.3s"
    stride_file = csv_path.stem        # e.g., "stride_10.0%"
    
    # Create output directory
    output_dir = Path(output_base_dir) / window_dir / stride_file
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 Processing: {csv_file}")
    print(f"📁 Output: {output_dir}")
    
    # Clean plots directory
    plots_dir = Path("./plots")
    if plots_dir.exists():
        for file in plots_dir.glob("*.png"):
            file.unlink()
    
    # Run evaluation
    python_exe = "C:/Users/muldi/Documents/Playground/University/senior-project-1/env/Scripts/python.exe"
    cmd = [
        python_exe,
        "scripts/evaluate_advanced_models_seperate.py",
        "--csv_file", str(csv_file),
        "--model_path", str(model_path),
        "--threshold", str(threshold)
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    # Create note.txt file to capture all output
    note_file = output_dir / "note.txt"
    
    # Run the evaluation and capture output to both console and file
    try:
        with open(note_file, 'w', encoding='utf-8') as f:
            # Write header information
            f.write(f"Evaluation Results\n")
            f.write(f"==================\n")
            f.write(f"CSV File: {csv_file}\n")
            f.write(f"Model Path: {model_path}\n")
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"\n" + "="*80 + "\n\n")
            f.flush()
            
            # Run the process and capture both stdout and stderr
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=os.getcwd(),
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            )
            
            # Read and write output in real-time
            for line in process.stdout:
                print(line, end='')  # Print to console
                f.write(line)       # Write to file
                f.flush()           # Ensure immediate writing
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                print(f"✅ Evaluation completed successfully")
                f.write(f"\n\n=== Evaluation completed successfully ===\n")
                
                # Move plot files
                moved_files = []
                if plots_dir.exists():
                    for plot_file in plots_dir.glob("*.png"):
                        dest_file = output_dir / plot_file.name
                        shutil.move(str(plot_file), str(dest_file))
                        moved_files.append(plot_file.name)
                        print(f"📊 Moved: {plot_file.name}")
                
                # Add file organization summary to note
                f.write(f"\n\n=== File Organization Summary ===\n")
                f.write(f"Output Directory: {output_dir}\n")
                f.write(f"Moved Plot Files: {len(moved_files)}\n")
                for file in moved_files:
                    f.write(f"  - {file}\n")
                f.write(f"Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                print(f"📄 All output saved to: {note_file}")
                return True
            else:
                print(f"❌ Evaluation failed with code: {return_code}")
                f.write(f"\n\n=== Evaluation failed with return code: {return_code} ===\n")
                return False
                
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
        # Write error to note file if it exists
        try:
            with open(note_file, 'a', encoding='utf-8') as f:
                f.write(f"\n\n=== ERROR ===\n{str(e)}\n")
        except:
            pass
        return False
    except KeyboardInterrupt:
        print(f"⏹️  Evaluation interrupted by user")
        # Write interruption to note file if it exists
        try:
            with open(note_file, 'a', encoding='utf-8') as f:
                f.write(f"\n\n=== INTERRUPTED BY USER ===\n")
        except:
            pass
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        # Write error to note file if it exists
        try:
            with open(note_file, 'a', encoding='utf-8') as f:
                f.write(f"\n\n=== UNEXPECTED ERROR ===\n{str(e)}\n")
        except:
            pass
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple eval_percent processor")
    parser.add_argument("--csv_file", required=True, help="CSV file to process")
    parser.add_argument("--model_path", default="./models/4_classes_max_steps", help="Model path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold")
    parser.add_argument("--output_dir", default="./evaluation_results/eval_percent", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"🎯 Simple eval_percent processor")
    print(f"📄 File: {args.csv_file}")
    print(f"📁 Model: {args.model_path}")
    print(f"🎯 Threshold: {args.threshold}")
    print(f"📂 Output: {args.output_dir}")
    
    success = process_single_file(
        args.csv_file,
        args.model_path, 
        args.threshold,
        args.output_dir
    )
    
    if success:
        print(f"✅ Processing completed successfully!")
    else:
        print(f"❌ Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

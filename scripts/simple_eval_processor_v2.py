#!/usr/bin/env python3
"""
Simple eval_percent processor with output redirection
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
    
    # Create note.txt file
    note_file = output_dir / "note.txt"
    
    # Run evaluation
    python_exe = "C:/Users/muldi/Documents/Playground/University/senior-project-1/env/Scripts/python.exe"
    cmd = [
        python_exe,
        "scripts/evaluate_advanced_models_seperate.py",
        "--csv_file", str(csv_file),
        "--model_path", str(model_path),
        "--threshold", str(threshold)
    ]
    
    print(f"🚀 Running evaluation and capturing output...")
    print(f"📄 Output will be saved to: {note_file}")
    
    # Write header to note file
    with open(note_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"==================\n")
        f.write(f"CSV File: {csv_file}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"\n" + "="*80 + "\n\n")
    
    # Run the evaluation and redirect all output to file
    try:
        with open(note_file, 'a', encoding='utf-8') as f:
            print("⏳ Running evaluation (this may take a while)...")
            result = subprocess.run(
                cmd, 
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                cwd=os.getcwd(),
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
            )
            
            if result.returncode == 0:
                print(f"✅ Evaluation completed successfully")
                
                # Add completion note
                f.write(f"\n\n=== Evaluation completed successfully ===\n")
                f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # Move plot files
                moved_files = []
                if plots_dir.exists():
                    for plot_file in plots_dir.glob("*.png"):
                        dest_file = output_dir / plot_file.name
                        shutil.move(str(plot_file), str(dest_file))
                        moved_files.append(plot_file.name)
                        print(f"📊 Moved: {plot_file.name}")
                
                # Add file organization summary
                f.write(f"\n=== File Organization Summary ===\n")
                f.write(f"Output Directory: {output_dir}\n")
                f.write(f"Moved Plot Files: {len(moved_files)}\n")
                for file in moved_files:
                    f.write(f"  - {file}\n")
                
                print(f"📄 All evaluation output saved to: {note_file}")
                print(f"📊 Moved {len(moved_files)} plot files")
                return True
            else:
                print(f"❌ Evaluation failed with code: {result.returncode}")
                f.write(f"\n\n=== Evaluation failed with return code: {result.returncode} ===\n")
                return False
                
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
        with open(note_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n=== ERROR ===\n{str(e)}\n")
        return False
    except KeyboardInterrupt:
        print(f"⏹️  Evaluation interrupted by user")
        with open(note_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n=== INTERRUPTED BY USER ===\n")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        with open(note_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n=== UNEXPECTED ERROR ===\n{str(e)}\n")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple eval_percent processor with output capture")
    parser.add_argument("--csv_file", required=True, help="CSV file to process")
    parser.add_argument("--model_path", default="./models/4_classes_max_steps", help="Model path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold")
    parser.add_argument("--output_dir", default="./evaluation_results/eval_percent", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"🎯 Simple eval_percent processor (with output capture)")
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

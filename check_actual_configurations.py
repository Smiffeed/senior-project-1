import os
from pathlib import Path

def check_actual_configurations():
    """Check what window/stride configurations actually exist"""
    base_path = Path("evaluation_results/4_classes")
    
    if not base_path.exists():
        print(f"Base path {base_path} does not exist")
        return
    
    all_configs = []
    
    # Get all window folders
    window_folders = sorted([f for f in base_path.iterdir() if f.is_dir() and f.name.startswith("window_")])
    
    for window_folder in window_folders:
        window_size = window_folder.name.replace("window_", "").replace("s", "")
        
        # Get all stride folders for this window
        stride_folders = sorted([f for f in window_folder.iterdir() if f.is_dir() and f.name.startswith("stride_")])
        
        print(f"\n=== Window {window_size}s ===")
        window_configs = []
        
        for stride_folder in stride_folders:
            stride_size = stride_folder.name.replace("stride_", "").replace("s", "")
            note_file = stride_folder / "note.txt"
            
            # Check if note.txt exists and has content
            if note_file.exists() and note_file.stat().st_size > 0:
                status = "✅ Has data"
                all_configs.append((window_size, stride_size))
                window_configs.append(stride_size)
            elif note_file.exists():
                status = "❌ Empty file"
            else:
                status = "❌ No note.txt"
            
            print(f"  Stride {stride_size}s: {status}")
        
        print(f"  Valid strides for {window_size}s: {window_configs}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total valid configurations: {len(all_configs)}")
    print("All valid window/stride combinations:")
    for window, stride in all_configs:
        print(f"  {window}s / {stride}s")
    
    return all_configs

if __name__ == "__main__":
    check_actual_configurations()

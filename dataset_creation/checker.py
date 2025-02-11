import pandas as pd
import os

def check_missing_files(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Get unique file paths
    unique_files = df['File Name'].unique()
    
    # Store results
    missing_files = []
    existing_files = []
    
    # Check each file
    for file_path in unique_files:
        # Normalize path (replace backslashes with forward slashes)
        normalized_path = file_path.replace('\\', '/')
        
        if not os.path.exists(normalized_path):
            missing_files.append(normalized_path)
        else:
            existing_files.append(normalized_path)
    
    # Print results
    print(f"\nTotal unique files: {len(unique_files)}")
    print(f"Existing files: {len(existing_files)}")
    print(f"Missing files: {len(missing_files)}")
    
    if missing_files:
        print("\nMissing files:")
        for file in missing_files:
            print(f"- {file}")
    else:
        print("\nAll files exist!")
    
    return missing_files, existing_files

# Usage
missing_files, existing_files = check_missing_files('./csv/profanity_dataset_word.csv')

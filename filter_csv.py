import pandas as pd
import os

def filter_csv_labels(input_file, output_file, labels_to_change):
    """
    Filter CSV file to change specific labels to 'none'
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Show initial counts
    print(f"Initial CSV file: {input_file}")
    print("Initial label distribution:")
    print(df['label'].value_counts().sort_index())
    print(f"Total rows: {len(df)}")
    
    # Change the unwanted labels to 'none'
    df_filtered = df.copy()
    changed_rows = df_filtered['label'].isin(labels_to_change)
    df_filtered.loc[changed_rows, 'label'] = 'none'
    
    # Show final counts
    print(f"\nAfter changing labels {labels_to_change} to 'none':")
    print("Final label distribution:")
    print(df_filtered['label'].value_counts().sort_index())
    print(f"Total rows: {len(df_filtered)} (unchanged)")
    print(f"Changed rows: {changed_rows.sum()}")
    
    # Save the filtered CSV
    df_filtered.to_csv(output_file, index=False)
    print(f"\nFiltered data saved to: {output_file}")

if __name__ == "__main__":
    # Define the labels to change to 'none'
    labels_to_change = ['ควย', 'สวะ', 'หี', 'แตด']
    
    # Define input and output files
    input_file = "csv/eval.csv"
    output_file = "csv/eval_filtered.csv"

    # Filter the CSV (change labels instead of removing rows)
    filter_csv_labels(input_file, output_file, labels_to_change)
    
    # Also create a backup of the original file
    backup_file = "csv/train_original_backup.csv"
    if not os.path.exists(backup_file):
        import shutil
        shutil.copy2(input_file, backup_file)
        print(f"Original file backed up to: {backup_file}")

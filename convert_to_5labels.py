import pandas as pd
import os
import argparse
import sys

def change_labels_to_none(input_file, output_file, labels_to_change):
    """
    Change specific labels to 'none' in CSV file
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Show initial counts
    print(f"Processing CSV file: {input_file}")
    print("Initial label distribution:")
    
    # Check what label column name is used
    label_columns = [col for col in df.columns if 'label' in col.lower()]
    print(f"Available label columns: {label_columns}")
    
    if not label_columns:
        print("❌ No label column found in the CSV file")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Use the first label column found
    label_col = label_columns[0]
    print(f"Using label column: '{label_col}'")
    
    print(df[label_col].value_counts().sort_index())
    print(f"Total rows: {len(df)}")
    
    # Change the unwanted labels to 'none'
    df_filtered = df.copy()
    changed_rows = df_filtered[label_col].isin(labels_to_change)
    df_filtered.loc[changed_rows, label_col] = 'none'
    
    # Show final counts
    print(f"\nAfter changing labels {labels_to_change} to 'none':")
    print("Final label distribution:")
    print(df_filtered[label_col].value_counts().sort_index())
    print(f"Total rows: {len(df_filtered)} (unchanged)")
    print(f"Changed rows: {changed_rows.sum()}")
    
    # Save the filtered CSV
    df_filtered.to_csv(output_file, index=False)
    print(f"\n✅ Processed data saved to: {output_file}")

def process_single_file(input_file, output_file=None):
    """Process a single CSV file"""
    # Define the labels to change to 'none' (keep only your 5 labels)
    labels_to_change = ['ควย', 'สวะ', 'หี', 'แตด']
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        extension = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_5labels{extension}"
    
    print(f"🏷️ Converting single file to 5-label format")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Labels to change: {labels_to_change}")
    print()
    
    # Process the file
    change_labels_to_none(input_file, output_file, labels_to_change)
    return True

def process_all_windowed_files():
    """Process all windowed CSV files"""
    # Define the labels to change to 'none' (keep only your 5 labels)
    labels_to_change = ['ควย', 'สวะ', 'หี', 'แตด']
    
    # Find all windowed CSV files
    csv_dir = "csv"
    windowed_files = []
    
    if os.path.exists(csv_dir):
        for file in os.listdir(csv_dir):
            if file.startswith("eval_windowed") and file.endswith(".csv"):
                windowed_files.append(file)
    
    if not windowed_files:
        print("❌ No windowed CSV files found")
        return
    
    print(f"Found {len(windowed_files)} windowed files: {windowed_files}")
    
    # Process each file
    for file in windowed_files:
        input_file = f"{csv_dir}/{file}"
        output_file = f"{csv_dir}/{file.replace('.csv', '_5labels.csv')}"
        
        print(f"\n{'='*50}")
        print(f"Processing: {file}")
        print(f"{'='*50}")
        
        change_labels_to_none(input_file, output_file, labels_to_change)

if __name__ == "__main__":
    print("🏷️ CSV Label Converter - Change Labels to 'none'")
    print("This script converts unwanted labels to 'none' to create 5-label datasets")
    print()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert CSV labels to 5-label format')
    parser.add_argument('--input', '-i', type=str, help='Input CSV file path')
    parser.add_argument('--output', '-o', type=str, help='Output CSV file path (optional)')
    parser.add_argument('--all', action='store_true', help='Process all windowed files in csv/ directory')
    
    args = parser.parse_args()
    
    # Determine what to do based on arguments
    if args.input:
        # Process single file
        success = process_single_file(args.input, args.output)
        if success:
            print(f"\n✅ Single file processing complete!")
        else:
            print(f"\n❌ Single file processing failed!")
            sys.exit(1)
            
    elif args.all:
        # Process all windowed files
        process_all_windowed_files()
        print(f"\n✅ All processing complete!")
        
    else:
        # No arguments provided - show usage examples
        print("Usage Examples:")
        print()
        print("1. Convert a single file:")
        print("   python convert_to_5labels.py --input csv/eval.csv")
        print("   python convert_to_5labels.py --input csv/eval.csv --output csv/eval_clean.csv")
        print()
        print("2. Convert all windowed files:")
        print("   python convert_to_5labels.py --all")
        print()
        print("3. Convert any CSV file:")
        print("   python convert_to_5labels.py --input path/to/your/file.csv")
        print()
        
        # Ask user what they want to do
        choice = input("Choose an option:\n1. Single file\n2. All windowed files\n3. Exit\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            input_file = input("Enter input CSV file path: ").strip()
            output_file = input("Enter output file path (or press Enter for auto-name): ").strip()
            if not output_file:
                output_file = None
            process_single_file(input_file, output_file)
            
        elif choice == "2":
            process_all_windowed_files()
            print(f"\n✅ All processing complete!")
            
        else:
            print("Exiting...")
            sys.exit(0)
    
    print("Use the generated *_5labels.csv files with your evaluation script")

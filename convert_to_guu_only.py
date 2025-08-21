import pandas as pd
import os
import argparse
import sys

def convert_to_guu_only(input_file, output_file, keep_none=True):
    """
    Convert all labels to 'none' except 'กู' word
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        keep_none: If True, keep existing 'none' labels. If False, convert them too.
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
        return False
    
    # Use the first label column found
    label_col = label_columns[0]
    print(f"Using label column: '{label_col}'")
    
    print(df[label_col].value_counts().sort_index())
    print(f"Total rows: {len(df)}")
    
    # Convert all labels to 'none' except 'กู' (and optionally keep existing 'none')
    df_converted = df.copy()
    
    # Define which labels to keep
    labels_to_keep = ['กู']
    if keep_none:
        labels_to_keep.append('none')
    
    # Convert all other labels to 'none'
    mask = ~df_converted[label_col].isin(labels_to_keep)
    changed_rows = mask.sum()
    df_converted.loc[mask, label_col] = 'none'
    
    # Show final counts
    print(f"\nAfter converting all labels except {labels_to_keep} to 'none':")
    print("Final label distribution:")
    print(df_converted[label_col].value_counts().sort_index())
    print(f"Total rows: {len(df_converted)} (unchanged)")
    print(f"Changed rows: {changed_rows}")
    
    # Save the converted CSV
    df_converted.to_csv(output_file, index=False)
    print(f"\n✅ Converted data saved to: {output_file}")
    return True

def process_single_file(input_file, output_file=None, keep_none=True):
    """Process a single CSV file to keep only 'กู' labels"""
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        extension = os.path.splitext(input_file)[1]
        output_file = f"{base_name}_guu_only{extension}"
    
    print(f"🏷️ Converting single file to 'กู' only format")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Keep existing 'none': {keep_none}")
    print()
    
    # Process the file
    success = convert_to_guu_only(input_file, output_file, keep_none)
    return success

def process_all_csv_files(csv_dir="csv", pattern=None):
    """Process all CSV files in a directory"""
    
    if not os.path.exists(csv_dir):
        print(f"❌ Directory not found: {csv_dir}")
        return
    
    # Find all CSV files
    csv_files = []
    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            if pattern is None or pattern in file:
                csv_files.append(file)
    
    if not csv_files:
        print(f"❌ No CSV files found in {csv_dir}")
        if pattern:
            print(f"   (searched for pattern: '{pattern}')")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {file}")
    print()
    
    # Ask for confirmation
    proceed = input(f"Process all {len(csv_files)} files? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Cancelled.")
        return
    
    # Process each file
    successful = 0
    for file in csv_files:
        input_file = os.path.join(csv_dir, file)
        output_file = os.path.join(csv_dir, file.replace('.csv', '_guu_only.csv'))
        
        print(f"\n{'='*60}")
        print(f"Processing: {file}")
        print(f"{'='*60}")
        
        success = convert_to_guu_only(input_file, output_file, keep_none=True)
        if success:
            successful += 1
    
    print(f"\n✅ Processing complete! {successful}/{len(csv_files)} files processed successfully.")

if __name__ == "__main__":
    print("🏷️ CSV Label Converter - Keep Only 'กู' Word")
    print("This script converts all labels to 'none' except 'กู' word")
    print()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert CSV labels to keep only กู word')
    parser.add_argument('--input', '-i', type=str, help='Input CSV file path')
    parser.add_argument('--output', '-o', type=str, help='Output CSV file path (optional)')
    parser.add_argument('--csv-dir', type=str, default='csv', help='Directory containing CSV files (default: csv)')
    parser.add_argument('--pattern', type=str, help='Pattern to filter CSV files (e.g., "eval", "windowed")')
    parser.add_argument('--all', action='store_true', help='Process all CSV files in the directory')
    parser.add_argument('--no-keep-none', action='store_true', help='Also convert existing none labels to none')
    
    args = parser.parse_args()
    
    # Determine what to do based on arguments
    if args.input:
        # Process single file
        keep_none = not args.no_keep_none
        success = process_single_file(args.input, args.output, keep_none)
        if success:
            print(f"\n✅ กู-only conversion complete!")
        else:
            print(f"\n❌ กู-only conversion failed!")
            sys.exit(1)
            
    elif args.all:
        # Process all CSV files
        process_all_csv_files(args.csv_dir, args.pattern)
        
    else:
        # No arguments provided - show usage examples and interactive mode
        print("Usage Examples:")
        print()
        print("1. Convert a single file:")
        print("   python convert_to_guu_only.py --input csv/eval.csv")
        print("   python convert_to_guu_only.py --input csv/eval.csv --output csv/eval_guu.csv")
        print()
        print("2. Convert all CSV files in a directory:")
        print("   python convert_to_guu_only.py --all")
        print("   python convert_to_guu_only.py --all --csv-dir csv/eval_0.5s")
        print("   python convert_to_guu_only.py --all --pattern windowed")
        print()
        print("3. Advanced options:")
        print("   python convert_to_guu_only.py --input csv/eval.csv --no-keep-none")
        print()
        
        # Interactive mode
        choice = input("Choose an option:\n1. Single file\n2. All CSV files in directory\n3. Exit\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            input_file = input("Enter input CSV file path: ").strip()
            output_file = input("Enter output file path (or press Enter for auto-name): ").strip()
            if not output_file:
                output_file = None
            
            keep_none = input("Keep existing 'none' labels? (y/n, default=y): ").strip().lower()
            keep_none = keep_none != 'n'
            
            success = process_single_file(input_file, output_file, keep_none)
            if success:
                print(f"\n✅ กู-only conversion complete!")
            
        elif choice == "2":
            csv_dir = input("Enter CSV directory path (default=csv): ").strip()
            if not csv_dir:
                csv_dir = "csv"
            
            pattern = input("Enter filename pattern to filter (or press Enter for all): ").strip()
            if not pattern:
                pattern = None
                
            process_all_csv_files(csv_dir, pattern)
            
        else:
            print("Exiting...")
            sys.exit(0)
    
    print("\nUse the generated *_guu_only.csv files with your evaluation script")
    print("The converted files will contain only 'กู' and 'none' labels.")

#!/usr/bin/env python3
"""
Modify train.csv to keep only 5-class labels and convert others to 'none'
"""

import pandas as pd

def modify_train_csv():
    # Define the labels to keep
    keep_labels = {
        'none': 0,
        'เย็ด': 1,
        'กู': 2,
        'มึง': 3,
        'เหี้ย': 4
    }
    
    # Labels to convert to 'none'
    convert_to_none = ['ควย', 'สวะ', 'หี', 'แตด', 'nonne']
    
    print("🔄 Modifying train.csv to 5-class format...")
    
    # Load the CSV
    df = pd.read_csv('csv/train.csv')
    
    print(f"📊 Original dataset:")
    print(f"   Total samples: {len(df)}")
    print(f"   Label distribution:")
    original_counts = df['label'].value_counts()
    for label, count in original_counts.items():
        print(f"     {label}: {count}")
    
    # Convert unwanted labels to 'none'
    converted_count = 0
    for label in convert_to_none:
        count = (df['label'] == label).sum()
        if count > 0:
            print(f"🔄 Converting {count} samples of '{label}' to 'none'")
            df.loc[df['label'] == label, 'label'] = 'none'
            converted_count += count
    
    # Save the modified CSV
    df.to_csv('csv/train.csv', index=False)
    
    print(f"\n✅ Modification completed!")
    print(f"   Converted {converted_count} samples to 'none'")
    print(f"\n📊 Modified dataset:")
    print(f"   Total samples: {len(df)}")
    print(f"   Label distribution:")
    final_counts = df['label'].value_counts()
    for label, count in final_counts.items():
        print(f"     {label}: {count}")
    
    # Verify only desired labels remain
    unique_labels = set(df['label'].unique())
    expected_labels = set(keep_labels.keys())
    
    if unique_labels == expected_labels:
        print(f"\n✅ Success! Dataset now contains only: {sorted(unique_labels)}")
    else:
        unexpected = unique_labels - expected_labels
        if unexpected:
            print(f"\n⚠️  Warning: Unexpected labels found: {unexpected}")
        missing = expected_labels - unique_labels
        if missing:
            print(f"⚠️  Warning: Missing expected labels: {missing}")

if __name__ == "__main__":
    modify_train_csv()
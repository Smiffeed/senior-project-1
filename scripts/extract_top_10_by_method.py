#!/usr/bin/env python3
"""
Extract Top 10 F1 Scores by Method
Finds top 10 Word F1 and Window F1 configurations for each evaluation method
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import os

def parse_window_stride_from_path(config_path):
    """Parse window and stride from configuration path"""
    # Extract window (e.g., "window_1.2s" -> 1.2)
    window_match = re.search(r'window_(\d+\.?\d*)s', config_path)
    window_numeric = float(window_match.group(1)) if window_match else None
    
    # Extract stride 
    stride_match = re.search(r'stride_(\d+\.?\d*)([s%])?', config_path)
    if stride_match:
        stride_val = float(stride_match.group(1))
        stride_unit = stride_match.group(2)
        
        if stride_unit == '%' or 'eval_percent' in config_path:
            # For eval_percent: stride is percentage
            stride_percentage = stride_val
            stride_numeric = window_numeric * (stride_val / 100) if window_numeric else stride_val / 100
        else:
            # For eval_by_0.05: stride is absolute value
            stride_numeric = stride_val
            stride_percentage = (stride_val / window_numeric * 100) if window_numeric else 100
            
        return window_numeric, stride_numeric, stride_percentage
    
    return window_numeric, None, None

def extract_f1_scores_from_notes(notes_file, method_name):
    """Extract F1 scores from evaluation notes based on method type"""
    if not os.path.exists(notes_file):
        return {}
    
    try:
        with open(notes_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        scores = {}
        
        # Method-specific parsing
        if method_name == 'word_eval':
            # Word-level evaluation - extract word-level F1
            binary_f1_match = re.search(r'F1-Score:\s*([\d.]+)', content)
            if binary_f1_match:
                scores['word_f1'] = float(binary_f1_match.group(1))
            
            # Extract multiclass F1 if available
            multiclass_f1_match = re.search(r'=== MULTICLASS CLASSIFICATION.*?F1-Score:\s*([\d.]+)', content, re.DOTALL)
            if multiclass_f1_match:
                scores['multiclass_f1'] = float(multiclass_f1_match.group(1))
                if 'word_f1' not in scores:
                    scores['word_f1'] = float(multiclass_f1_match.group(1))
            
            scores['window_f1'] = 0.0  # Not available in word_eval
            scores['combined_iou'] = 0.0  # Not available in word_eval
            
        elif method_name == 'window_eval':
            # Window-level evaluation - extract window-level F1
            binary_f1_match = re.search(r'F1-Score:\s*([\d.]+)', content)
            if binary_f1_match:
                scores['window_f1'] = float(binary_f1_match.group(1))
                scores['word_f1'] = float(binary_f1_match.group(1))  # Same score for consistency
            
            scores['combined_iou'] = 0.0  # Not available in window_eval
            
        elif method_name == 'iou_eval':
            # IoU evaluation - extract IoU metrics and F1 from thresholds
            # Extract Combined Mean IoU
            iou_match = re.search(r'Combined Mean IoU:\s*([\d.]+)', content)
            if iou_match:
                scores['combined_iou'] = float(iou_match.group(1))
            
            # Extract F1 from IoU threshold 0.5 (binary classification)
            threshold_05_section = re.search(r'--- IoU Threshold 0\.5 ---.*?F1-Score:\s*([\d.]+)', content, re.DOTALL)
            if threshold_05_section:
                scores['word_f1'] = float(threshold_05_section.group(1))
            else:
                # If no 0.5 threshold, try 0.1 threshold
                threshold_01_section = re.search(r'--- IoU Threshold 0\.1 ---.*?F1-Score:\s*([\d.]+)', content, re.DOTALL)
                if threshold_01_section:
                    scores['word_f1'] = float(threshold_01_section.group(1))
            
            scores['window_f1'] = 0.0  # Not available in iou_eval
            
        elif method_name == 'word_iou_eval':
            # Word+IoU evaluation - combination of word and IoU metrics
            # Extract word-level F1
            binary_f1_match = re.search(r'F1-Score:\s*([\d.]+)', content)
            if binary_f1_match:
                scores['word_f1'] = float(binary_f1_match.group(1))
            
            # Extract IoU if available
            iou_match = re.search(r'Combined Mean IoU:\s*([\d.]+)', content)
            if iou_match:
                scores['combined_iou'] = float(iou_match.group(1))
            else:
                scores['combined_iou'] = 0.0
            
            scores['window_f1'] = 0.0  # Not available in word_iou_eval
        
        # Set defaults for missing scores
        if 'word_f1' not in scores:
            scores['word_f1'] = 0.0
        if 'window_f1' not in scores:
            scores['window_f1'] = 0.0
        if 'combined_iou' not in scores:
            scores['combined_iou'] = 0.0
        
        return scores
        
    except Exception as e:
        print(f"Error reading {notes_file}: {e}")
        return {}

def extract_method_data(base_dir, eval_type, method_name):
    """Extract data for a specific evaluation method"""
    method_dir = Path(base_dir) / eval_type / eval_type / method_name
    
    if not method_dir.exists():
        print(f"❌ Method directory not found: {method_dir}")
        return []
    
    print(f"📊 Processing {method_name} in {eval_type}...")
    
    all_configs = []
    
    # Traverse window directories
    for window_dir in method_dir.iterdir():
        if not window_dir.is_dir() or not window_dir.name.startswith("window_"):
            continue
            
        # Traverse stride directories  
        for stride_dir in window_dir.iterdir():
            if not stride_dir.is_dir() or not stride_dir.name.startswith("stride_"):
                continue
            
            # Look for notes file
            notes_file = stride_dir / "note.txt"
            if not notes_file.exists():
                continue
            
            # Extract configuration info
            config_path = f"{window_dir.name}/{stride_dir.name}"
            window_numeric, stride_numeric, stride_percentage = parse_window_stride_from_path(config_path)
            
            # Extract F1 scores
            scores = extract_f1_scores_from_notes(notes_file, method_name)
            
            if scores:
                config_data = {
                    'eval_type': eval_type,
                    'method': method_name,
                    'window': window_dir.name,
                    'stride': stride_dir.name,
                    'success': True,
                    'duration_seconds': 0.0,  # Not available from notes
                    'worker_id': 0,  # Not available from notes
                    'window_f1': scores.get('window_f1', 0.0),
                    'word_f1': scores.get('word_f1', 0.0), 
                    'combined_iou': scores.get('combined_iou', 0.0),
                    'window_numeric': window_numeric,
                    'stride_numeric': stride_numeric,
                    'stride_percentage': stride_percentage,
                    'configuration': config_path
                }
                all_configs.append(config_data)
    
    print(f"✅ Found {len(all_configs)} configurations for {method_name}")
    return all_configs

def create_method_top10_files():
    """Create top 10 CSV files for each evaluation method"""
    
    base_dir = "fixed_smart_parallel_results"
    methods = ['word_eval', 'word_iou_eval', 'iou_eval', 'window_eval']
    eval_types = ['eval_by_0.05', 'eval_percent']
    
    output_dir = Path("fixed_smart_parallel_results/method_top10_analysis")
    output_dir.mkdir(exist_ok=True)
    
    print("=== EXTRACTING TOP 10 F1 SCORES BY METHOD ===\n")
    
    for method_name in methods:
        print(f"\n🔍 PROCESSING METHOD: {method_name.upper()}")
        print("="*60)
        
        all_method_data = []
        
        # Collect data from both eval types
        for eval_type in eval_types:
            method_configs = extract_method_data(base_dir, eval_type, method_name)
            all_method_data.extend(method_configs)
        
        if not all_method_data:
            print(f"❌ No data found for method: {method_name}")
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_method_data)
        
        # Filter out zero scores
        df_valid = df[(df['window_f1'] > 0) | (df['word_f1'] > 0)]
        
        if df_valid.empty:
            print(f"❌ No valid scores found for method: {method_name}")
            continue
        
        print(f"📊 Total configurations: {len(df_valid)}")
        print(f"📊 Window F1 range: {df_valid['window_f1'].min():.4f} - {df_valid['window_f1'].max():.4f}")
        print(f"📊 Word F1 range: {df_valid['word_f1'].min():.4f} - {df_valid['word_f1'].max():.4f}")
        
        # Create top 10 Window F1
        top10_window = df_valid.nlargest(10, 'window_f1')
        window_file = output_dir / f"top_10_window_f1_{method_name}.csv"
        
        # Select columns matching your sample format
        columns = ['eval_type', 'window', 'stride', 'success', 'duration_seconds', 'worker_id',
                  'window_f1', 'word_f1', 'combined_iou', 'window_numeric', 'stride_numeric', 'stride_percentage']
        
        top10_window[columns].to_csv(window_file, index=False)
        print(f"💾 Saved: {window_file}")
        
        # Create top 10 Word F1  
        top10_word = df_valid.nlargest(10, 'word_f1')
        word_file = output_dir / f"top_10_word_f1_{method_name}.csv"
        top10_word[columns].to_csv(word_file, index=False)
        print(f"💾 Saved: {word_file}")
        
        # Print top 3 for preview
        print(f"\n🏆 TOP 3 WINDOW F1 for {method_name}:")
        for i, row in top10_window.head(3).iterrows():
            print(f"   {row['window_f1']:.4f} - {row['eval_type']} | {row['window']} | {row['stride']}")
        
        print(f"\n🏆 TOP 3 WORD F1 for {method_name}:")
        for i, row in top10_word.head(3).iterrows():
            print(f"   {row['word_f1']:.4f} - {row['eval_type']} | {row['window']} | {row['stride']}")
    
    print(f"\n🎉 All top 10 files saved to: {output_dir}")
    print(f"📁 Files created:")
    for method in methods:
        print(f"   - top_10_window_f1_{method}.csv")  
        print(f"   - top_10_word_f1_{method}.csv")

if __name__ == "__main__":
    create_method_top10_files()

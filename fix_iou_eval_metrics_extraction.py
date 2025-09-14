#!/usr/bin/env python3
"""
Fix IoU Eval Metrics Extraction

This script properly extracts both binary and multiclass F1 scores from iou_eval note.txt files
and updates the threshold optimization analysis to include all available metrics.

The iou_eval method actually provides comprehensive binary and multiclass analysis,
but the CSV files only contain basic metrics. The note.txt files contain the full analysis.
"""

import pandas as pd
import numpy as np
import re
import glob
import os
from pathlib import Path

def extract_metrics_from_note(note_file_path):
    """Extract binary and multiclass F1 scores from iou_eval note.txt files"""
    try:
        with open(note_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metrics_by_threshold = {}
        
        # Extract IoU threshold sections
        threshold_pattern = r'--- IoU Threshold (0\.\d+) ---'
        threshold_matches = list(re.finditer(threshold_pattern, content))
        
        for i, match in enumerate(threshold_matches):
            threshold = float(match.group(1))
            start_pos = match.end()
            
            # Find end position (next threshold or end of file)
            if i + 1 < len(threshold_matches):
                end_pos = threshold_matches[i + 1].start()
            else:
                end_pos = len(content)
            
            section = content[start_pos:end_pos]
            
            # Extract binary F1 score
            binary_f1_pattern = r'Binary Classification.*?F1-Score:\s*(0\.\d+)'
            binary_match = re.search(binary_f1_pattern, section, re.DOTALL)
            binary_f1 = float(binary_match.group(1)) if binary_match else np.nan
            
            # Extract multiclass accuracy (since multiclass F1 varies by implementation)
            multiclass_acc_pattern = r'Multiclass Classification.*?Accuracy:\s*(0\.\d+)'
            multiclass_match = re.search(multiclass_acc_pattern, section, re.DOTALL)
            multiclass_accuracy = float(multiclass_match.group(1)) if multiclass_match else np.nan
            
            # Extract weighted avg F1 from multiclass classification report
            weighted_f1_pattern = r'weighted avg\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)'
            weighted_f1_match = re.search(weighted_f1_pattern, section)
            multiclass_f1 = float(weighted_f1_match.group(1)) if weighted_f1_match else np.nan
            
            metrics_by_threshold[threshold] = {
                'binary_f1': binary_f1,
                'multiclass_f1': multiclass_f1,
                'multiclass_accuracy': multiclass_accuracy
            }
        
        return metrics_by_threshold
        
    except Exception as e:
        print(f"Error processing {note_file_path}: {e}")
        return {}

def load_enhanced_iou_eval_data():
    """Load iou_eval data with enhanced metrics from note.txt files"""
    base_path = "fixed_smart_parallel_results/eval_by_0.05/eval_by_0.05/iou_eval"
    
    all_data = []
    
    # Find all note.txt files in iou_eval directories
    note_files = glob.glob(f"{base_path}/window_*s/stride_*s/note.txt")
    
    print(f"Found {len(note_files)} note.txt files to process...")
    
    for note_file in note_files:
        # Extract window and stride from path
        path_parts = Path(note_file).parts
        window_dir = next(p for p in path_parts if p.startswith('window_'))
        stride_dir = next(p for p in path_parts if p.startswith('stride_'))
        
        window = float(window_dir.replace('window_', '').replace('s', ''))
        stride = float(stride_dir.replace('stride_', '').replace('s', ''))
        
        print(f"Processing {window_dir}/{stride_dir}...")
        
        # Extract metrics from note.txt
        metrics_by_threshold = extract_metrics_from_note(note_file)
        
        # Also load CSV data for comparison
        csv_file = note_file.replace('note.txt', 'iou_threshold_reference.csv')
        if os.path.exists(csv_file):
            try:
                csv_data = pd.read_csv(csv_file)
                
                for _, row in csv_data.iterrows():
                    threshold = row['threshold']
                    
                    # Get enhanced metrics from note.txt
                    note_metrics = metrics_by_threshold.get(threshold, {})
                    
                    record = {
                        'method': 'iou_eval_enhanced',
                        'window': window,
                        'stride': stride,
                        'threshold': threshold,
                        'f1': row.get('f1', np.nan),  # Original CSV F1
                        'precision': row.get('precision', np.nan),
                        'recall': row.get('recall', np.nan),
                        'accuracy': row.get('accuracy', np.nan),
                        'balanced_accuracy': row.get('balanced_accuracy', np.nan),
                        'binary_f1': note_metrics.get('binary_f1', np.nan),  # Enhanced from note
                        'multiclass_f1': note_metrics.get('multiclass_f1', np.nan),  # Enhanced from note
                        'multiclass_accuracy': note_metrics.get('multiclass_accuracy', np.nan)
                    }
                    
                    all_data.append(record)
                    
            except Exception as e:
                print(f"Error loading CSV {csv_file}: {e}")
    
    return pd.DataFrame(all_data)

def create_enhanced_threshold_optimization():
    """Create enhanced threshold optimization analysis with proper iou_eval metrics"""
    
    print("Loading enhanced iou_eval data from note.txt files...")
    iou_eval_data = load_enhanced_iou_eval_data()
    
    print("Loading word_iou_eval data...")
    # Load word_iou_eval data (keeping existing logic)
    word_iou_base_path = "fixed_smart_parallel_results/eval_by_0.05/eval_by_0.05/word_iou_eval"
    word_iou_files = glob.glob(f"{word_iou_base_path}/window_*s/stride_*s/word_iou_threshold_reference.csv")
    
    word_iou_data = []
    for file_path in word_iou_files:
        path_parts = Path(file_path).parts
        window_dir = next(p for p in path_parts if p.startswith('window_'))
        stride_dir = next(p for p in path_parts if p.startswith('stride_'))
        
        window = float(window_dir.replace('window_', '').replace('s', ''))
        stride = float(stride_dir.replace('stride_', '').replace('s', ''))
        
        try:
            df = pd.read_csv(file_path)
            df['method'] = 'word_iou_eval'
            df['window'] = window
            df['stride'] = stride
            word_iou_data.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    word_iou_df = pd.concat(word_iou_data, ignore_index=True) if word_iou_data else pd.DataFrame()
    
    # Combine datasets
    print("Combining datasets...")
    combined_data = pd.concat([iou_eval_data, word_iou_df], ignore_index=True)
    
    # Analyze optimal thresholds for each method
    print("Analyzing optimal thresholds...")
    results = []
    
    for method in ['word_iou_eval', 'iou_eval_enhanced']:
        method_data = combined_data[combined_data['method'] == method]
        
        if len(method_data) == 0:
            continue
            
        print(f"\nAnalyzing {method}...")
        
        # Group by threshold and calculate statistics
        threshold_stats = method_data.groupby('threshold').agg({
            'f1': ['mean', 'std', 'count'],
            'binary_f1': ['mean', 'std', 'count'],
            'multiclass_f1': ['mean', 'std', 'count'],
            'accuracy': ['mean', 'std'],
            'balanced_accuracy': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        threshold_stats.columns = ['_'.join(col).strip() for col in threshold_stats.columns]
        threshold_stats = threshold_stats.reset_index()
        
        # Add method column
        threshold_stats['method'] = method
        
        # Calculate coefficient of variation
        for metric in ['f1', 'binary_f1', 'multiclass_f1']:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            cv_col = f'{metric}_cv'
            
            if mean_col in threshold_stats.columns and std_col in threshold_stats.columns:
                threshold_stats[cv_col] = (threshold_stats[std_col] / threshold_stats[mean_col]).fillna(0)
        
        results.append(threshold_stats)
    
    # Combine results
    final_results = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    # Save enhanced results
    output_file = "enhanced_threshold_optimization_summary.csv"
    final_results.to_csv(output_file, index=False)
    
    print(f"\nEnhanced analysis saved to {output_file}")
    
    # Print summary of findings
    print("\n=== ENHANCED THRESHOLD OPTIMIZATION SUMMARY ===")
    
    for method in final_results['method'].unique():
        method_data = final_results[final_results['method'] == method]
        
        print(f"\n{method.upper()}:")
        
        # Find optimal thresholds for different metrics
        for metric in ['f1', 'binary_f1', 'multiclass_f1']:
            mean_col = f'{metric}_mean'
            if mean_col in method_data.columns:
                best_row = method_data.loc[method_data[mean_col].idxmax()]
                optimal_threshold = best_row['threshold']
                optimal_value = best_row[mean_col]
                
                print(f"  Optimal {metric.upper()}: IoU={optimal_threshold} (Score={optimal_value:.4f})")
    
    return final_results

def main():
    """Main execution function"""
    print("=== FIXING IoU EVAL METRICS EXTRACTION ===")
    print("Extracting comprehensive metrics from note.txt files...")
    
    try:
        enhanced_results = create_enhanced_threshold_optimization()
        
        print("\n✅ Enhanced threshold optimization analysis completed!")
        print("📊 Now both methods show proper binary and multiclass F1 scores")
        print("🎯 IoU_eval metrics extracted from comprehensive note.txt analysis")
        
        return enhanced_results
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        raise

if __name__ == "__main__":
    results = main()

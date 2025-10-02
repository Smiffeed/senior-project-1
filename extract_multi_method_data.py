#!/usr/bin/env python3
"""
Multi-Method Evaluation Data Extractor
Extracts data from ALL evaluation methods: frame-level, word-level, IoU-based, etc.
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

def extract_metrics_from_note(note_path: str) -> Dict:
    """Extract key metrics from a note.txt file"""
    metrics = {
        'window_size': None,
        'stride_value': None,
        'stride_type': None,
        'eval_type': None,
        'evaluation_method': None,  # NEW: track evaluation method
        'combined_mean_iou': None,
        'binary_f1': None,
        'multiclass_f1': None,
        'binary_accuracy': None,
        'binary_precision': None,
        'binary_recall': None,
        'multiclass_accuracy': None,
        'word_level_binary_f1': None,  # NEW: word-level metrics
        'word_level_multiclass_f1': None,
        'iou_by_class': {},
        'individual_mean_iou': None,
        'total_gt_words': None
    }
    
    if not os.path.exists(note_path):
        return metrics
    
    try:
        with open(note_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Detect evaluation method from content
        if 'WORD EVALUATION' in content.upper():
            metrics['evaluation_method'] = 'word_eval'
        elif 'WORD-IOU EVALUATION' in content.upper():
            metrics['evaluation_method'] = 'word_iou_eval'
        elif 'IOU EVALUATION' in content.upper():
            metrics['evaluation_method'] = 'iou_eval'
        elif 'WINDOW EVALUATION' in content.upper() or 'FRAME EVALUATION' in content.upper():
            metrics['evaluation_method'] = 'window_eval'
        else:
            # Default based on path
            metrics['evaluation_method'] = 'window_eval'
        
        # Extract basic configuration
        if 'Window Size:' in content:
            window_match = re.search(r'Window Size: ([\d.]+)s', content)
            if window_match:
                metrics['window_size'] = float(window_match.group(1))
        
        if 'Stride:' in content:
            stride_match = re.search(r'Stride: ([\d.]+) \((\w+)\)', content)
            if stride_match:
                metrics['stride_value'] = float(stride_match.group(1))
                metrics['stride_type'] = stride_match.group(2)
        
        if 'Evaluation Type:' in content:
            eval_match = re.search(r'Evaluation Type: (\w+)', content)
            if eval_match:
                metrics['eval_type'] = eval_match.group(1)
        
        # Extract performance metrics (frame-level)
        if 'Combined Mean IoU:' in content:
            iou_match = re.search(r'Combined Mean IoU: ([\d.]+)', content)
            if iou_match:
                metrics['combined_mean_iou'] = float(iou_match.group(1)) * 100
        
        # Extract Binary F1 (could be window-level or word-level)
        binary_f1_matches = re.findall(r'(?:Binary |Window-level Binary |Word-level Binary )?F1:? ([\d.]+)', content)
        if binary_f1_matches:
            metrics['binary_f1'] = float(binary_f1_matches[0])
        
        # Look for specific word-level metrics
        if 'Word-level Binary F1:' in content:
            word_binary_match = re.search(r'Word-level Binary F1: ([\d.]+)', content)
            if word_binary_match:
                metrics['word_level_binary_f1'] = float(word_binary_match.group(1))
        
        if 'Word-level Multiclass F1:' in content:
            word_multi_match = re.search(r'Word-level Multiclass F1: ([\d.]+)', content)
            if word_multi_match:
                metrics['word_level_multiclass_f1'] = float(word_multi_match.group(1))
        
        # Extract Multiclass F1
        multiclass_f1_matches = re.findall(r'(?:Multiclass |Word-level Multiclass )?F1:? ([\d.]+)', content)
        if multiclass_f1_matches:
            metrics['multiclass_f1'] = float(multiclass_f1_matches[0])
        
        # Extract IoU by class
        iou_section = re.search(r'=== MEAN IoU BY WORD CLASS ===(.*?)===', content, re.DOTALL)
        if iou_section:
            iou_text = iou_section.group(1)
            for line in iou_text.strip().split('\n'):
                if ':' in line:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        class_name = parts[0].strip()
                        iou_value = float(parts[1].strip()) * 100
                        metrics['iou_by_class'][class_name] = iou_value
        
        # Extract binary classification metrics
        binary_section = re.search(r'=== (?:1\. )?BINARY CLASSIFICATION.*?===(.*?)===', content, re.DOTALL)
        if binary_section:
            binary_text = binary_section.group(1)
            
            acc_match = re.search(r'Accuracy: ([\d.]+)', binary_text)
            if acc_match:
                metrics['binary_accuracy'] = float(acc_match.group(1))
            
            prec_match = re.search(r'Precision: ([\d.]+)', binary_text)
            if prec_match:
                metrics['binary_precision'] = float(prec_match.group(1))
            
            recall_match = re.search(r'Recall: ([\d.]+)', binary_text)
            if recall_match:
                metrics['binary_recall'] = float(recall_match.group(1))
        
        # Extract total ground truth words
        if 'Total Ground Truth Words:' in content:
            gt_match = re.search(r'Total Ground Truth Words: (\d+)', content)
            if gt_match:
                metrics['total_gt_words'] = int(gt_match.group(1))
        
    except Exception as e:
        print(f"Error processing {note_path}: {e}")
    
    return metrics

def scan_all_evaluation_methods(base_results_dirs: List[str]) -> pd.DataFrame:
    """Scan all evaluation methods from multiple result directories"""
    all_metrics = []
    
    evaluation_methods = ['window_eval', 'word_eval', 'word_iou_eval', 'iou_eval']
    
    for results_dir in base_results_dirs:
        results_path = Path(results_dir)
        if not results_path.exists():
            print(f"Results directory not found: {results_dir}")
            continue
        
        print(f"\n🔍 Scanning {results_dir}...")
        
        # Method 1: Direct frame-level results (current approach)
        for eval_type in ['eval_percent', 'eval_by_0.05']:
            eval_dir = results_path / eval_type
            if eval_dir.exists():
                print(f"  📊 Processing {eval_type} (frame-level)...")
                all_metrics.extend(scan_frame_level_results(eval_dir, eval_type))
        
        # Method 2: Multi-method results (word_eval, iou_eval, etc.)
        for eval_type in ['eval_percent', 'eval_by_0.05']:
            for method in evaluation_methods:
                method_dir = results_path / eval_type / method
                if method_dir.exists():
                    print(f"  📝 Processing {eval_type}/{method}...")
                    all_metrics.extend(scan_method_results(method_dir, eval_type, method))
    
    df = pd.DataFrame(all_metrics)
    print(f"\n✅ Total configurations extracted: {len(df)}")
    
    if len(df) > 0:
        print(f"📊 Evaluation methods found: {df['evaluation_method'].value_counts().to_dict()}")
    
    return df

def scan_frame_level_results(eval_dir: Path, eval_type: str) -> List[Dict]:
    """Scan frame-level results (original approach)"""
    metrics_list = []
    
    for window_dir in eval_dir.iterdir():
        if not window_dir.is_dir() or not window_dir.name.startswith('window_'):
            continue
        
        for stride_dir in window_dir.iterdir():
            if not stride_dir.is_dir() or not stride_dir.name.startswith('stride_'):
                continue
            
            note_file = stride_dir / 'note.txt'
            if note_file.exists():
                metrics = extract_metrics_from_note(str(note_file))
                
                # Add path information
                metrics['path'] = str(stride_dir)
                metrics['window_dir'] = window_dir.name
                metrics['stride_dir'] = stride_dir.name
                metrics['eval_type'] = eval_type
                metrics['evaluation_method'] = 'window_eval'  # Frame-level
                
                metrics_list.append(metrics)
    
    return metrics_list

def scan_method_results(method_dir: Path, eval_type: str, method: str) -> List[Dict]:
    """Scan specific evaluation method results"""
    metrics_list = []
    
    for window_dir in method_dir.iterdir():
        if not window_dir.is_dir() or not window_dir.name.startswith('window_'):
            continue
        
        for stride_dir in window_dir.iterdir():
            if not stride_dir.is_dir() or not stride_dir.name.startswith('stride_'):
                continue
            
            note_file = stride_dir / 'note.txt'
            if note_file.exists():
                metrics = extract_metrics_from_note(str(note_file))
                
                # Add path information
                metrics['path'] = str(stride_dir)
                metrics['window_dir'] = window_dir.name
                metrics['stride_dir'] = stride_dir.name
                metrics['eval_type'] = eval_type
                metrics['evaluation_method'] = method
                
                metrics_list.append(metrics)
    
    return metrics_list

def create_method_comparison_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create comparison data across different evaluation methods"""
    comparison_data = {}
    
    # Group by evaluation method
    for method in df['evaluation_method'].unique():
        method_df = df[df['evaluation_method'] == method].copy()
        
        if len(method_df) == 0:
            continue
        
        # Calculate statistics by window size
        stats = method_df.groupby('window_size').agg({
            'binary_f1': ['mean', 'std', 'count'],
            'multiclass_f1': ['mean', 'std', 'count'],
            'combined_mean_iou': ['mean', 'std', 'count'],
            'word_level_binary_f1': ['mean', 'std', 'count'],
            'word_level_multiclass_f1': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        stats.columns = ['window_size'] + [f"{col[0]}_{col[1]}" for col in stats.columns[1:]]
        
        comparison_data[method] = stats
    
    return comparison_data

def export_multi_method_data(df: pd.DataFrame, output_dir: str):
    """Export multi-method evaluation data"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Exporting multi-method data to {output_dir}...")
    
    # 1. Export complete multi-method dataset
    df.to_csv(output_path / 'multi_method_evaluation_data.csv', index=False)
    print("✅ Exported multi_method_evaluation_data.csv")
    
    # 2. Export method comparison data
    comparison_data = create_method_comparison_data(df)
    for method, data in comparison_data.items():
        data.to_csv(output_path / f'{method}_comparison_data.csv', index=False)
        print(f"✅ Exported {method}_comparison_data.csv")
    
    # 3. Export method summary
    method_summary = df.groupby('evaluation_method').agg({
        'binary_f1': ['count', 'mean', 'std', 'min', 'max'],
        'multiclass_f1': ['mean', 'std', 'min', 'max'],
        'combined_mean_iou': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    method_summary.to_csv(output_path / 'evaluation_method_summary.csv')
    print("✅ Exported evaluation_method_summary.csv")
    
    # 4. Export best configurations per method
    best_configs = []
    for method in df['evaluation_method'].unique():
        method_df = df[df['evaluation_method'] == method]
        
        if len(method_df) == 0:
            continue
        
        # Best binary F1
        best_binary = method_df.loc[method_df['binary_f1'].idxmax()]
        best_configs.append({
            'evaluation_method': method,
            'metric': 'binary_f1',
            'window_size': best_binary['window_size'],
            'stride_value': best_binary['stride_value'],
            'value': best_binary['binary_f1'],
            'combined_mean_iou': best_binary.get('combined_mean_iou', 'N/A')
        })
        
        # Best multiclass F1
        best_multi = method_df.loc[method_df['multiclass_f1'].idxmax()]
        best_configs.append({
            'evaluation_method': method,
            'metric': 'multiclass_f1',
            'window_size': best_multi['window_size'],
            'stride_value': best_multi['stride_value'],
            'value': best_multi['multiclass_f1'],
            'combined_mean_iou': best_multi.get('combined_mean_iou', 'N/A')
        })
        
        # Best IoU (if available)
        if not method_df['combined_mean_iou'].isna().all():
            best_iou = method_df.loc[method_df['combined_mean_iou'].idxmax()]
            best_configs.append({
                'evaluation_method': method,
                'metric': 'combined_mean_iou',
                'window_size': best_iou['window_size'],
                'stride_value': best_iou['stride_value'],
                'value': best_iou['combined_mean_iou'],
                'binary_f1': best_iou.get('binary_f1', 'N/A')
            })
    
    best_configs_df = pd.DataFrame(best_configs)
    best_configs_df.to_csv(output_path / 'best_configurations_per_method.csv', index=False)
    print("✅ Exported best_configurations_per_method.csv")
    
    return output_path

def main():
    """Main function to extract multi-method evaluation data"""
    print("🎯 MULTI-METHOD EVALUATION DATA EXTRACTOR")
    print("=" * 60)
    
    # Define result directories to scan
    base_results_dirs = [
        "./frame_level_evaluation_results",
        "./new_evaluation_results", 
        "./fixed_smart_parallel_results"
    ]
    
    # Extract data from all methods
    df = scan_all_evaluation_methods(base_results_dirs)
    
    if len(df) == 0:
        print("❌ No evaluation results found!")
        return
    
    # Export all data
    output_dir = "./multi_method_extracted_data"
    export_multi_method_data(df, output_dir)
    
    print(f"\n🎉 MULTI-METHOD DATA EXTRACTION COMPLETED!")
    print(f"📁 Data exported to: {output_dir}")
    print(f"\n📊 EVALUATION METHODS FOUND:")
    for method, count in df['evaluation_method'].value_counts().items():
        print(f"  • {method}: {count} configurations")
    
    print(f"\n💡 NOW YOU CAN CREATE TRUE METHOD COMPARISON GRAPHS!")
    print(f"  • Frame-level vs Word-level vs IoU-based evaluation")
    print(f"  • Performance across different evaluation stages")
    print(f"  • Complete evaluation pipeline analysis")

if __name__ == "__main__":
    main()
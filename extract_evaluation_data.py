#!/usr/bin/env python3
"""
Data Extraction Script for Comprehensive Frame-Level Evaluation Results
Extracts data from all note.txt files to recreate the analysis graphs
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
        'combined_mean_iou': None,
        'binary_f1': None,
        'multiclass_f1': None,
        'binary_accuracy': None,
        'binary_precision': None,
        'binary_recall': None,
        'multiclass_accuracy': None,
        'iou_by_class': {},
        'individual_mean_iou': None,
        'total_gt_words': None
    }
    
    if not os.path.exists(note_path):
        return metrics
    
    try:
        with open(note_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
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
        
        # Extract performance metrics
        if 'Combined Mean IoU:' in content:
            iou_match = re.search(r'Combined Mean IoU: ([\d.]+)', content)
            if iou_match:
                metrics['combined_mean_iou'] = float(iou_match.group(1)) * 100  # Convert to percentage
        
        if 'Individual Mean IoU:' in content:
            ind_iou_match = re.search(r'Individual Mean IoU: ([\d.]+)', content)
            if ind_iou_match:
                metrics['individual_mean_iou'] = float(ind_iou_match.group(1)) * 100
        
        if 'Binary F1:' in content:
            binary_f1_match = re.search(r'Binary F1: ([\d.]+)', content)
            if binary_f1_match:
                metrics['binary_f1'] = float(binary_f1_match.group(1))
        
        if 'Multiclass F1:' in content:
            multi_f1_match = re.search(r'Multiclass F1: ([\d.]+)', content)
            if multi_f1_match:
                metrics['multiclass_f1'] = float(multi_f1_match.group(1))
        
        # Extract IoU by class
        iou_section = re.search(r'=== MEAN IoU BY WORD CLASS ===(.*?)===', content, re.DOTALL)
        if iou_section:
            iou_text = iou_section.group(1)
            for line in iou_text.strip().split('\n'):
                if ':' in line:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        class_name = parts[0].strip()
                        iou_value = float(parts[1].strip()) * 100  # Convert to percentage
                        metrics['iou_by_class'][class_name] = iou_value
        
        # Extract binary classification metrics from IoU >= 0.1 section
        binary_section = re.search(r'=== 1\. BINARY CLASSIFICATION \(Profane vs None\) ===(.*?)===', content, re.DOTALL)
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

def scan_evaluation_results(results_dir: str) -> pd.DataFrame:
    """Scan all evaluation results and extract metrics"""
    all_metrics = []
    results_path = Path(results_dir)
    
    print(f"Scanning {results_dir} for evaluation results...")
    
    # Scan both eval_percent and eval_by_0.05
    for eval_type in ['eval_percent', 'eval_by_0.05']:
        eval_dir = results_path / eval_type
        if not eval_dir.exists():
            continue
        
        print(f"Processing {eval_type}...")
        
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
                    
                    # Infer eval_type if not found
                    if not metrics['eval_type']:
                        metrics['eval_type'] = eval_type
                    
                    all_metrics.append(metrics)
    
    df = pd.DataFrame(all_metrics)
    print(f"Extracted {len(df)} configurations")
    return df

def create_heatmap_data(df: pd.DataFrame, eval_type: str, metric: str) -> pd.DataFrame:
    """Create heatmap data for a specific evaluation type and metric"""
    filtered_df = df[df['eval_type'] == eval_type].copy()
    
    if len(filtered_df) == 0:
        return pd.DataFrame()
    
    # Create pivot table
    if eval_type == 'eval_percent':
        # For percentage-based strides
        pivot_data = filtered_df.pivot_table(
            values=metric,
            index='stride_value',
            columns='window_size',
            aggfunc='mean'
        )
    else:
        # For absolute time-based strides
        pivot_data = filtered_df.pivot_table(
            values=metric,
            index='stride_value',
            columns='window_size',
            aggfunc='mean'
        )
    
    return pivot_data

def create_iou_class_heatmap_data(df: pd.DataFrame, eval_type: str, class_name: str) -> pd.DataFrame:
    """Create heatmap data for IoU by specific class"""
    filtered_df = df[df['eval_type'] == eval_type].copy()
    
    if len(filtered_df) == 0:
        return pd.DataFrame()
    
    # Extract IoU for specific class
    filtered_df[f'iou_{class_name}'] = filtered_df['iou_by_class'].apply(
        lambda x: x.get(class_name, np.nan) if isinstance(x, dict) else np.nan
    )
    
    pivot_data = filtered_df.pivot_table(
        values=f'iou_{class_name}',
        index='stride_value',
        columns='window_size',
        aggfunc='mean'
    )
    
    return pivot_data

def create_line_graph_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create line graph data for method comparisons"""
    line_data = {}
    
    # Group by window size and calculate means
    for eval_type in ['eval_percent', 'eval_by_0.05']:
        filtered_df = df[df['eval_type'] == eval_type].copy()
        
        if len(filtered_df) == 0:
            continue
        
        # Group by window size and calculate statistics
        grouped = filtered_df.groupby('window_size').agg({
            'binary_f1': ['mean', 'std', 'count'],
            'multiclass_f1': ['mean', 'std', 'count'],
            'combined_mean_iou': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['window_size'] + [f"{col[0]}_{col[1]}" for col in grouped.columns[1:]]
        
        line_data[eval_type] = grouped
    
    return line_data

def export_data_for_graphs(df: pd.DataFrame, output_dir: str):
    """Export all data needed to recreate the graphs"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting data to {output_dir}...")
    
    # 1. Export complete dataset
    df.to_csv(output_path / 'complete_evaluation_data.csv', index=False)
    print("✅ Exported complete_evaluation_data.csv")
    
    # 2. Export binary F1 heatmap data
    for eval_type in ['eval_percent', 'eval_by_0.05']:
        binary_f1_heatmap = create_heatmap_data(df, eval_type, 'binary_f1')
        if not binary_f1_heatmap.empty:
            binary_f1_heatmap.to_csv(output_path / f'{eval_type}_binary_f1_heatmap.csv')
            print(f"✅ Exported {eval_type}_binary_f1_heatmap.csv")
    
    # 3. Export multiclass F1 heatmap data
    for eval_type in ['eval_percent', 'eval_by_0.05']:
        multi_f1_heatmap = create_heatmap_data(df, eval_type, 'multiclass_f1')
        if not multi_f1_heatmap.empty:
            multi_f1_heatmap.to_csv(output_path / f'{eval_type}_multiclass_f1_heatmap.csv')
            print(f"✅ Exported {eval_type}_multiclass_f1_heatmap.csv")
    
    # 4. Export IoU heatmap data
    for eval_type in ['eval_percent', 'eval_by_0.05']:
        iou_heatmap = create_heatmap_data(df, eval_type, 'combined_mean_iou')
        if not iou_heatmap.empty:
            iou_heatmap.to_csv(output_path / f'{eval_type}_combined_mean_iou_heatmap.csv')
            print(f"✅ Exported {eval_type}_combined_mean_iou_heatmap.csv")
    
    # 5. Export IoU by class heatmaps
    thai_classes = ['เย็ด', 'กู', 'มึง', 'เหี้ย']
    for eval_type in ['eval_percent', 'eval_by_0.05']:
        for class_name in thai_classes:
            class_heatmap = create_iou_class_heatmap_data(df, eval_type, class_name)
            if not class_heatmap.empty:
                safe_class_name = class_name.replace('/', '_')
                class_heatmap.to_csv(output_path / f'{eval_type}_iou_{safe_class_name}_heatmap.csv')
                print(f"✅ Exported {eval_type}_iou_{safe_class_name}_heatmap.csv")
    
    # 6. Export line graph data
    line_data = create_line_graph_data(df)
    for eval_type, data in line_data.items():
        data.to_csv(output_path / f'{eval_type}_line_graph_data.csv', index=False)
        print(f"✅ Exported {eval_type}_line_graph_data.csv")
    
    # 7. Export optimal configurations
    optimal_configs = []
    
    for eval_type in ['eval_percent', 'eval_by_0.05']:
        filtered_df = df[df['eval_type'] == eval_type].copy()
        
        if len(filtered_df) == 0:
            continue
        
        # Find optimal for binary F1
        best_binary = filtered_df.loc[filtered_df['binary_f1'].idxmax()]
        optimal_configs.append({
            'eval_type': eval_type,
            'metric': 'binary_f1',
            'window_size': best_binary['window_size'],
            'stride_value': best_binary['stride_value'],
            'stride_type': best_binary['stride_type'],
            'value': best_binary['binary_f1'],
            'combined_mean_iou': best_binary['combined_mean_iou']
        })
        
        # Find optimal for multiclass F1
        best_multi = filtered_df.loc[filtered_df['multiclass_f1'].idxmax()]
        optimal_configs.append({
            'eval_type': eval_type,
            'metric': 'multiclass_f1',
            'window_size': best_multi['window_size'],
            'stride_value': best_multi['stride_value'],
            'stride_type': best_multi['stride_type'],
            'value': best_multi['multiclass_f1'],
            'combined_mean_iou': best_multi['combined_mean_iou']
        })
        
        # Find optimal for IoU
        best_iou = filtered_df.loc[filtered_df['combined_mean_iou'].idxmax()]
        optimal_configs.append({
            'eval_type': eval_type,
            'metric': 'combined_mean_iou',
            'window_size': best_iou['window_size'],
            'stride_value': best_iou['stride_value'],
            'stride_type': best_iou['stride_type'],
            'value': best_iou['combined_mean_iou'],
            'binary_f1': best_iou['binary_f1']
        })
    
    optimal_df = pd.DataFrame(optimal_configs)
    optimal_df.to_csv(output_path / 'optimal_configurations.csv', index=False)
    print("✅ Exported optimal_configurations.csv")
    
    # 8. Export summary statistics
    summary_stats = {}
    
    for eval_type in ['eval_percent', 'eval_by_0.05']:
        filtered_df = df[df['eval_type'] == eval_type].copy()
        
        if len(filtered_df) == 0:
            continue
        
        stats = {
            'total_configurations': len(filtered_df),
            'window_sizes': sorted(filtered_df['window_size'].unique().tolist()),
            'stride_range': {
                'min': filtered_df['stride_value'].min(),
                'max': filtered_df['stride_value'].max(),
                'unique_count': len(filtered_df['stride_value'].unique())
            },
            'performance_ranges': {
                'binary_f1': {
                    'min': filtered_df['binary_f1'].min(),
                    'max': filtered_df['binary_f1'].max(),
                    'mean': filtered_df['binary_f1'].mean()
                },
                'multiclass_f1': {
                    'min': filtered_df['multiclass_f1'].min(),
                    'max': filtered_df['multiclass_f1'].max(),
                    'mean': filtered_df['multiclass_f1'].mean()
                },
                'combined_mean_iou': {
                    'min': filtered_df['combined_mean_iou'].min(),
                    'max': filtered_df['combined_mean_iou'].max(),
                    'mean': filtered_df['combined_mean_iou'].mean()
                }
            }
        }
        
        summary_stats[eval_type] = stats
    
    with open(output_path / 'summary_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    print("✅ Exported summary_statistics.json")
    
    return output_path

def main():
    """Main function to extract all evaluation data"""
    results_dir = "./frame_level_evaluation_results"
    output_dir = "./extracted_evaluation_data"
    
    print("🎯 COMPREHENSIVE EVALUATION DATA EXTRACTOR")
    print("=" * 50)
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Extract all metrics
    df = scan_evaluation_results(results_dir)
    
    if len(df) == 0:
        print("❌ No evaluation results found!")
        return
    
    print(f"\n📊 EXTRACTION SUMMARY:")
    print(f"Total configurations: {len(df)}")
    
    # Show breakdown by evaluation type
    for eval_type in df['eval_type'].unique():
        count = len(df[df['eval_type'] == eval_type])
        print(f"  • {eval_type}: {count} configurations")
    
    # Show window size range
    window_sizes = sorted(df['window_size'].unique())
    print(f"Window sizes: {window_sizes[0]}s - {window_sizes[-1]}s ({len(window_sizes)} sizes)")
    
    # Show performance ranges
    print(f"\n📈 PERFORMANCE RANGES:")
    print(f"Binary F1: {df['binary_f1'].min():.3f} - {df['binary_f1'].max():.3f}")
    print(f"Multiclass F1: {df['multiclass_f1'].min():.3f} - {df['multiclass_f1'].max():.3f}")
    print(f"Combined Mean IoU: {df['combined_mean_iou'].min():.2f}% - {df['combined_mean_iou'].max():.2f}%")
    
    # Export all data
    output_path = export_data_for_graphs(df, output_dir)
    
    print(f"\n🎉 DATA EXTRACTION COMPLETED!")
    print(f"📁 All data exported to: {output_path}")
    print(f"\n📋 EXPORTED FILES:")
    print(f"  • complete_evaluation_data.csv - Full dataset")
    print(f"  • *_heatmap.csv - Heatmap data for graphs")
    print(f"  • *_line_graph_data.csv - Line graph data") 
    print(f"  • optimal_configurations.csv - Best performing configs")
    print(f"  • summary_statistics.json - Overall statistics")
    
    print(f"\n💡 USE CASES:")
    print(f"  • Import CSV files into plotting libraries (matplotlib, seaborn)")
    print(f"  • Create custom visualizations with extracted data")
    print(f"  • Analyze optimal configurations for different metrics")
    print(f"  • Compare performance across evaluation methods")

if __name__ == "__main__":
    main()
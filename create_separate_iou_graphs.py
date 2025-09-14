#!/usr/bin/env python3
"""
IoU Percentage Analysis - Separated by Evaluation Type
Create separate graphs for eval_by_0.05 and eval_percent
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import os

def extract_iou_percent_data():
    """Extract IoU percentage data from evaluation results"""
    
    base_dir = Path("fixed_smart_parallel_results")
    all_results = []
    
    # Process both eval_by_0.05 and eval_percent directories
    for eval_type in ["eval_by_0.05", "eval_percent"]:
        eval_dir = base_dir / eval_type / eval_type
        
        if not eval_dir.exists():
            print(f"Directory not found: {eval_dir}")
            continue
        
        print(f"Processing {eval_type}...")
        
        # Method 1: IoU Evaluation
        iou_eval_dir = eval_dir / "iou_eval"
        if iou_eval_dir.exists():
            for window_dir in iou_eval_dir.iterdir():
                if window_dir.is_dir() and window_dir.name.startswith("window_"):
                    for stride_dir in window_dir.iterdir():
                        if stride_dir.is_dir() and stride_dir.name.startswith("stride_"):
                            note_file = stride_dir / "note.txt"
                            if note_file.exists():
                                iou_data = extract_iou_percentages(note_file)
                                if iou_data:
                                    window_match = re.search(r'window_(\d+\.?\d*)s', window_dir.name)
                                    
                                    # Handle both time-based (0.3s) and percentage-based (90.0%) stride formats
                                    stride_match_time = re.search(r'stride_(\d+\.?\d*)s', stride_dir.name)
                                    stride_match_percent = re.search(r'stride_(\d+\.?\d*)%', stride_dir.name)
                                    
                                    if window_match and (stride_match_time or stride_match_percent):
                                        if stride_match_time:
                                            stride_numeric = float(stride_match_time.group(1))
                                        else:
                                            # Convert percentage to decimal for consistency (50% -> 0.5)
                                            stride_numeric = float(stride_match_percent.group(1))
                                        
                                        config_data = {
                                            'eval_type': eval_type,
                                            'method': 'iou_eval',
                                            'window': window_dir.name,
                                            'stride': stride_dir.name,
                                            'window_numeric': float(window_match.group(1)),
                                            'stride_numeric': stride_numeric,
                                        }
                                        config_data.update(iou_data)
                                        all_results.append(config_data)
        
        # Method 2: Word-IoU Evaluation
        word_iou_eval_dir = eval_dir / "word_iou_eval"
        if word_iou_eval_dir.exists():
            for window_dir in word_iou_eval_dir.iterdir():
                if window_dir.is_dir() and window_dir.name.startswith("window_"):
                    for stride_dir in window_dir.iterdir():
                        if stride_dir.is_dir() and stride_dir.name.startswith("stride_"):
                            note_file = stride_dir / "note.txt"
                            if note_file.exists():
                                iou_data = extract_word_iou_percentages(note_file)
                                if iou_data:
                                    window_match = re.search(r'window_(\d+\.?\d*)s', window_dir.name)
                                    
                                    # Handle both time-based (0.3s) and percentage-based (90.0%) stride formats
                                    stride_match_time = re.search(r'stride_(\d+\.?\d*)s', stride_dir.name)
                                    stride_match_percent = re.search(r'stride_(\d+\.?\d*)%', stride_dir.name)
                                    
                                    if window_match and (stride_match_time or stride_match_percent):
                                        if stride_match_time:
                                            stride_numeric = float(stride_match_time.group(1))
                                        else:
                                            # Convert percentage to decimal for consistency (50% -> 0.5)
                                            stride_numeric = float(stride_match_percent.group(1))
                                        
                                        config_data = {
                                            'eval_type': eval_type,
                                            'method': 'word_iou_eval',
                                            'window': window_dir.name,
                                            'stride': stride_dir.name,
                                            'window_numeric': float(window_match.group(1)),
                                            'stride_numeric': stride_numeric,
                                        }
                                        config_data.update(iou_data)
                                        all_results.append(config_data)
    
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n✅ Total configurations extracted: {len(df)}")
        print(f"Evaluation types: {df['eval_type'].unique()}")
        print(f"Methods: {df['method'].unique()}")
        return df
    else:
        return pd.DataFrame()

def extract_word_iou_percentages(note_file):
    """Extract IoU metrics from Word-IoU evaluation note files"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        iou_data = {}
        
        # Extract F1 scores at different IoU thresholds
        thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
        
        for threshold in thresholds:
            # Extract F1 score as IoU performance metric
            f1_pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?F1-Score:\s*([\d.]+)'
            f1_match = re.search(f1_pattern, content, re.DOTALL)
            if f1_match:
                iou_data[f'iou_f1_{threshold}'] = float(f1_match.group(1))
            
            # Extract precision and recall for context
            precision_pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?Precision:\s*([\d.]+)'
            recall_pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?Recall:\s*([\d.]+)'
            
            precision_match = re.search(precision_pattern, content, re.DOTALL)
            recall_match = re.search(recall_pattern, content, re.DOTALL)
            
            if precision_match:
                iou_data[f'precision_{threshold}'] = float(precision_match.group(1))
            if recall_match:
                iou_data[f'recall_{threshold}'] = float(recall_match.group(1))
        
        return iou_data
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return {}

def extract_iou_percentages(note_file):
    """Extract IoU metrics from IoU evaluation note files"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        iou_data = {}
        
        # Extract Combined Mean IoU (main IoU metric)
        combined_iou_pattern = r'Combined Mean IoU:\s*([\d.]+)'
        combined_match = re.search(combined_iou_pattern, content)
        if combined_match:
            # Convert to percentage
            iou_data['combined_mean_iou_percent'] = float(combined_match.group(1)) * 100
        
        # Extract Individual Mean IoU
        individual_iou_pattern = r'Individual Mean IoU:\s*([\d.]+)'
        individual_match = re.search(individual_iou_pattern, content)
        if individual_match:
            iou_data['individual_mean_iou_percent'] = float(individual_match.group(1)) * 100
        
        # Extract F1 scores at different IoU thresholds
        thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
        
        for threshold in thresholds:
            # Extract F1 score as IoU performance metric
            f1_pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?F1-Score:\s*([\d.]+)'
            f1_match = re.search(f1_pattern, content, re.DOTALL)
            if f1_match:
                iou_data[f'iou_f1_{threshold}'] = float(f1_match.group(1))
            
            # Extract precision and recall for context
            precision_pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?Precision:\s*([\d.]+)'
            recall_pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?Recall:\s*([\d.]+)'
            
            precision_match = re.search(precision_pattern, content, re.DOTALL)
            recall_match = re.search(recall_pattern, content, re.DOTALL)
            
            if precision_match:
                iou_data[f'precision_{threshold}'] = float(precision_match.group(1))
            if recall_match:
                iou_data[f'recall_{threshold}'] = float(recall_match.group(1))
        
        return iou_data
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return {}

def create_single_heatmap(data, col_name, output_dir, title, filename, cbar_label):
    """Create a single heatmap"""
    
    # Create pivot table for heatmap
    pivot_data = data.pivot_table(
        values=col_name,
        index='stride_numeric',
        columns='window_numeric',
        aggfunc='mean'
    )
    
    if pivot_data.empty:
        return
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    
    # Use a color map that emphasizes higher values
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.2f' if 'percent' in col_name else '.3f',
        cmap='YlOrRd',
        cbar_kws={'label': cbar_label},
        square=False,
        linewidths=0.5,
        annot_kws={'size': 10}
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Stride Size (percent)', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved: {filename}")

def create_separated_heatmaps(df, output_dir):
    """Create separate heatmaps for each evaluation type and method"""
    
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type].copy()
        
        # Create subdirectory for this evaluation type
        eval_output_dir = output_dir / eval_type
        eval_output_dir.mkdir(exist_ok=True)
        
        print(f"\n🎯 Creating heatmaps for {eval_type}")
        
        for method in eval_data['method'].unique():
            method_data = eval_data[eval_data['method'] == method].copy()
            
            if method_data.empty:
                continue
            
            method_name = 'IoU Evaluation' if method == 'iou_eval' else 'Word-IoU Evaluation'
            print(f"  📊 Processing {method_name}")
            
            if method == 'iou_eval':
                # IoU percentage metrics
                iou_metrics = [
                    ('combined_mean_iou_percent', 'Combined Mean IoU Percentage'),
                    ('individual_mean_iou_percent', 'Individual Mean IoU Percentage')
                ]
                
                for col_name, title_suffix in iou_metrics:
                    if col_name not in method_data.columns:
                        continue
                    
                    create_single_heatmap(
                        method_data, col_name, eval_output_dir,
                        f'{eval_type} - {method_name} - {title_suffix}',
                        f"{eval_type}_{method.lower()}_{col_name}_heatmap.png",
                        f'{title_suffix} (%)'
                    )
                
                # Also create F1 score heatmaps for IoU evaluation
                thresholds = ['0.3', '0.5', '0.9']
                for threshold in thresholds:
                    col_name = f'iou_f1_{threshold}'
                    if col_name not in method_data.columns:
                        continue
                    
                    create_single_heatmap(
                        method_data, col_name, eval_output_dir,
                        f'{eval_type} - {method_name} - F1 Score at IoU ≥ {threshold}',
                        f"{eval_type}_{method.lower()}_f1_iou_{threshold}_heatmap.png",
                        f'F1 Score (IoU ≥ {threshold})'
                    )
            
            elif method == 'word_iou_eval':
                # F1 scores at IoU thresholds
                thresholds = ['0.3', '0.5', '0.9']
                
                for threshold in thresholds:
                    col_name = f'iou_f1_{threshold}'
                    if col_name not in method_data.columns:
                        continue
                    
                    create_single_heatmap(
                        method_data, col_name, eval_output_dir,
                        f'{eval_type} - {method_name} - F1 Score at IoU ≥ {threshold}',
                        f"{eval_type}_{method.lower()}_f1_iou_{threshold}_heatmap.png",
                        f'F1 Score (IoU ≥ {threshold})'
                    )

def create_comparison_graphs(df, output_dir):
    """Create comparison graphs between eval_by_0.05 and eval_percent"""
    
    methods = [('word_iou_eval', 'Word-IoU Evaluation'), ('iou_eval', 'IoU Evaluation')]
    
    for method, method_name in methods:
        method_data = df[df['method'] == method].copy()
        if method_data.empty:
            continue
        
        print(f"\n📈 Creating comparison graphs for {method_name}")
        
        if method == 'word_iou_eval':
            # Compare F1 scores at different IoU thresholds
            thresholds = ['0.3', '0.5', '0.9']
            
            fig, axes = plt.subplots(1, len(thresholds), figsize=(18, 6))
            if len(thresholds) == 1:
                axes = [axes]
            
            for i, threshold in enumerate(thresholds):
                col_name = f'iou_f1_{threshold}'
                if col_name not in method_data.columns:
                    continue
                
                ax = axes[i]
                
                for eval_type in method_data['eval_type'].unique():
                    eval_subset = method_data[method_data['eval_type'] == eval_type]
                    
                    # Group by window size and calculate mean/std
                    window_stats = eval_subset.groupby('window_numeric')[col_name].agg(['mean', 'std']).reset_index()
                    
                    # Plot line with error bars
                    ax.errorbar(window_stats['window_numeric'], window_stats['mean'], 
                               yerr=window_stats['std'], label=eval_type, 
                               marker='o', linewidth=2, capsize=5, alpha=0.8)
                
                ax.set_xlabel('Window Size (seconds)', fontweight='bold')
                ax.set_ylabel('F1 Score', fontweight='bold')
                ax.set_title(f'F1 Score at IoU ≥ {threshold}', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            plt.suptitle(f'{method_name} - Comparison Between Evaluation Types', 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(output_dir / f"{method.lower()}_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        elif method == 'iou_eval':
            # Compare IoU percentages
            iou_metrics = [
                ('combined_mean_iou_percent', 'Combined Mean IoU'),
                ('individual_mean_iou_percent', 'Individual Mean IoU')
            ]
            
            fig, axes = plt.subplots(1, len(iou_metrics), figsize=(14, 6))
            if len(iou_metrics) == 1:
                axes = [axes]
            
            for i, (col_name, metric_name) in enumerate(iou_metrics):
                if col_name not in method_data.columns:
                    continue
                
                ax = axes[i]
                
                for eval_type in method_data['eval_type'].unique():
                    eval_subset = method_data[method_data['eval_type'] == eval_type]
                    
                    # Group by window size and calculate mean/std
                    window_stats = eval_subset.groupby('window_numeric')[col_name].agg(['mean', 'std']).reset_index()
                    
                    # Plot line with error bars
                    ax.errorbar(window_stats['window_numeric'], window_stats['mean'], 
                               yerr=window_stats['std'], label=eval_type, 
                               marker='o', linewidth=2, capsize=5, alpha=0.8)
                
                ax.set_xlabel('Window Size (seconds)', fontweight='bold')
                ax.set_ylabel('IoU Percentage (%)', fontweight='bold')
                ax.set_title(f'{metric_name} (%)', fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            plt.suptitle(f'{method_name} - Comparison Between Evaluation Types', 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(output_dir / f"{method.lower()}_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

def create_summary_csv(df, output_dir):
    """Create summary CSV files for each evaluation type"""
    
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type].copy()
        
        summary_stats = []
        
        for method in eval_data['method'].unique():
            method_data = eval_data[eval_data['method'] == method]
            
            if method == 'word_iou_eval':
                # Analyze F1 scores at different IoU thresholds
                for threshold in ['0.3', '0.5', '0.9']:
                    col_name = f'iou_f1_{threshold}'
                    if col_name not in method_data.columns:
                        continue
                    
                    stats = method_data[col_name].describe()
                    best_config = method_data.loc[method_data[col_name].idxmax()]
                    
                    summary_stats.append({
                        'eval_type': eval_type,
                        'method': method,
                        'metric_type': f'F1_IoU_{threshold}',
                        'mean_value': stats['mean'],
                        'std_value': stats['std'],
                        'min_value': stats['min'],
                        'max_value': stats['max'],
                        'median_value': stats['50%'],
                        'best_window': best_config['window'],
                        'best_stride': best_config['stride'],
                        'best_value': best_config[col_name],
                        'total_configs': len(method_data)
                    })
            
            elif method == 'iou_eval':
                # Analyze Mean IoU percentages
                iou_metrics = [
                    ('combined_mean_iou_percent', 'Combined_Mean_IoU_Percent'),
                    ('individual_mean_iou_percent', 'Individual_Mean_IoU_Percent')
                ]
                
                for col_name, metric_name in iou_metrics:
                    if col_name not in method_data.columns:
                        continue
                    
                    stats = method_data[col_name].describe()
                    best_config = method_data.loc[method_data[col_name].idxmax()]
                    
                    summary_stats.append({
                        'eval_type': eval_type,
                        'method': method,
                        'metric_type': metric_name,
                        'mean_value': stats['mean'],
                        'std_value': stats['std'],
                        'min_value': stats['min'],
                        'max_value': stats['max'],
                        'median_value': stats['50%'],
                        'best_window': best_config['window'],
                        'best_stride': best_config['stride'],
                        'best_value': best_config[col_name],
                        'total_configs': len(method_data)
                    })
        
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            summary_df = summary_df.round(4)
            summary_df.to_csv(output_dir / f"{eval_type}_summary_statistics.csv", index=False)
            
            print(f"💾 Saved: {eval_type}_summary_statistics.csv")

def main():
    """Main function to run separated IoU percentage analysis"""
    
    print("=== CREATING SEPARATED IoU PERCENTAGE ANALYSIS ===\n")
    
    # Extract data
    df = extract_iou_percent_data()
    
    if df.empty:
        print("❌ No IoU percentage data extracted!")
        return
    
    print(f"✅ Extracted data from {len(df)} configurations")
    print(f"   Evaluation types: {list(df['eval_type'].unique())}")
    print(f"   Methods: {list(df['method'].unique())}")
    
    # Create output directory
    output_dir = Path("separated_iou_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Create analysis components
    create_separated_heatmaps(df, output_dir)
    create_comparison_graphs(df, output_dir)
    create_summary_csv(df, output_dir)
    
    print(f"\n🎉 Separated IoU percentage analysis completed successfully!")
    print(f"📁 Output directory: {output_dir}")
    
    # Print quick summary for each evaluation type
    print("\n📊 QUICK SUMMARY:")
    for eval_type in df['eval_type'].unique():
        print(f"\n{eval_type.upper()}:")
        eval_data = df[df['eval_type'] == eval_type]
        
        for method in eval_data['method'].unique():
            method_data = eval_data[eval_data['method'] == method]
            method_name = 'IoU Eval' if method == 'iou_eval' else 'Word-IoU Eval'
            
            if method == 'word_iou_eval':
                for threshold in ['0.3', '0.5', '0.9']:
                    col_name = f'iou_f1_{threshold}'
                    if col_name in method_data.columns:
                        best_config = method_data.loc[method_data[col_name].idxmax()]
                        print(f"   {method_name} F1 (IoU≥{threshold}): {best_config[col_name]:.4f} "
                              f"({best_config['window']}, {best_config['stride']})")
            
            elif method == 'iou_eval':
                # Show Mean IoU percentages
                for col_name, label in [('combined_mean_iou_percent', 'Combined Mean IoU'), 
                                       ('individual_mean_iou_percent', 'Individual Mean IoU')]:
                    if col_name in method_data.columns:
                        best_config = method_data.loc[method_data[col_name].idxmax()]
                        print(f"   {method_name} {label}: {best_config[col_name]:.2f}% "
                              f"({best_config['window']}, {best_config['stride']})")

if __name__ == "__main__":
    main()

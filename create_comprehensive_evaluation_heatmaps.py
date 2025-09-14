#!/usr/bin/env python3
"""
Comprehensive Evaluation Heatmap Analysis
Create separate heatmaps for ALL evaluation methods (window_eval, word_eval, iou_eval, word_iou_eval)
across both eval_by_0.05 and eval_percent directories
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import os

def extract_all_evaluation_data():
    """Extract data from all evaluation methods"""
    
    base_dir = Path("fixed_smart_parallel_results")
    all_results = []
    
    # Process both eval_by_0.05 and eval_percent directories
    for eval_type in ["eval_by_0.05", "eval_percent"]:
        eval_dir = base_dir / eval_type / eval_type
        
        if not eval_dir.exists():
            print(f"Directory not found: {eval_dir}")
            continue
        
        print(f"Processing {eval_type}...")
        
        # All evaluation methods
        methods = {
            "window_eval": extract_window_eval_metrics,
            "word_eval": extract_word_eval_metrics,
            "iou_eval": extract_iou_eval_metrics,
            "word_iou_eval": extract_word_iou_eval_metrics
        }
        
        for method_name, extract_func in methods.items():
            method_dir = eval_dir / method_name
            if method_dir.exists():
                print(f"  Processing {method_name}...")
                
                for window_dir in method_dir.iterdir():
                    if window_dir.is_dir() and window_dir.name.startswith("window_"):
                        for stride_dir in window_dir.iterdir():
                            if stride_dir.is_dir() and stride_dir.name.startswith("stride_"):
                                note_file = stride_dir / "note.txt"
                                if note_file.exists():
                                    metrics_data = extract_func(note_file)
                                    if metrics_data:
                                        window_match = re.search(r'window_(\d+\.?\d*)s', window_dir.name)
                                        
                                        # Handle both time-based and percentage-based stride formats
                                        stride_match_time = re.search(r'stride_(\d+\.?\d*)s', stride_dir.name)
                                        stride_match_percent = re.search(r'stride_(\d+\.?\d*)%', stride_dir.name)
                                        
                                        if window_match and (stride_match_time or stride_match_percent):
                                            if stride_match_time:
                                                stride_numeric = float(stride_match_time.group(1))
                                            else:
                                                stride_numeric = float(stride_match_percent.group(1))
                                            
                                            config_data = {
                                                'eval_type': eval_type,
                                                'method': method_name,
                                                'window': window_dir.name,
                                                'stride': stride_dir.name,
                                                'window_numeric': float(window_match.group(1)),
                                                'stride_numeric': stride_numeric,
                                            }
                                            config_data.update(metrics_data)
                                            all_results.append(config_data)
    
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n✅ Total configurations extracted: {len(df)}")
        print(f"Evaluation types: {list(df['eval_type'].unique())}")
        print(f"Methods: {list(df['method'].unique())}")
        return df
    else:
        return pd.DataFrame()

def extract_window_eval_metrics(note_file):
    """Extract metrics from Window evaluation note files"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metrics_data = {}
        
        # Extract Binary Classification metrics
        binary_f1_pattern = r'=== BINARY CLASSIFICATION.*?F1-Score:\s*([\d.]+)'
        binary_f1_match = re.search(binary_f1_pattern, content, re.DOTALL)
        if binary_f1_match:
            metrics_data['window_binary_f1'] = float(binary_f1_match.group(1))
        
        binary_accuracy_pattern = r'=== BINARY CLASSIFICATION.*?Accuracy:\s*([\d.]+)'
        binary_accuracy_match = re.search(binary_accuracy_pattern, content, re.DOTALL)
        if binary_accuracy_match:
            metrics_data['window_binary_accuracy'] = float(binary_accuracy_match.group(1))
        
        binary_precision_pattern = r'=== BINARY CLASSIFICATION.*?Precision:\s*([\d.]+)'
        binary_recall_pattern = r'=== BINARY CLASSIFICATION.*?Recall:\s*([\d.]+)'
        
        precision_match = re.search(binary_precision_pattern, content, re.DOTALL)
        recall_match = re.search(binary_recall_pattern, content, re.DOTALL)
        
        if precision_match:
            metrics_data['window_binary_precision'] = float(precision_match.group(1))
        if recall_match:
            metrics_data['window_binary_recall'] = float(recall_match.group(1))
        
        # Extract Multiclass Classification metrics
        multiclass_f1_pattern = r'=== MULTICLASS CLASSIFICATION.*?F1-Score:\s*([\d.]+)'
        multiclass_f1_match = re.search(multiclass_f1_pattern, content, re.DOTALL)
        if multiclass_f1_match:
            metrics_data['window_multiclass_f1'] = float(multiclass_f1_match.group(1))
        
        multiclass_accuracy_pattern = r'=== MULTICLASS CLASSIFICATION.*?Accuracy:\s*([\d.]+)'
        multiclass_accuracy_match = re.search(multiclass_accuracy_pattern, content, re.DOTALL)
        if multiclass_accuracy_match:
            metrics_data['window_multiclass_accuracy'] = float(multiclass_accuracy_match.group(1))
        
        return metrics_data
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return {}

def extract_word_eval_metrics(note_file):
    """Extract metrics from Word evaluation note files"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metrics_data = {}
        
        # Extract Binary Classification metrics
        binary_f1_pattern = r'=== BINARY CLASSIFICATION.*?F1-Score:\s*([\d.]+)'
        binary_f1_match = re.search(binary_f1_pattern, content, re.DOTALL)
        if binary_f1_match:
            metrics_data['word_binary_f1'] = float(binary_f1_match.group(1))
        
        binary_accuracy_pattern = r'=== BINARY CLASSIFICATION.*?Accuracy:\s*([\d.]+)'
        binary_accuracy_match = re.search(binary_accuracy_pattern, content, re.DOTALL)
        if binary_accuracy_match:
            metrics_data['word_binary_accuracy'] = float(binary_accuracy_match.group(1))
        
        binary_precision_pattern = r'=== BINARY CLASSIFICATION.*?Precision:\s*([\d.]+)'
        binary_recall_pattern = r'=== BINARY CLASSIFICATION.*?Recall:\s*([\d.]+)'
        
        precision_match = re.search(binary_precision_pattern, content, re.DOTALL)
        recall_match = re.search(binary_recall_pattern, content, re.DOTALL)
        
        if precision_match:
            metrics_data['word_binary_precision'] = float(precision_match.group(1))
        if recall_match:
            metrics_data['word_binary_recall'] = float(recall_match.group(1))
        
        # Extract Multiclass Classification metrics
        multiclass_f1_pattern = r'=== MULTICLASS CLASSIFICATION.*?F1-Score:\s*([\d.]+)'
        multiclass_f1_match = re.search(multiclass_f1_pattern, content, re.DOTALL)
        if multiclass_f1_match:
            metrics_data['word_multiclass_f1'] = float(multiclass_f1_match.group(1))
        
        multiclass_accuracy_pattern = r'=== MULTICLASS CLASSIFICATION.*?Accuracy:\s*([\d.]+)'
        multiclass_accuracy_match = re.search(multiclass_accuracy_pattern, content, re.DOTALL)
        if multiclass_accuracy_match:
            metrics_data['word_multiclass_accuracy'] = float(multiclass_accuracy_match.group(1))
        
        return metrics_data
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return {}

def extract_iou_eval_metrics(note_file):
    """Extract metrics from IoU evaluation note files"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metrics_data = {}
        
        # Extract Combined Mean IoU
        combined_iou_pattern = r'Combined Mean IoU:\s*([\d.]+)'
        combined_match = re.search(combined_iou_pattern, content)
        if combined_match:
            metrics_data['combined_mean_iou_percent'] = float(combined_match.group(1)) * 100
        
        # Extract Individual Mean IoU
        individual_iou_pattern = r'Individual Mean IoU:\s*([\d.]+)'
        individual_match = re.search(individual_iou_pattern, content)
        if individual_match:
            metrics_data['individual_mean_iou_percent'] = float(individual_match.group(1)) * 100
        
        # Extract F1 scores at different IoU thresholds
        thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
        
        for threshold in thresholds:
            f1_pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?F1-Score:\s*([\d.]+)'
            f1_match = re.search(f1_pattern, content, re.DOTALL)
            if f1_match:
                metrics_data[f'iou_f1_{threshold}'] = float(f1_match.group(1))
            
            precision_pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?Precision:\s*([\d.]+)'
            recall_pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?Recall:\s*([\d.]+)'
            
            precision_match = re.search(precision_pattern, content, re.DOTALL)
            recall_match = re.search(recall_pattern, content, re.DOTALL)
            
            if precision_match:
                metrics_data[f'iou_precision_{threshold}'] = float(precision_match.group(1))
            if recall_match:
                metrics_data[f'iou_recall_{threshold}'] = float(recall_match.group(1))
        
        return metrics_data
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return {}

def extract_word_iou_eval_metrics(note_file):
    """Extract metrics from Word-IoU evaluation note files"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metrics_data = {}
        
        # Extract F1 scores at different IoU thresholds
        thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
        
        for threshold in thresholds:
            f1_pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?F1-Score:\s*([\d.]+)'
            f1_match = re.search(f1_pattern, content, re.DOTALL)
            if f1_match:
                metrics_data[f'word_iou_f1_{threshold}'] = float(f1_match.group(1))
            
            precision_pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?Precision:\s*([\d.]+)'
            recall_pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?Recall:\s*([\d.]+)'
            
            precision_match = re.search(precision_pattern, content, re.DOTALL)
            recall_match = re.search(recall_pattern, content, re.DOTALL)
            
            if precision_match:
                metrics_data[f'word_iou_precision_{threshold}'] = float(precision_match.group(1))
            if recall_match:
                metrics_data[f'word_iou_recall_{threshold}'] = float(recall_match.group(1))
        
        return metrics_data
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return {}

def create_single_heatmap(data, col_name, output_dir, title, filename, cbar_label, fmt='.3f'):
    """Create a single heatmap"""
    
    # Create pivot table for heatmap
    pivot_data = data.pivot_table(
        values=col_name,
        index='stride_numeric',
        columns='window_numeric',
        aggfunc='mean'
    )
    
    if pivot_data.empty:
        print(f"No data for {col_name}")
        return
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    
    # Use a color map that emphasizes higher values
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=fmt,
        cmap='YlOrRd',
        cbar_kws={'label': cbar_label},
        square=False,
        linewidths=0.5,
        annot_kws={'size': 10}
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Stride Size (seconds/percent)', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved: {filename}")

def create_comprehensive_heatmaps(df, output_dir):
    """Create comprehensive heatmaps for all methods and evaluation types"""
    
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type].copy()
        
        # Create subdirectory for this evaluation type
        eval_output_dir = output_dir / eval_type
        eval_output_dir.mkdir(exist_ok=True)
        
        print(f"\n🎯 Creating comprehensive heatmaps for {eval_type}")
        
        for method in eval_data['method'].unique():
            method_data = eval_data[eval_data['method'] == method].copy()
            
            if method_data.empty:
                continue
            
            print(f"  📊 Processing {method}")
            
            if method == 'window_eval':
                # Window evaluation metrics
                metrics = [
                    ('window_binary_f1', 'Window Binary F1 Score'),
                    ('window_binary_accuracy', 'Window Binary Accuracy'),
                    ('window_binary_precision', 'Window Binary Precision'),
                    ('window_binary_recall', 'Window Binary Recall'),
                    ('window_multiclass_f1', 'Window Multiclass F1 Score'),
                    ('window_multiclass_accuracy', 'Window Multiclass Accuracy')
                ]
                
                for col_name, metric_name in metrics:
                    if col_name in method_data.columns:
                        create_single_heatmap(
                            method_data, col_name, eval_output_dir,
                            f'{eval_type} - Window Evaluation - {metric_name}',
                            f"{eval_type}_{method}_{col_name}_heatmap.png",
                            metric_name
                        )
            
            elif method == 'word_eval':
                # Word evaluation metrics
                metrics = [
                    ('word_binary_f1', 'Word Binary F1 Score'),
                    ('word_binary_accuracy', 'Word Binary Accuracy'),
                    ('word_binary_precision', 'Word Binary Precision'),
                    ('word_binary_recall', 'Word Binary Recall'),
                    ('word_multiclass_f1', 'Word Multiclass F1 Score'),
                    ('word_multiclass_accuracy', 'Word Multiclass Accuracy')
                ]
                
                for col_name, metric_name in metrics:
                    if col_name in method_data.columns:
                        create_single_heatmap(
                            method_data, col_name, eval_output_dir,
                            f'{eval_type} - Word Evaluation - {metric_name}',
                            f"{eval_type}_{method}_{col_name}_heatmap.png",
                            metric_name
                        )
            
            elif method == 'iou_eval':
                # IoU evaluation metrics
                iou_metrics = [
                    ('combined_mean_iou_percent', 'Combined Mean IoU Percentage'),
                    ('individual_mean_iou_percent', 'Individual Mean IoU Percentage')
                ]
                
                for col_name, metric_name in iou_metrics:
                    if col_name in method_data.columns:
                        create_single_heatmap(
                            method_data, col_name, eval_output_dir,
                            f'{eval_type} - IoU Evaluation - {metric_name}',
                            f"{eval_type}_{method}_{col_name}_heatmap.png",
                            f'{metric_name} (%)',
                            fmt='.2f'
                        )
                
                # F1 scores at IoU thresholds
                thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
                for threshold in thresholds:
                    col_name = f'iou_f1_{threshold}'
                    if col_name in method_data.columns:
                        create_single_heatmap(
                            method_data, col_name, eval_output_dir,
                            f'{eval_type} - IoU Evaluation - F1 Score at IoU ≥ {threshold}',
                            f"{eval_type}_{method}_f1_iou_{threshold}_heatmap.png",
                            f'F1 Score (IoU ≥ {threshold})'
                        )
            
            elif method == 'word_iou_eval':
                # Word-IoU evaluation metrics
                thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
                
                for threshold in thresholds:
                    col_name = f'word_iou_f1_{threshold}'
                    if col_name in method_data.columns:
                        create_single_heatmap(
                            method_data, col_name, eval_output_dir,
                            f'{eval_type} - Word-IoU Evaluation - F1 Score at IoU ≥ {threshold}',
                            f"{eval_type}_{method}_f1_iou_{threshold}_heatmap.png",
                            f'F1 Score (IoU ≥ {threshold})'
                        )

def create_method_comparison_graphs(df, output_dir):
    """Create comparison graphs between different methods for each evaluation type"""
    
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type].copy()
        
        print(f"\n📈 Creating method comparison graphs for {eval_type}")
        
        # Compare F1 scores across methods
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        methods_info = [
            ('window_eval', 'window_binary_f1', 'Window Binary F1 Score'),
            ('word_eval', 'word_binary_f1', 'Word Binary F1 Score'),
            ('iou_eval', 'iou_f1_0.3', 'IoU F1 Score (≥0.3)'),
            ('word_iou_eval', 'word_iou_f1_0.3', 'Word-IoU F1 Score (≥0.3)')
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (method, metric_col, title) in enumerate(methods_info):
            method_data = eval_data[eval_data['method'] == method]
            
            if method_data.empty or metric_col not in method_data.columns:
                axes[i].text(0.5, 0.5, f'No data for {title}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(title, fontweight='bold')
                continue
            
            # Group by window size and calculate mean/std
            window_stats = method_data.groupby('window_numeric')[metric_col].agg(['mean', 'std']).reset_index()
            
            # Plot line with error bars
            axes[i].errorbar(window_stats['window_numeric'], window_stats['mean'], 
                           yerr=window_stats['std'], 
                           marker='o', linewidth=2, capsize=5, alpha=0.8,
                           color=colors[i])
            
            axes[i].set_xlabel('Window Size (seconds)', fontweight='bold')
            axes[i].set_ylabel('Score', fontweight='bold')
            axes[i].set_title(title, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(f'{eval_type} - Method Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / f"{eval_type}_method_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_statistics(df, output_dir):
    """Create comprehensive summary statistics for all methods"""
    
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type].copy()
        
        summary_stats = []
        
        for method in eval_data['method'].unique():
            method_data = eval_data[eval_data['method'] == method]
            
            # Get all numeric columns for this method
            numeric_cols = method_data.select_dtypes(include=[np.number]).columns
            excluded_cols = ['window_numeric', 'stride_numeric']
            metric_cols = [col for col in numeric_cols if col not in excluded_cols]
            
            for col in metric_cols:
                if col in method_data.columns and not method_data[col].isna().all():
                    stats = method_data[col].describe()
                    best_config = method_data.loc[method_data[col].idxmax()]
                    
                    summary_stats.append({
                        'eval_type': eval_type,
                        'method': method,
                        'metric': col,
                        'mean_value': stats['mean'],
                        'std_value': stats['std'],
                        'min_value': stats['min'],
                        'max_value': stats['max'],
                        'median_value': stats['50%'],
                        'best_window': best_config['window'],
                        'best_stride': best_config['stride'],
                        'best_value': best_config[col],
                        'total_configs': len(method_data)
                    })
        
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            summary_df = summary_df.round(4)
            summary_df.to_csv(output_dir / f"{eval_type}_comprehensive_summary.csv", index=False)
            
            print(f"💾 Saved: {eval_type}_comprehensive_summary.csv")

def main():
    """Main function to run comprehensive evaluation analysis"""
    
    print("=== CREATING COMPREHENSIVE EVALUATION HEATMAP ANALYSIS ===\n")
    
    # Extract data from all methods
    df = extract_all_evaluation_data()
    
    if df.empty:
        print("❌ No evaluation data extracted!")
        return
    
    print(f"✅ Extracted data from {len(df)} configurations")
    print(f"   Evaluation types: {list(df['eval_type'].unique())}")
    print(f"   Methods: {list(df['method'].unique())}")
    
    # Create output directory
    output_dir = Path("comprehensive_evaluation_heatmaps")
    output_dir.mkdir(exist_ok=True)
    
    # Create analysis components
    create_comprehensive_heatmaps(df, output_dir)
    create_method_comparison_graphs(df, output_dir)
    create_summary_statistics(df, output_dir)
    
    print(f"\n🎉 Comprehensive evaluation heatmap analysis completed successfully!")
    print(f"📁 Output directory: {output_dir}")
    
    # Print quick summary for each method and evaluation type
    print("\n📊 QUICK SUMMARY:")
    for eval_type in df['eval_type'].unique():
        print(f"\n{eval_type.upper()}:")
        eval_data = df[df['eval_type'] == eval_type]
        
        for method in eval_data['method'].unique():
            method_data = eval_data[eval_data['method'] == method]
            print(f"  {method}: {len(method_data)} configurations")
            
            # Show best metric for each method
            if method == 'window_eval' and 'window_binary_f1' in method_data.columns:
                best = method_data.loc[method_data['window_binary_f1'].idxmax()]
                print(f"    Best Window F1: {best['window_binary_f1']:.4f} ({best['window']}, {best['stride']})")
            
            elif method == 'word_eval' and 'word_binary_f1' in method_data.columns:
                best = method_data.loc[method_data['word_binary_f1'].idxmax()]
                print(f"    Best Word F1: {best['word_binary_f1']:.4f} ({best['window']}, {best['stride']})")
            
            elif method == 'iou_eval' and 'combined_mean_iou_percent' in method_data.columns:
                best = method_data.loc[method_data['combined_mean_iou_percent'].idxmax()]
                print(f"    Best IoU: {best['combined_mean_iou_percent']:.2f}% ({best['window']}, {best['stride']})")
            
            elif method == 'word_iou_eval' and 'word_iou_f1_0.3' in method_data.columns:
                best = method_data.loc[method_data['word_iou_f1_0.3'].idxmax()]
                print(f"    Best Word-IoU F1: {best['word_iou_f1_0.3']:.4f} ({best['window']}, {best['stride']})")

if __name__ == "__main__":
    main()

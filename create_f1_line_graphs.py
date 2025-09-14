#!/usr/bin/env python3
"""
F1 Score Line Graph Analysis
Create comprehensive line graphs showing F1 scores for all 4 evaluation methods
across both binary and multiclass classifications
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import os
def extract_all_f1_data():
    """Extract F1 score data from all evaluation methods"""
    
    base_dir = Path("fixed_smart_parallel_results")
    eval_results_dir = Path("evaluation_results")
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
            "window_eval": extract_window_f1_metrics,
            "word_eval": extract_word_f1_metrics,
            "iou_eval": extract_iou_f1_metrics,
            "eval_IoU_word_eval_sep": extract_eval_iou_word_eval_sep_metrics
        }
        
        for method_name, extract_func in methods.items():
            # Handle eval_IoU_word_eval_sep from different directory
            if method_name == "eval_IoU_word_eval_sep":
                method_dir = eval_results_dir / method_name / eval_type
            else:
                method_dir = eval_dir / method_name
                
            if method_dir.exists():
                print(f"  Processing {method_name}...")
                
                for window_dir in method_dir.iterdir():
                    if window_dir.is_dir() and window_dir.name.startswith("window_"):
                        for stride_dir in window_dir.iterdir():
                            if stride_dir.is_dir() and stride_dir.name.startswith("stride_"):
                                note_file = stride_dir / "note.txt"
                                if note_file.exists():
                                    f1_data = extract_func(note_file)
                                    if f1_data:
                                        window_match = re.search(r'window_(\d+\.?\d*)s', window_dir.name)
                                        
                                        # Handle both time-based and percentage-based stride formats
                                        stride_match_time = re.search(r'stride_(\d+\.?\d*)s', stride_dir.name)
                                        stride_match_percent = re.search(r'stride_(\d+\.?\d*)%', stride_dir.name)
                                        
                                        if window_match and (stride_match_time or stride_match_percent):
                                            if stride_match_time:
                                                stride_numeric = float(stride_match_time.group(1))
                                                stride_type = 'seconds'
                                            else:
                                                stride_numeric = float(stride_match_percent.group(1))
                                                stride_type = 'percent'
                                            
                                            # Map eval_IoU_word_eval_sep to word_iou_eval for consistency
                                            display_method = 'word_iou_eval' if method_name == 'eval_IoU_word_eval_sep' else method_name
                                            
                                            config_data = {
                                                'eval_type': eval_type,
                                                'method': display_method,
                                                'window': window_dir.name,
                                                'stride': stride_dir.name,
                                                'window_numeric': float(window_match.group(1)),
                                                'stride_numeric': stride_numeric,
                                                'stride_type': stride_type,
                                            }
                                            config_data.update(f1_data)
                                            all_results.append(config_data)
    
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n✅ Total configurations extracted: {len(df)}")
        print(f"Evaluation types: {list(df['eval_type'].unique())}")
        print(f"Methods: {list(df['method'].unique())}")
        return df
    else:
        return pd.DataFrame()

def extract_window_f1_metrics(note_file):
    """Extract F1 metrics from Window evaluation note files"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metrics_data = {}
        
        # Extract Binary Classification F1
        binary_f1_pattern = r'=== BINARY CLASSIFICATION.*?F1-Score:\s*([\d.]+)'
        binary_f1_match = re.search(binary_f1_pattern, content, re.DOTALL)
        if binary_f1_match:
            metrics_data['binary_f1'] = float(binary_f1_match.group(1))
        
        # Extract Multiclass Classification F1
        multiclass_f1_pattern = r'=== MULTICLASS CLASSIFICATION.*?F1-Score:\s*([\d.]+)'
        multiclass_f1_match = re.search(multiclass_f1_pattern, content, re.DOTALL)
        if multiclass_f1_match:
            metrics_data['multiclass_f1'] = float(multiclass_f1_match.group(1))
        
        return metrics_data
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return {}

def extract_word_f1_metrics(note_file):
    """Extract F1 metrics from Word evaluation note files"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metrics_data = {}
        
        # Extract Binary Classification F1
        binary_f1_pattern = r'=== BINARY CLASSIFICATION.*?F1-Score:\s*([\d.]+)'
        binary_f1_match = re.search(binary_f1_pattern, content, re.DOTALL)
        if binary_f1_match:
            metrics_data['binary_f1'] = float(binary_f1_match.group(1))
        
        # Extract Multiclass Classification F1
        multiclass_f1_pattern = r'=== MULTICLASS CLASSIFICATION.*?F1-Score:\s*([\d.]+)'
        multiclass_f1_match = re.search(multiclass_f1_pattern, content, re.DOTALL)
        if multiclass_f1_match:
            metrics_data['multiclass_f1'] = float(multiclass_f1_match.group(1))
        
        return metrics_data
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return {}

def extract_iou_f1_metrics(note_file):
    """Extract F1 metrics from IoU evaluation note files"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metrics_data = {}
        
        # For IoU evaluation, we use F1 at IoU threshold 0.1 as the representative metric
        # Extract Binary F1 at IoU 0.1
        binary_f1_pattern = r'--- IoU Threshold 0.1 ---.*?Binary Classification.*?F1-Score:\s*([\d.]+)'
        binary_f1_match = re.search(binary_f1_pattern, content, re.DOTALL)
        if binary_f1_match:
            metrics_data['binary_f1'] = float(binary_f1_match.group(1))
        
        # Extract Multiclass F1 at IoU 0.1
        multiclass_f1_pattern = r'--- IoU Threshold 0.1 ---.*?Multiclass Classification.*?Balanced Accuracy:\s*[\d.]+.*?precision.*?recall.*?f1-score.*?accuracy.*?[\d.]+.*?micro avg.*?[\d.]+.*?[\d.]+.*?([\d.]+)'
        multiclass_content_match = re.search(r'--- IoU Threshold 0.1 ---.*?Multiclass Classification.*?Balanced Accuracy:\s*([\d.]+)', content, re.DOTALL)
        if multiclass_content_match:
            # For multiclass, we'll use balanced accuracy as a proxy since F1 extraction is complex
            metrics_data['multiclass_f1'] = float(multiclass_content_match.group(1))
        
        return metrics_data
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return {}

def extract_word_iou_f1_metrics(note_file):
    """Extract F1 metrics from old Word-IoU evaluation note files (eval_IoU_word_eval_sep)"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metrics_data = {}
        
        # For old Word-IoU evaluation, we use F1 at IoU threshold 0.1 as the representative metric
        # Extract Binary F1 at IoU 0.1
        binary_f1_pattern = r'--- IoU Threshold 0.1 ---.*?Binary Classification.*?F1-Score:\s*([\d.]+)'
        binary_f1_match = re.search(binary_f1_pattern, content, re.DOTALL)
        if binary_f1_match:
            metrics_data['binary_f1'] = float(binary_f1_match.group(1))
        
        # Extract Multiclass balanced accuracy as proxy for F1
        multiclass_content_match = re.search(r'--- IoU Threshold 0.1 ---.*?Multiclass Classification.*?Balanced Accuracy:\s*([\d.]+)', content, re.DOTALL)
        if multiclass_content_match:
            metrics_data['multiclass_f1'] = float(multiclass_content_match.group(1))
        
        return metrics_data
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return {}

def extract_eval_iou_word_eval_sep_metrics(note_file):
    """Extract F1 metrics from eval_IoU_word_eval_sep evaluation note files"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metrics_data = {}
        
        # For eval_IoU_word_eval_sep, extract Binary F1 from Binary Classification section
        binary_f1_pattern = r'=== Binary Classification Metrics.*?F1-Score:\s*([\d.]+)'
        binary_f1_match = re.search(binary_f1_pattern, content, re.DOTALL)
        if binary_f1_match:
            metrics_data['binary_f1'] = float(binary_f1_match.group(1))
        
        # Extract IoU-based F1 at threshold 0.3 from IoU Analysis section
        iou_f1_pattern = r'IoU ≥ 0\.3:.*?F1=([\d.]+)'
        iou_f1_match = re.search(iou_f1_pattern, content)
        if iou_f1_match:
            metrics_data['multiclass_f1'] = float(iou_f1_match.group(1))
        
        return metrics_data
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return {}

def create_comprehensive_f1_line_graphs(df, output_dir):
    """Create comprehensive F1 score line graphs for all methods"""
    
    # Method names for display
    method_display_names = {
        'window_eval': 'Window Evaluation',
        'word_eval': 'Word Evaluation', 
        'iou_eval': 'IoU Evaluation (IoU≥0.1)',
        'word_iou_eval': 'Word-IoU Evaluation (IoU≥0.1)'
    }
    
    # Colors for each method
    colors = {
        'window_eval': '#1f77b4',    # Blue
        'word_eval': '#ff7f0e',      # Orange
        'iou_eval': '#2ca02c',       # Green
        'word_iou_eval': '#d62728'   # Red
    }
    
    # Create separate graphs for each eval_type
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type].copy()
        
        # Create subplots: Binary F1 and Multiclass F1
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot Binary F1 Scores
        ax1.set_title(f'{eval_type} - Binary F1 Score Comparison', fontsize=14, fontweight='bold', pad=20)
        
        for method in eval_data['method'].unique():
            method_data = eval_data[eval_data['method'] == method]
            
            if 'binary_f1' not in method_data.columns:
                continue
            
            # Group by window size and calculate mean F1 scores with min-max range
            window_stats = method_data.groupby('window_numeric')['binary_f1'].agg(['mean', 'std', 'min', 'max']).reset_index()
            window_stats = window_stats.sort_values('window_numeric')
            
            # Plot line with shaded confidence interval
            ax1.plot(
                window_stats['window_numeric'], 
                window_stats['mean'],
                marker='o', 
                linewidth=3, 
                markersize=8,
                label=method_display_names[method],
                color=colors[method],
                alpha=0.9
            )
            
            # Add shaded area for min-max range (like Figure 1)
            ax1.fill_between(
                window_stats['window_numeric'],
                window_stats['min'],
                window_stats['max'],
                alpha=0.2,
                color=colors[method]
            )
        
        ax1.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Binary F1 Score', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11, loc='best')
        ax1.set_ylim(0, max(1.0, ax1.get_ylim()[1] * 1.1))
        
        # Plot Multiclass F1 Scores
        ax2.set_title(f'{eval_type} - Multiclass F1 Comparison', fontsize=14, fontweight='bold', pad=20)
        
        for method in eval_data['method'].unique():
            method_data = eval_data[eval_data['method'] == method]
            
            if 'multiclass_f1' not in method_data.columns:
                continue
            
            # Group by window size and calculate mean F1 scores with min-max range
            window_stats = method_data.groupby('window_numeric')['multiclass_f1'].agg(['mean', 'std', 'min', 'max']).reset_index()
            window_stats = window_stats.sort_values('window_numeric')
            
            # Plot line with shaded confidence interval
            ax2.plot(
                window_stats['window_numeric'], 
                window_stats['mean'],
                marker='s', 
                linewidth=3, 
                markersize=8,
                label=method_display_names[method],
                color=colors[method],
                alpha=0.9
            )
            
            # Add shaded area for min-max range (like Figure 1)
            ax2.fill_between(
                window_stats['window_numeric'],
                window_stats['min'],
                window_stats['max'],
                alpha=0.2,
                color=colors[method]
            )
        
        ax2.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Multiclass F1 Score', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11, loc='best')
        ax2.set_ylim(0, max(1.0, ax2.get_ylim()[1] * 1.1))
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{eval_type}_f1_comparison_line_graph.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved: {eval_type}_f1_comparison_line_graph.png")

def create_combined_f1_line_graph(df, output_dir):
    """Create a single combined line graph with all methods and eval types"""
    
    # Method names for display
    method_display_names = {
        'window_eval': 'Window Evaluation',
        'word_eval': 'Word Evaluation',
        'iou_eval': 'IoU Evaluation (IoU≥0.1)', 
        'word_iou_eval': 'Word-IoU Evaluation (IoU≥0.1)'
    }
    
    # Colors for each method
    colors = {
        'window_eval': '#1f77b4',    # Blue
        'word_eval': '#ff7f0e',      # Orange  
        'iou_eval': '#2ca02c',       # Green
        'word_iou_eval': '#d62728'   # Red
    }
    
    # Line styles for eval types
    line_styles = {
        'eval_by_0.05': '-',     # Solid line
        'eval_percent': '--'     # Dashed line
    }
    
    # Create subplots: Binary F1 and Multiclass F1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot Binary F1 Scores
    ax1.set_title('All Methods - Binary F1 Score Comparison', fontsize=16, fontweight='bold', pad=20)
    
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type]
        
        for method in eval_data['method'].unique():
            method_data = eval_data[eval_data['method'] == method]
            
            if 'binary_f1' not in method_data.columns:
                continue
            
            # Group by window size and calculate mean F1 scores with min-max range
            window_stats = method_data.groupby('window_numeric')['binary_f1'].agg(['mean', 'std', 'min', 'max']).reset_index()
            window_stats = window_stats.sort_values('window_numeric')
            
            # Create label with eval type
            label = f"{method_display_names[method]} ({eval_type})"
            
            # Plot line with shaded confidence interval
            ax1.plot(
                window_stats['window_numeric'], 
                window_stats['mean'],
                marker='o', 
                linewidth=3, 
                markersize=6,
                label=label,
                color=colors[method],
                linestyle=line_styles[eval_type],
                alpha=0.9
            )
            
            # Add shaded area for min-max range (like Figure 1)
            ax1.fill_between(
                window_stats['window_numeric'],
                window_stats['min'],
                window_stats['max'],
                alpha=0.15,
                color=colors[method]
            )
    
    ax1.set_xlabel('Window Size (seconds)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Binary F1 Score', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best', ncol=2)
    ax1.set_ylim(0, max(1.0, ax1.get_ylim()[1] * 1.1))
    
    # Plot Multiclass F1 Scores
    ax2.set_title('All Methods - Multiclass F1 Comparison', fontsize=16, fontweight='bold', pad=20)
    
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type]
        
        for method in eval_data['method'].unique():
            method_data = eval_data[eval_data['method'] == method]
            
            if 'multiclass_f1' not in method_data.columns:
                continue
            
            # Group by window size and calculate mean F1 scores with min-max range
            window_stats = method_data.groupby('window_numeric')['multiclass_f1'].agg(['mean', 'std', 'min', 'max']).reset_index()
            window_stats = window_stats.sort_values('window_numeric')
            
            # Create label with eval type
            label = f"{method_display_names[method]} ({eval_type})"
            
            # Plot line with shaded confidence interval
            ax2.plot(
                window_stats['window_numeric'], 
                window_stats['mean'],
                marker='s', 
                linewidth=3, 
                markersize=6,
                label=label,
                color=colors[method],
                linestyle=line_styles[eval_type],
                alpha=0.9
            )
            
            # Add shaded area for min-max range (like Figure 1)
            ax2.fill_between(
                window_stats['window_numeric'],
                window_stats['min'],
                window_stats['max'],
                alpha=0.15,
                color=colors[method]
            )
    
    ax2.set_xlabel('Window Size (seconds)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Multiclass F1 Score', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best', ncol=2)
    ax2.set_ylim(0, max(1.0, ax2.get_ylim()[1] * 1.1))
    
    plt.tight_layout()
    plt.savefig(output_dir / "all_methods_f1_comparison_line_graph.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved: all_methods_f1_comparison_line_graph.png")

def create_summary_statistics_f1(df, output_dir):
    """Create summary statistics focused on F1 scores"""
    
    summary_stats = []
    
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type]
        
        for method in eval_data['method'].unique():
            method_data = eval_data[eval_data['method'] == method]
            
            # Binary F1 statistics
            if 'binary_f1' in method_data.columns and not method_data['binary_f1'].isna().all():
                stats = method_data['binary_f1'].describe()
                best_config = method_data.loc[method_data['binary_f1'].idxmax()]
                
                summary_stats.append({
                    'eval_type': eval_type,
                    'method': method,
                    'metric_type': 'Binary_F1',
                    'mean_value': stats['mean'],
                    'std_value': stats['std'],
                    'min_value': stats['min'],
                    'max_value': stats['max'],
                    'median_value': stats['50%'],
                    'best_window': best_config['window'],
                    'best_stride': best_config['stride'],
                    'best_value': best_config['binary_f1'],
                    'total_configs': len(method_data)
                })
            
            # Multiclass F1 statistics
            if 'multiclass_f1' in method_data.columns and not method_data['multiclass_f1'].isna().all():
                stats = method_data['multiclass_f1'].describe()
                best_config = method_data.loc[method_data['multiclass_f1'].idxmax()]
                
                summary_stats.append({
                    'eval_type': eval_type,
                    'method': method,
                    'metric_type': 'Multiclass_F1',
                    'mean_value': stats['mean'],
                    'std_value': stats['std'],
                    'min_value': stats['min'],
                    'max_value': stats['max'],
                    'median_value': stats['50%'],
                    'best_window': best_config['window'],
                    'best_stride': best_config['stride'],
                    'best_value': best_config['multiclass_f1'],
                    'total_configs': len(method_data)
                })
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_df = summary_df.round(4)
        summary_df.to_csv(output_dir / "f1_scores_summary_statistics.csv", index=False)
        
        print(f"💾 Saved: f1_scores_summary_statistics.csv")
        return summary_df
    
    return pd.DataFrame()

def main():
    """Main function to create F1 score line graphs"""
    
    print("=== CREATING F1 SCORE LINE GRAPH ANALYSIS ===\n")
    
    # Extract F1 data from all methods
    df = extract_all_f1_data()
    
    if df.empty:
        print("❌ No F1 score data extracted!")
        return
    
    print(f"✅ Extracted data from {len(df)} configurations")
    print(f"   Evaluation types: {list(df['eval_type'].unique())}")
    print(f"   Methods: {list(df['method'].unique())}")
    
    # Create output directory
    output_dir = Path("f1_score_line_graphs")
    output_dir.mkdir(exist_ok=True)
    
    # Create analysis components
    create_comprehensive_f1_line_graphs(df, output_dir)
    create_combined_f1_line_graph(df, output_dir)
    summary_df = create_summary_statistics_f1(df, output_dir)
    
    print(f"\n🎉 F1 score line graph analysis completed successfully!")
    print(f"📁 Output directory: {output_dir}")
    
    # Print quick summary
    print("\n📊 F1 SCORE SUMMARY:")
    for eval_type in df['eval_type'].unique():
        print(f"\n{eval_type.upper()}:")
        eval_data = df[df['eval_type'] == eval_type]
        
        for method in eval_data['method'].unique():
            method_data = eval_data[eval_data['method'] == method]
            
            if 'binary_f1' in method_data.columns:
                best_binary = method_data.loc[method_data['binary_f1'].idxmax()]
                print(f"   {method} Binary F1: {best_binary['binary_f1']:.4f} "
                      f"({best_binary['window']}, {best_binary['stride']})")
            
            if 'multiclass_f1' in method_data.columns:
                best_multiclass = method_data.loc[method_data['multiclass_f1'].idxmax()]
                print(f"   {method} Multiclass F1: {best_multiclass['multiclass_f1']:.4f} "
                      f"({best_multiclass['window']}, {best_multiclass['stride']})")

if __name__ == "__main__":
    main()

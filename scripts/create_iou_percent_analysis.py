#!/usr/bin/env python3
"""
IoU Percentage Analysis
Create graphs and CSV files showing IoU percentage performance across window/stride configurations
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
                                    stride_match = re.search(r'stride_(\d+\.?\d*)s', stride_dir.name)
                                    
                                    if window_match and stride_match:
                                        config_data = {
                                            'eval_type': eval_type,
                                            'method': 'word_iou_eval',
                                            'window': window_dir.name,
                                            'stride': stride_dir.name,
                                            'window_numeric': float(window_match.group(1)),
                                            'stride_numeric': float(stride_match.group(1)),
                                            'config': f"{window_dir.name}_{stride_dir.name}",
                                            **iou_data
                                        }
                                        all_results.append(config_data)
        
        # Method 3: IoU Evaluation
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
                                    stride_match = re.search(r'stride_(\d+\.?\d*)s', stride_dir.name)
                                    
                                    if window_match and stride_match:
                                        config_data = {
                                            'eval_type': eval_type,
                                            'method': 'iou_eval',
                                            'window': window_dir.name,
                                            'stride': stride_dir.name,
                                            'window_numeric': float(window_match.group(1)),
                                            'stride_numeric': float(stride_match.group(1)),
                                            'config': f"{window_dir.name}_{stride_dir.name}",
                                            **iou_data
                                        }
                                        all_results.append(config_data)
    
    return pd.DataFrame(all_results)

def extract_word_iou_percentages(note_file):
    """Extract IoU metrics from word-IoU evaluation note files"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        iou_data = {}
        
        # Extract F1 scores at different IoU thresholds (used as IoU performance metric)
        thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
        
        for threshold in thresholds:
            # Extract F1 score as primary IoU performance metric
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

def create_iou_percent_heatmaps(df, output_dir):
    """Create heatmaps showing IoU performance across window/stride combinations"""
    
    # For Word-IoU Evaluation: use F1 scores at different thresholds
    # For IoU Evaluation: use Mean IoU percentages and F1 scores
    
    methods = [('word_iou_eval', 'Word-IoU Evaluation'), ('iou_eval', 'IoU Evaluation')]
    
    for method, method_name in methods:
        method_data = df[df['method'] == method].copy()
        if method_data.empty:
            continue
        
        print(f"📊 Creating heatmaps for {method_name}")
        
        if method == 'word_iou_eval':
            # Use F1 scores at different IoU thresholds
            thresholds = ['0.3', '0.5', '0.9']
            for threshold in thresholds:
                col_name = f'iou_f1_{threshold}'
                if col_name not in method_data.columns:
                    continue
                
                create_single_heatmap(method_data, col_name, output_dir, 
                                    f'{method_name} - F1 Score at IoU ≥ {threshold}',
                                    f"{method.lower()}_f1_iou_{threshold}_heatmap.png",
                                    f'F1 Score (IoU ≥ {threshold})')
        
        elif method == 'iou_eval':
            # Use Mean IoU percentages
            iou_metrics = [
                ('combined_mean_iou_percent', 'Combined Mean IoU Percentage'),
                ('individual_mean_iou_percent', 'Individual Mean IoU Percentage')
            ]
            
            for col_name, title_suffix in iou_metrics:
                if col_name not in method_data.columns:
                    continue
                
                create_single_heatmap(method_data, col_name, output_dir,
                                    f'{method_name} - {title_suffix}',
                                    f"{method.lower()}_{col_name}_heatmap.png",
                                    f'{title_suffix} (%)')
            
            # Also create F1 score heatmaps for IoU evaluation
            thresholds = ['0.3', '0.5', '0.9']
            for threshold in thresholds:
                col_name = f'iou_f1_{threshold}'
                if col_name not in method_data.columns:
                    continue
                
                create_single_heatmap(method_data, col_name, output_dir,
                                    f'{method_name} - F1 Score at IoU ≥ {threshold}',
                                    f"{method.lower()}_f1_iou_{threshold}_heatmap.png",
                                    f'F1 Score (IoU ≥ {threshold})')

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
        linewidths=0.5
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Stride Size (seconds)', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved: {filename}")

def create_iou_percent_trend_graphs(df, output_dir):
    """Create line graphs showing IoU performance trends"""
    
    methods = [('word_iou_eval', 'Word-IoU Evaluation'), ('iou_eval', 'IoU Evaluation')]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for method, method_name in methods:
        method_data = df[df['method'] == method].copy()
        if method_data.empty:
            continue
        
        print(f"📈 Creating trend graphs for {method_name}")
        
        if method == 'word_iou_eval':
            # Create F1 score trends at different IoU thresholds
            thresholds = ['0.3', '0.5', '0.9']
            
            # Group by window size
            agg_dict = {f'iou_f1_{t}': ['mean', 'std', 'count'] for t in thresholds}
            window_stats = method_data.groupby('window_numeric').agg(agg_dict).round(3)
            
            # Flatten column names
            window_stats.columns = ['_'.join(col).strip() for col in window_stats.columns]
            window_stats = window_stats.reset_index()
            
            # Create trend plot
            plt.figure(figsize=(12, 8))
            
            for i, threshold in enumerate(thresholds):
                mean_col = f'iou_f1_{threshold}_mean'
                std_col = f'iou_f1_{threshold}_std'
                
                if mean_col in window_stats.columns:
                    plt.plot(window_stats['window_numeric'], window_stats[mean_col],
                            marker='o', linewidth=3, markersize=8,
                            color=colors[i], label=f'IoU ≥ {threshold}', alpha=0.8)
                    
                    # Add error bars
                    if std_col in window_stats.columns:
                        plt.fill_between(window_stats['window_numeric'],
                                       window_stats[mean_col] - window_stats[std_col],
                                       window_stats[mean_col] + window_stats[std_col],
                                       alpha=0.2, color=colors[i])
            
            plt.xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
            plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
            plt.title(f'{method_name} - F1 Score Trends by IoU Threshold', fontsize=14, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
            
        elif method == 'iou_eval':
            # Create Mean IoU percentage trends
            iou_metrics = ['combined_mean_iou_percent', 'individual_mean_iou_percent']
            available_metrics = [col for col in iou_metrics if col in method_data.columns]
            
            if available_metrics:
                # Group by window size
                agg_dict = {metric: ['mean', 'std', 'count'] for metric in available_metrics}
                window_stats = method_data.groupby('window_numeric').agg(agg_dict).round(2)
                
                # Flatten column names
                window_stats.columns = ['_'.join(col).strip() for col in window_stats.columns]
                window_stats = window_stats.reset_index()
                
                # Create trend plot
                plt.figure(figsize=(12, 8))
                
                for i, metric in enumerate(available_metrics):
                    mean_col = f'{metric}_mean'
                    std_col = f'{metric}_std'
                    
                    if mean_col in window_stats.columns:
                        label = 'Combined Mean IoU' if 'combined' in metric else 'Individual Mean IoU'
                        plt.plot(window_stats['window_numeric'], window_stats[mean_col],
                                marker='o', linewidth=3, markersize=8,
                                color=colors[i], label=label, alpha=0.8)
                        
                        # Add error bars
                        if std_col in window_stats.columns:
                            plt.fill_between(window_stats['window_numeric'],
                                           window_stats[mean_col] - window_stats[std_col],
                                           window_stats[mean_col] + window_stats[std_col],
                                           alpha=0.2, color=colors[i])
                
                plt.xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
                plt.ylabel('Mean IoU (%)', fontsize=12, fontweight='bold')
                plt.title(f'{method_name} - Mean IoU Percentage Trends', fontsize=14, fontweight='bold', pad=20)
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=11)
        
        plt.tight_layout()
        filename = f"{method.lower()}_trends.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved: {filename}")

def create_top10_csv_files(df, output_dir):
    """Create CSV files with top 10 configurations for each method"""
    
    methods = [('word_iou_eval', 'Word-IoU Evaluation'), ('iou_eval', 'IoU Evaluation')]
    
    for method, method_name in methods:
        method_data = df[df['method'] == method].copy()
        if method_data.empty:
            continue
        
        print(f"📋 Creating top 10 CSV files for {method_name}")
        
        if method == 'word_iou_eval':
            # Create top 10 for F1 scores at different IoU thresholds
            thresholds = ['0.3', '0.5', '0.9']
            
            for threshold in thresholds:
                col_name = f'iou_f1_{threshold}'
                if col_name not in method_data.columns:
                    continue
                
                # Sort by F1 score and get top 10
                top_configs = method_data.nlargest(10, col_name)
                
                # Select relevant columns for the CSV
                csv_columns = [
                    'eval_type', 'window', 'stride', 'window_numeric', 'stride_numeric',
                    col_name
                ]
                
                # Add precision and recall if available
                precision_col = f'precision_{threshold}'
                recall_col = f'recall_{threshold}'
                
                if precision_col in method_data.columns:
                    csv_columns.append(precision_col)
                if recall_col in method_data.columns:
                    csv_columns.append(recall_col)
                
                # Create the CSV
                top_configs_csv = top_configs[csv_columns].copy()
                top_configs_csv = top_configs_csv.round(4)
                
                # Add rank column
                top_configs_csv.insert(0, 'rank', range(1, len(top_configs_csv) + 1))
                
                # Save CSV
                filename = f"top10_{method.lower()}_f1_iou_{threshold}.csv"
                top_configs_csv.to_csv(output_dir / filename, index=False)
                
                print(f"💾 Saved: {filename}")
                print(f"   Top F1 (IoU≥{threshold}): {top_configs_csv[col_name].iloc[0]:.4f} "
                      f"({top_configs_csv['window'].iloc[0]}, {top_configs_csv['stride'].iloc[0]})")
        
        elif method == 'iou_eval':
            # Create top 10 for Mean IoU percentages
            iou_metrics = [
                ('combined_mean_iou_percent', 'Combined Mean IoU'),
                ('individual_mean_iou_percent', 'Individual Mean IoU')
            ]
            
            for col_name, metric_name in iou_metrics:
                if col_name not in method_data.columns:
                    continue
                
                # Sort by IoU percentage and get top 10
                top_configs = method_data.nlargest(10, col_name)
                
                # Select relevant columns for the CSV
                csv_columns = [
                    'eval_type', 'window', 'stride', 'window_numeric', 'stride_numeric',
                    col_name
                ]
                
                # Add F1 scores if available
                for threshold in ['0.3', '0.5', '0.9']:
                    f1_col = f'iou_f1_{threshold}'
                    if f1_col in method_data.columns:
                        csv_columns.append(f1_col)
                
                # Create the CSV
                top_configs_csv = top_configs[csv_columns].copy()
                top_configs_csv = top_configs_csv.round(4)
                
                # Add rank column
                top_configs_csv.insert(0, 'rank', range(1, len(top_configs_csv) + 1))
                
                # Save CSV
                filename = f"top10_{method.lower()}_{col_name}.csv"
                top_configs_csv.to_csv(output_dir / filename, index=False)
                
                print(f"💾 Saved: {filename}")
                print(f"   Top {metric_name}: {top_configs_csv[col_name].iloc[0]:.2f}% "
                      f"({top_configs_csv['window'].iloc[0]}, {top_configs_csv['stride'].iloc[0]})")

def create_combined_comparison(df, output_dir):
    """Create combined comparison showing both methods"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    methods = [('word_iou_eval', 'Word-IoU Evaluation'), ('iou_eval', 'IoU Evaluation')]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Word-IoU Evaluation: F1 scores at IoU thresholds
    method_data = df[df['method'] == 'word_iou_eval'].copy()
    if not method_data.empty:
        thresholds = ['0.3', '0.5', '0.9']
        
        # Group by window size
        agg_dict = {f'iou_f1_{t}': ['mean', 'std'] for t in thresholds}
        window_stats = method_data.groupby('window_numeric').agg(agg_dict).round(3)
        
        window_stats.columns = ['_'.join(col).strip() for col in window_stats.columns]
        window_stats = window_stats.reset_index()
        
        ax = axes[0, 0]
        for i, threshold in enumerate(thresholds):
            mean_col = f'iou_f1_{threshold}_mean'
            std_col = f'iou_f1_{threshold}_std'
            
            if mean_col in window_stats.columns:
                ax.plot(window_stats['window_numeric'], window_stats[mean_col],
                       marker='o', linewidth=3, markersize=6,
                       color=colors[i], label=f'IoU ≥ {threshold}', alpha=0.8)
                
                if std_col in window_stats.columns:
                    ax.fill_between(window_stats['window_numeric'],
                                   window_stats[mean_col] - window_stats[std_col],
                                   window_stats[mean_col] + window_stats[std_col],
                                   alpha=0.2, color=colors[i])
        
        ax.set_xlabel('Window Size (seconds)', fontsize=10, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=10, fontweight='bold')
        ax.set_title('Word-IoU Evaluation - F1 Scores', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    # IoU Evaluation: Mean IoU percentages
    method_data = df[df['method'] == 'iou_eval'].copy()
    if not method_data.empty:
        iou_metrics = ['combined_mean_iou_percent', 'individual_mean_iou_percent']
        available_metrics = [col for col in iou_metrics if col in method_data.columns]
        
        if available_metrics:
            # Group by window size
            agg_dict = {metric: ['mean', 'std'] for metric in available_metrics}
            window_stats = method_data.groupby('window_numeric').agg(agg_dict).round(2)
            
            window_stats.columns = ['_'.join(col).strip() for col in window_stats.columns]
            window_stats = window_stats.reset_index()
            
            ax = axes[0, 1]
            for i, metric in enumerate(available_metrics):
                mean_col = f'{metric}_mean'
                std_col = f'{metric}_std'
                
                if mean_col in window_stats.columns:
                    label = 'Combined' if 'combined' in metric else 'Individual'
                    ax.plot(window_stats['window_numeric'], window_stats[mean_col],
                           marker='s', linewidth=3, markersize=6,
                           color=colors[i], label=f'{label} Mean IoU', alpha=0.8)
                    
                    if std_col in window_stats.columns:
                        ax.fill_between(window_stats['window_numeric'],
                                       window_stats[mean_col] - window_stats[std_col],
                                       window_stats[mean_col] + window_stats[std_col],
                                       alpha=0.2, color=colors[i])
            
            ax.set_xlabel('Window Size (seconds)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Mean IoU (%)', fontsize=10, fontweight='bold')
            ax.set_title('IoU Evaluation - Mean IoU Percentages', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
    
    # Remove unused subplots
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "combined_iou_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved: combined_iou_performance_comparison.png")

def main():
    """Main function to run IoU percentage analysis"""
    
    print("=== CREATING IoU PERCENTAGE ANALYSIS ===\n")
    
    # Extract data
    df = extract_iou_percent_data()
    
    if df.empty:
        print("❌ No IoU percentage data extracted!")
        return
    
    print(f"✅ Extracted data from {len(df)} configurations")
    
    # Create output directory
    output_dir = Path("iou_percentage_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Create analysis components
    create_iou_percent_heatmaps(df, output_dir)
    create_iou_percent_trend_graphs(df, output_dir)
    create_top10_csv_files(df, output_dir)
    create_combined_comparison(df, output_dir)
    
    # Create summary statistics
    create_summary_statistics(df, output_dir)
    
    print(f"\n🎉 IoU percentage analysis completed successfully!")
    print(f"📁 Output directory: {output_dir}")
    
    # Print quick summary
    print("\n📊 QUICK SUMMARY:")
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        if method == 'word_iou_eval':
            for threshold in ['0.3', '0.5', '0.9']:
                col_name = f'iou_f1_{threshold}'
                if col_name in method_data.columns:
                    best_config = method_data.loc[method_data[col_name].idxmax()]
                    print(f"   {method} F1 (IoU≥{threshold}): {best_config[col_name]:.4f} "
                          f"({best_config['window']}, {best_config['stride']})")
        
        elif method == 'iou_eval':
            # Show Mean IoU percentages
            for col_name, label in [('combined_mean_iou_percent', 'Combined Mean IoU'), 
                                   ('individual_mean_iou_percent', 'Individual Mean IoU')]:
                if col_name in method_data.columns:
                    best_config = method_data.loc[method_data[col_name].idxmax()]
                    print(f"   {method} {label}: {best_config[col_name]:.2f}% "
                          f"({best_config['window']}, {best_config['stride']})")

def create_summary_statistics(df, output_dir):
    """Create summary statistics CSV"""
    
    summary_stats = []
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        if method == 'word_iou_eval':
            # Analyze F1 scores at different IoU thresholds
            for threshold in ['0.3', '0.5', '0.9']:
                col_name = f'iou_f1_{threshold}'
                if col_name not in method_data.columns:
                    continue
                
                stats = method_data[col_name].describe()
                best_config = method_data.loc[method_data[col_name].idxmax()]
                
                summary_stats.append({
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
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.round(4)
    summary_df.to_csv(output_dir / "iou_performance_summary_statistics.csv", index=False)
    
    print(f"💾 Saved: iou_performance_summary_statistics.csv")

if __name__ == "__main__":
    main()

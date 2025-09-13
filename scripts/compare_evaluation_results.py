#!/usr/bin/env python3
"""
Comparison and analysis script for structured evaluation results.
This script analyzes all evaluation results in the 4_classes_auto folder
and generates comprehensive comparison reports and visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def setup_thai_font():
    """Setup Thai font for matplotlib"""
    try:
        import matplotlib.font_manager as fm
        font_path = './fonts/THSarabunNew.ttf'
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'Cordia New'
    except:
        pass

def parse_directory_structure(base_dir="./evaluation_results/4_classes_auto"):
    """
    Parse the directory structure to extract evaluation results.
    
    Returns:
        dict: Nested dictionary with results organized by window and stride
    """
    results = {}
    
    if not os.path.exists(base_dir):
        print(f"❌ Base directory not found: {base_dir}")
        return results
    
    # Walk through window directories
    for window_dir in os.listdir(base_dir):
        window_path = os.path.join(base_dir, window_dir)
        
        if not os.path.isdir(window_path) or not window_dir.startswith('window_'):
            continue
        
        window_size = window_dir
        results[window_size] = {}
        
        # Walk through stride directories
        for stride_dir in os.listdir(window_path):
            stride_path = os.path.join(window_path, stride_dir)
            
            if not os.path.isdir(stride_path) or not stride_dir.startswith('stride_'):
                continue
            
            stride_percent = stride_dir
            
            # Look for evaluation report
            report_path = os.path.join(stride_path, 'logs', 'evaluation_report.txt')
            csv_path = os.path.join(stride_path, 'csv', 'eval_results_detailed.csv')
            
            if os.path.exists(report_path):
                try:
                    metrics = parse_evaluation_report(report_path)
                    results[window_size][stride_percent] = {
                        'metrics': metrics,
                        'report_path': report_path,
                        'csv_path': csv_path if os.path.exists(csv_path) else None,
                        'plots_dir': os.path.join(stride_path, 'plots')
                    }
                except Exception as e:
                    print(f"⚠️  Error parsing {report_path}: {e}")
    
    return results

def parse_evaluation_report(report_path):
    """
    Parse evaluation report file to extract metrics.
    
    Returns:
        dict: Extracted metrics
    """
    metrics = {}
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # Try to convert to float if possible
            try:
                if key in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 
                          'Word Precision', 'Word Recall', 'Word F1-Score']:
                    metrics[key.lower().replace(' ', '_').replace('-', '_')] = float(value)
                elif key in ['Total Windows', 'True Positives', 'False Positives', 
                           'True Negatives', 'False Negatives']:
                    metrics[key.lower().replace(' ', '_')] = int(value)
                elif key in ['Window Size', 'Stride']:
                    metrics[key.lower().replace(' ', '_')] = value
            except ValueError:
                pass
    
    return metrics

def create_comparison_dataframe(results):
    """
    Create a DataFrame suitable for comparison analysis.
    
    Returns:
        pd.DataFrame: Comparison data
    """
    rows = []
    
    for window_size, window_data in results.items():
        for stride_percent, config_data in window_data.items():
            metrics = config_data['metrics']
            
            row = {
                'window_size': window_size,
                'stride_percent': stride_percent,
                'window_seconds': float(window_size.replace('window_', '').replace('s', '')),
                'stride_numeric': float(stride_percent.replace('stride_', '').replace('%', '')),
            }
            
            # Add all metrics
            row.update(metrics)
            rows.append(row)
    
    return pd.DataFrame(rows)

def create_heatmap_comparison(df, metric='f1_score', output_dir='./evaluation_results/4_classes_auto'):
    """Create heatmap comparing metric across window sizes and stride percentages"""
    setup_thai_font()
    
    # Create pivot table
    pivot_table = df.pivot(index='stride_numeric', columns='window_seconds', values=metric)
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                cbar_kws={'label': metric.replace('_', ' ').title()})
    
    plt.title(f'{metric.replace("_", " ").title()} Comparison Across Configurations', 
              fontsize=16, pad=20)
    plt.xlabel('Window Size (seconds)', fontsize=12)
    plt.ylabel('Stride Percentage (%)', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(f"{output_dir}/comparison_plots", exist_ok=True)
    plot_path = f"{output_dir}/comparison_plots/heatmap_{metric}.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✅ Heatmap saved to {plot_path}")
    
    return pivot_table

def create_line_plots(df, output_dir='./evaluation_results/4_classes_auto'):
    """Create line plots showing trends"""
    setup_thai_font()
    
    metrics = ['f1_score', 'accuracy', 'precision', 'recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot lines for each window size
        window_sizes = sorted(df['window_seconds'].unique())
        
        for window_size in window_sizes:
            window_data = df[df['window_seconds'] == window_size]
            if len(window_data) > 0:
                ax.plot(window_data['stride_numeric'], window_data[metric], 
                       marker='o', label=f'{window_size}s', linewidth=2)
        
        ax.set_xlabel('Stride Percentage (%)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs Stride Percentage')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"{output_dir}/comparison_plots/line_plots_trends.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✅ Line plots saved to {plot_path}")

def find_best_configurations(df, top_n=5):
    """Find the best configurations for different metrics"""
    
    metrics = ['f1_score', 'accuracy', 'precision', 'recall', 'word_f1_score']
    best_configs = {}
    
    for metric in metrics:
        if metric in df.columns:
            top_configs = df.nlargest(top_n, metric)[['window_size', 'stride_percent', metric]]
            best_configs[metric] = top_configs
    
    return best_configs

def create_comprehensive_report(df, best_configs, output_dir='./evaluation_results/4_classes_auto'):
    """Create a comprehensive comparison report"""
    
    report_path = f"{output_dir}/comprehensive_comparison_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE EVALUATION COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Configurations Evaluated: {len(df)}\n")
        f.write(f"Window Sizes: {sorted(df['window_seconds'].unique())}\n")
        f.write(f"Stride Percentages: {sorted(df['stride_numeric'].unique())}\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 40 + "\n")
        for metric in ['f1_score', 'accuracy', 'precision', 'recall']:
            if metric in df.columns:
                f.write(f"{metric.replace('_', ' ').title()}:\n")
                f.write(f"  Mean: {df[metric].mean():.4f}\n")
                f.write(f"  Std:  {df[metric].std():.4f}\n")
                f.write(f"  Min:  {df[metric].min():.4f}\n")
                f.write(f"  Max:  {df[metric].max():.4f}\n\n")
        
        # Best configurations
        f.write("BEST CONFIGURATIONS:\n")
        f.write("-" * 40 + "\n")
        for metric, configs in best_configs.items():
            f.write(f"\nTop 5 {metric.replace('_', ' ').title()}:\n")
            for i, (_, row) in enumerate(configs.iterrows(), 1):
                f.write(f"  {i}. {row['window_size']} / {row['stride_percent']}: {row[metric]:.4f}\n")
        
        # Window size analysis
        f.write(f"\nWINDOW SIZE ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        window_analysis = df.groupby('window_seconds')['f1_score'].agg(['mean', 'std', 'count'])
        for window_size, stats in window_analysis.iterrows():
            f.write(f"Window {window_size}s: Mean F1={stats['mean']:.4f}, Std={stats['std']:.4f}, Count={stats['count']}\n")
        
        # Stride analysis
        f.write(f"\nSTRIDE PERCENTAGE ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        stride_analysis = df.groupby('stride_numeric')['f1_score'].agg(['mean', 'std', 'count'])
        for stride_percent, stats in stride_analysis.iterrows():
            f.write(f"Stride {stride_percent}%: Mean F1={stats['mean']:.4f}, Std={stats['std']:.4f}, Count={stats['count']}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"📋 Comprehensive report saved to {report_path}")

def create_detailed_csv_export(df, output_dir='./evaluation_results/4_classes_auto'):
    """Export detailed results as CSV for further analysis"""
    
    csv_path = f"{output_dir}/detailed_comparison_results.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"📊 Detailed CSV exported to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare and analyze structured evaluation results')
    parser.add_argument('--base_dir', type=str, default='./evaluation_results/4_classes_auto',
                       help='Base directory containing evaluation results')
    parser.add_argument('--top_n', type=int, default=5,
                       help='Number of top configurations to show')
    
    args = parser.parse_args()
    
    print("📊 Starting comprehensive comparison analysis...")
    print(f"   Base directory: {args.base_dir}")
    
    # Parse all results
    results = parse_directory_structure(args.base_dir)
    
    if not results:
        print("❌ No evaluation results found!")
        return
    
    total_configs = sum(len(window_data) for window_data in results.values())
    print(f"📁 Found results for {total_configs} configurations across {len(results)} window sizes")
    
    # Create comparison DataFrame
    df = create_comparison_dataframe(results)
    
    if df.empty:
        print("❌ No valid metrics found in results!")
        return
    
    print(f"✅ Loaded {len(df)} configuration results")
    
    # Create output directory for comparison plots
    os.makedirs(f"{args.base_dir}/comparison_plots", exist_ok=True)
    
    # Generate visualizations
    print("🎨 Generating comparison visualizations...")
    
    # Heatmaps for different metrics
    metrics_to_plot = ['f1_score', 'accuracy', 'precision', 'recall']
    for metric in metrics_to_plot:
        if metric in df.columns:
            create_heatmap_comparison(df, metric, args.base_dir)
    
    # Line plots
    create_line_plots(df, args.base_dir)
    
    # Find best configurations
    print("🏆 Finding best configurations...")
    best_configs = find_best_configurations(df, args.top_n)
    
    # Create comprehensive report
    print("📋 Creating comprehensive report...")
    create_comprehensive_report(df, best_configs, args.base_dir)
    
    # Export detailed CSV
    create_detailed_csv_export(df, args.base_dir)
    
    # Print summary to console
    print(f"\n{'='*60}")
    print(f"COMPARISON ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Configurations analyzed: {len(df)}")
    print(f"Best F1-Score: {df['f1_score'].max():.4f}")
    print(f"Best configuration: {df.loc[df['f1_score'].idxmax(), 'window_size']} / {df.loc[df['f1_score'].idxmax(), 'stride_percent']}")
    print(f"Mean F1-Score: {df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f}")
    print(f"\n📁 All outputs saved to: {args.base_dir}/")
    print(f"   • Comparison plots: {args.base_dir}/comparison_plots/")
    print(f"   • Comprehensive report: {args.base_dir}/comprehensive_comparison_report.txt")
    print(f"   • Detailed CSV: {args.base_dir}/detailed_comparison_results.csv")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive Analysis of Window Size, Stride, and F1 Score Correlations
Analyzes both eval_by_0.05 and eval_percent datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def parse_window_stride(df):
    """Parse window and stride values to numeric format"""
    df_parsed = df.copy()
    
    # Parse window values (e.g., "window_0.3s" -> 0.3)
    df_parsed['window_numeric'] = df_parsed['window'].str.extract(r'window_(\d+\.?\d*)s').astype(float)
    
    # Parse stride values differently for eval_by_0.05 and eval_percent
    stride_numeric = []
    stride_percentage = []
    
    for idx, row in df_parsed.iterrows():
        stride_val = row['stride']
        if row['eval_type'] == 'eval_by_0.05':
            # For eval_by_0.05: stride_0.125s -> 0.125
            numeric_val = float(re.search(r'stride_(\d+\.?\d*)', stride_val).group(1))
            stride_numeric.append(numeric_val)
            # Calculate percentage relative to window
            percentage = (numeric_val / row['window_numeric']) * 100
            stride_percentage.append(percentage)
        else:
            # For eval_percent: stride_30.0% -> 30.0
            if '%' in stride_val:
                percentage = float(re.search(r'stride_(\d+\.?\d*)%', stride_val).group(1))
                stride_percentage.append(percentage)
                # Calculate absolute value
                numeric_val = (percentage / 100) * row['window_numeric']
                stride_numeric.append(numeric_val)
            else:
                # Handle special cases like stride_100.0%
                percentage = float(re.search(r'stride_(\d+\.?\d*)', stride_val).group(1))
                stride_percentage.append(percentage)
                numeric_val = (percentage / 100) * row['window_numeric']
                stride_numeric.append(numeric_val)
    
    df_parsed['stride_numeric'] = stride_numeric
    df_parsed['stride_percentage'] = stride_percentage
    
    return df_parsed

def create_correlation_analysis(df):
    """Create comprehensive correlation analysis"""
    
    print("=== COMPREHENSIVE F1 SCORE CORRELATION ANALYSIS ===\n")
    
    # Basic statistics
    print("📊 DATASET OVERVIEW:")
    print(f"Total configurations: {len(df)}")
    print(f"eval_by_0.05 configurations: {len(df[df['eval_type'] == 'eval_by_0.05'])}")
    print(f"eval_percent configurations: {len(df[df['eval_type'] == 'eval_percent'])}")
    print(f"Successful evaluations: {len(df[df['success'] == True])}")
    print()
    
    # Filter successful evaluations
    df_success = df[df['success'] == True].copy()
    
    # F1 score statistics
    print("🎯 F1 SCORE STATISTICS:")
    for f1_type in ['window_f1', 'word_f1']:
        print(f"\n{f1_type.upper().replace('_', ' ')}:")
        print(f"  Mean: {df_success[f1_type].mean():.4f}")
        print(f"  Std:  {df_success[f1_type].std():.4f}")
        print(f"  Min:  {df_success[f1_type].min():.4f}")
        print(f"  Max:  {df_success[f1_type].max():.4f}")
    
    print(f"\nCOMBINED IoU:")
    print(f"  Mean: {df_success['combined_iou'].mean():.4f}")
    print(f"  Std:  {df_success['combined_iou'].std():.4f}")
    print(f"  Min:  {df_success['combined_iou'].min():.4f}")
    print(f"  Max:  {df_success['combined_iou'].max():.4f}")
    print()
    
    return df_success

def find_best_configurations(df_success):
    """Find best performing configurations"""
    
    print("🏆 TOP PERFORMING CONFIGURATIONS:\n")
    
    # Top configurations for each metric
    metrics = {
        'window_f1': 'Window-level F1',
        'word_f1': 'Word-level F1',
        'combined_iou': 'Combined IoU'
    }
    
    top_configs = {}
    
    for metric, metric_name in metrics.items():
        print(f"🥇 TOP 5 {metric_name}:")
        top_5 = df_success.nlargest(5, metric)[['eval_type', 'window', 'stride', 'window_numeric', 
                                               'stride_percentage', metric]].round(4)
        
        for idx, row in top_5.iterrows():
            print(f"  {row[metric]:.4f} - {row['eval_type']} | Window: {row['window_numeric']}s | "
                  f"Stride: {row['stride_percentage']:.1f}% | {row['window']}/{row['stride']}")
        
        top_configs[metric] = top_5
        print()
    
    return top_configs

def analyze_correlations(df_success):
    """Analyze correlations between window, stride, and F1 scores"""
    
    print("🔍 CORRELATION ANALYSIS:\n")
    
    # Calculate correlations
    numeric_cols = ['window_numeric', 'stride_numeric', 'stride_percentage', 
                   'window_f1', 'word_f1', 'combined_iou']
    
    correlation_matrix = df_success[numeric_cols].corr()
    
    print("📈 CORRELATION COEFFICIENTS:")
    print("(Window Size vs F1 Scores)")
    print(f"  Window vs Window F1: {correlation_matrix.loc['window_numeric', 'window_f1']:.4f}")
    print(f"  Window vs Word F1:   {correlation_matrix.loc['window_numeric', 'word_f1']:.4f}")
    print(f"  Window vs IoU:       {correlation_matrix.loc['window_numeric', 'combined_iou']:.4f}")
    
    print("\n(Stride Percentage vs F1 Scores)")
    print(f"  Stride% vs Window F1: {correlation_matrix.loc['stride_percentage', 'window_f1']:.4f}")
    print(f"  Stride% vs Word F1:   {correlation_matrix.loc['stride_percentage', 'word_f1']:.4f}")
    print(f"  Stride% vs IoU:       {correlation_matrix.loc['stride_percentage', 'combined_iou']:.4f}")
    
    print("\n(Stride Absolute vs F1 Scores)")
    print(f"  Stride vs Window F1:  {correlation_matrix.loc['stride_numeric', 'window_f1']:.4f}")
    print(f"  Stride vs Word F1:    {correlation_matrix.loc['stride_numeric', 'word_f1']:.4f}")
    print(f"  Stride vs IoU:        {correlation_matrix.loc['stride_numeric', 'combined_iou']:.4f}")
    print()
    
    return correlation_matrix

def compare_evaluation_types(df_success):
    """Compare performance between eval_by_0.05 and eval_percent"""
    
    print("⚖️  EVALUATION TYPE COMPARISON:\n")
    
    eval_05 = df_success[df_success['eval_type'] == 'eval_by_0.05']
    eval_percent = df_success[df_success['eval_type'] == 'eval_percent']
    
    metrics = ['window_f1', 'word_f1', 'combined_iou']
    
    for metric in metrics:
        print(f"{metric.upper().replace('_', ' ')}:")
        print(f"  eval_by_0.05:  Mean={eval_05[metric].mean():.4f}, Std={eval_05[metric].std():.4f}")
        print(f"  eval_percent:  Mean={eval_percent[metric].mean():.4f}, Std={eval_percent[metric].std():.4f}")
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(eval_05[metric], eval_percent[metric])
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"  Difference: t={t_stat:.3f}, p={p_value:.6f} {significance}")
        print()

def analyze_window_stride_patterns(df_success):
    """Analyze patterns in window and stride combinations"""
    
    print("🔬 WINDOW-STRIDE PATTERN ANALYSIS:\n")
    
    # Group by window size and analyze
    window_analysis = df_success.groupby('window_numeric').agg({
        'window_f1': ['mean', 'std', 'max'],
        'word_f1': ['mean', 'std', 'max'],
        'combined_iou': ['mean', 'std', 'max'],
        'stride_percentage': ['mean', 'min', 'max']
    }).round(4)
    
    print("📏 PERFORMANCE BY WINDOW SIZE:")
    print("Window | Window F1 (mean±std, max) | Word F1 (mean±std, max) | IoU (mean±std, max) | Stride% Range")
    print("-" * 100)
    
    for window in sorted(df_success['window_numeric'].unique()):
        w_data = window_analysis.loc[window]
        print(f"{window:4.1f}s  | {w_data[('window_f1', 'mean')]:.3f}±{w_data[('window_f1', 'std')]:.3f}, {w_data[('window_f1', 'max')]:.3f} | "
              f"{w_data[('word_f1', 'mean')]:.3f}±{w_data[('word_f1', 'std')]:.3f}, {w_data[('word_f1', 'max')]:.3f} | "
              f"{w_data[('combined_iou', 'mean')]:.3f}±{w_data[('combined_iou', 'std')]:.3f}, {w_data[('combined_iou', 'max')]:.3f} | "
              f"{w_data[('stride_percentage', 'min')]:.0f}-{w_data[('stride_percentage', 'max')]:.0f}%")
    
    print()
    
    # Optimal stride percentage analysis
    print("🎯 OPTIMAL STRIDE PERCENTAGES:")
    
    # Find best stride percentage for each window size
    best_stride_by_window = df_success.loc[df_success.groupby('window_numeric')['window_f1'].idxmax()]
    
    print("Window | Best Window F1 | Optimal Stride% | Word F1 | IoU")
    print("-" * 65)
    for _, row in best_stride_by_window.sort_values('window_numeric').iterrows():
        print(f"{row['window_numeric']:4.1f}s  | {row['window_f1']:10.4f} | {row['stride_percentage']:10.1f}% | {row['word_f1']:6.4f} | {row['combined_iou']:6.4f}")
    
    print()

def create_visualizations(df_success, output_dir):
    """Create correlation visualizations"""
    
    print("📊 CREATING VISUALIZATIONS...\n")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    viz_dir = Path(output_dir) / "correlation_analysis_plots"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Correlation heatmap
    plt.figure(figsize=(12, 8))
    numeric_cols = ['window_numeric', 'stride_numeric', 'stride_percentage', 
                   'window_f1', 'word_f1', 'combined_iou']
    correlation_matrix = df_success[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', cbar_kws={"shrink": .8}, mask=mask)
    plt.title('Correlation Matrix: Window Size, Stride, and Performance Metrics')
    plt.tight_layout()
    plt.savefig(viz_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Window size vs F1 scores
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Window F1 by evaluation type
    for i, eval_type in enumerate(['eval_by_0.05', 'eval_percent']):
        data = df_success[df_success['eval_type'] == eval_type]
        axes[0, i].scatter(data['window_numeric'], data['window_f1'], alpha=0.6, s=30)
        axes[0, i].set_xlabel('Window Size (seconds)')
        axes[0, i].set_ylabel('Window F1 Score')
        axes[0, i].set_title(f'Window F1 vs Window Size - {eval_type}')
        axes[0, i].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(data['window_numeric'], data['window_f1'], 1)
        p = np.poly1d(z)
        axes[0, i].plot(data['window_numeric'], p(data['window_numeric']), "r--", alpha=0.8)
    
    # Word F1 by evaluation type
    for i, eval_type in enumerate(['eval_by_0.05', 'eval_percent']):
        data = df_success[df_success['eval_type'] == eval_type]
        axes[1, i].scatter(data['window_numeric'], data['word_f1'], alpha=0.6, s=30, color='orange')
        axes[1, i].set_xlabel('Window Size (seconds)')
        axes[1, i].set_ylabel('Word F1 Score')
        axes[1, i].set_title(f'Word F1 vs Window Size - {eval_type}')
        axes[1, i].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(data['window_numeric'], data['word_f1'], 1)
        p = np.poly1d(z)
        axes[1, i].plot(data['window_numeric'], p(data['window_numeric']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(viz_dir / "window_vs_f1_scores.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Stride percentage vs F1 scores
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Window F1 vs Stride percentage
    for i, eval_type in enumerate(['eval_by_0.05', 'eval_percent']):
        data = df_success[df_success['eval_type'] == eval_type]
        axes[0, i].scatter(data['stride_percentage'], data['window_f1'], alpha=0.6, s=30)
        axes[0, i].set_xlabel('Stride Percentage (%)')
        axes[0, i].set_ylabel('Window F1 Score')
        axes[0, i].set_title(f'Window F1 vs Stride % - {eval_type}')
        axes[0, i].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(data['stride_percentage'], data['window_f1'], 1)
        p = np.poly1d(z)
        axes[0, i].plot(data['stride_percentage'], p(data['stride_percentage']), "r--", alpha=0.8)
    
    # Word F1 vs Stride percentage
    for i, eval_type in enumerate(['eval_by_0.05', 'eval_percent']):
        data = df_success[df_success['eval_type'] == eval_type]
        axes[1, i].scatter(data['stride_percentage'], data['word_f1'], alpha=0.6, s=30, color='orange')
        axes[1, i].set_xlabel('Stride Percentage (%)')
        axes[1, i].set_ylabel('Word F1 Score')
        axes[1, i].set_title(f'Word F1 vs Stride % - {eval_type}')
        axes[1, i].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(data['stride_percentage'], data['word_f1'], 1)
        p = np.poly1d(z)
        axes[1, i].plot(data['stride_percentage'], p(data['stride_percentage']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(viz_dir / "stride_vs_f1_scores.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 3D scatter plot for best performing configurations
    fig = plt.figure(figsize=(15, 5))
    
    for i, metric in enumerate(['window_f1', 'word_f1', 'combined_iou']):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Color by evaluation type
        eval_05 = df_success[df_success['eval_type'] == 'eval_by_0.05']
        eval_percent = df_success[df_success['eval_type'] == 'eval_percent']
        
        ax.scatter(eval_05['window_numeric'], eval_05['stride_percentage'], eval_05[metric], 
                  alpha=0.6, s=20, label='eval_by_0.05', c='blue')
        ax.scatter(eval_percent['window_numeric'], eval_percent['stride_percentage'], eval_percent[metric], 
                  alpha=0.6, s=20, label='eval_percent', c='red')
        
        ax.set_xlabel('Window Size (s)')
        ax.set_ylabel('Stride Percentage (%)')
        ax.set_zlabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(viz_dir / "3d_performance_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to: {viz_dir}")
    return viz_dir

def save_detailed_analysis(df_success, top_configs, correlation_matrix, output_dir):
    """Save detailed analysis to CSV files"""
    
    analysis_dir = Path(output_dir) / "detailed_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Save complete analyzed dataset
    df_success.to_csv(analysis_dir / "complete_analysis_data.csv", index=False)
    
    # Save top configurations
    for metric, top_df in top_configs.items():
        top_df.to_csv(analysis_dir / f"top_5_{metric}_configurations.csv", index=False)
    
    # Save correlation matrix
    correlation_matrix.to_csv(analysis_dir / "correlation_matrix.csv")
    
    # Save window-stride summary
    window_stride_summary = df_success.groupby(['window_numeric', 'stride_percentage']).agg({
        'window_f1': ['mean', 'std', 'count'],
        'word_f1': ['mean', 'std', 'count'],
        'combined_iou': ['mean', 'std', 'count'],
        'eval_type': lambda x: list(x.unique())
    }).round(4)
    
    window_stride_summary.to_csv(analysis_dir / "window_stride_performance_summary.csv")
    
    # Best configuration per window size
    best_per_window = df_success.loc[df_success.groupby('window_numeric')['window_f1'].idxmax()]
    best_per_window[['window_numeric', 'stride_percentage', 'window_f1', 'word_f1', 
                    'combined_iou', 'eval_type', 'window', 'stride']].to_csv(
        analysis_dir / "best_configuration_per_window.csv", index=False)
    
    print(f"📁 Detailed analysis saved to: {analysis_dir}")
    return analysis_dir

def main():
    # Load the summary data
    summary_path = Path("fixed_smart_parallel_results/fixed_smart_parallel_summary.csv")
    
    if not summary_path.exists():
        print(f"❌ Summary file not found: {summary_path}")
        return
    
    print(f"📊 Loading data from: {summary_path}")
    df = pd.read_csv(summary_path)
    
    # Parse window and stride values
    df_parsed = parse_window_stride(df)
    
    # Create correlation analysis
    df_success = create_correlation_analysis(df_parsed)
    
    # Find best configurations
    top_configs = find_best_configurations(df_success)
    
    # Analyze correlations
    correlation_matrix = analyze_correlations(df_success)
    
    # Compare evaluation types
    compare_evaluation_types(df_success)
    
    # Analyze window-stride patterns
    analyze_window_stride_patterns(df_success)
    
    # Create visualizations
    viz_dir = create_visualizations(df_success, "fixed_smart_parallel_results")
    
    # Save detailed analysis
    analysis_dir = save_detailed_analysis(df_success, top_configs, correlation_matrix, 
                                        "fixed_smart_parallel_results")
    
    print("\n🎉 ANALYSIS COMPLETE!")
    print(f"📊 Visualizations: {viz_dir}")
    print(f"📁 Detailed data: {analysis_dir}")
    
    # Summary insights
    print("\n💡 KEY INSIGHTS:")
    
    # Best overall configuration
    best_window_f1 = df_success.loc[df_success['window_f1'].idxmax()]
    best_word_f1 = df_success.loc[df_success['word_f1'].idxmax()]
    
    print(f"🏆 Best Window F1: {best_window_f1['window_f1']:.4f}")
    print(f"   Configuration: {best_window_f1['eval_type']} | Window: {best_window_f1['window_numeric']}s | Stride: {best_window_f1['stride_percentage']:.1f}%")
    
    print(f"🏆 Best Word F1: {best_word_f1['word_f1']:.4f}")
    print(f"   Configuration: {best_word_f1['eval_type']} | Window: {best_word_f1['window_numeric']}s | Stride: {best_word_f1['stride_percentage']:.1f}%")
    
    # Correlation insights
    window_f1_corr = correlation_matrix.loc['window_numeric', 'window_f1']
    stride_f1_corr = correlation_matrix.loc['stride_percentage', 'window_f1']
    
    print(f"\n📈 Window size correlation with Window F1: {window_f1_corr:.4f}")
    if window_f1_corr > 0.3:
        print("   → Larger windows generally improve Window F1 performance")
    elif window_f1_corr < -0.3:
        print("   → Smaller windows generally improve Window F1 performance")
    else:
        print("   → Window size has weak correlation with Window F1 performance")
    
    print(f"📈 Stride percentage correlation with Window F1: {stride_f1_corr:.4f}")
    if stride_f1_corr > 0.3:
        print("   → Larger stride percentages generally improve Window F1 performance")
    elif stride_f1_corr < -0.3:
        print("   → Smaller stride percentages generally improve Window F1 performance")
    else:
        print("   → Stride percentage has weak correlation with Window F1 performance")

if __name__ == "__main__":
    main()

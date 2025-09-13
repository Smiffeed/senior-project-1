#!/usr/bin/env python3
"""
Simple Correlation Analysis - No visualization dependencies required
Analyzes correlations between window size, stride, and F1 scores
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from scipy import stats

def parse_window_stride(df):
    """Parse window and stride values to numeric format"""
    df_parsed = df.copy()
    
    # Parse window values (e.g., "window_0.3s" -> 0.3)
    df_parsed['window_numeric'] = df_parsed['window'].str.extract(r'window_(\d+\.?\d*)s').astype(float)
    
    # Parse stride values
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
                # Handle special cases
                percentage = float(re.search(r'stride_(\d+\.?\d*)', stride_val).group(1))
                stride_percentage.append(percentage)
                numeric_val = (percentage / 100) * row['window_numeric']
                stride_numeric.append(numeric_val)
    
    df_parsed['stride_numeric'] = stride_numeric
    df_parsed['stride_percentage'] = stride_percentage
    
    return df_parsed

def main():
    print("=== WINDOW, STRIDE, AND F1 SCORE CORRELATION ANALYSIS ===\n")
    
    # Load data
    summary_path = Path("fixed_smart_parallel_results/fixed_smart_parallel_summary.csv")
    df = pd.read_csv(summary_path)
    
    # Parse and filter successful evaluations
    df_parsed = parse_window_stride(df)
    df_success = df_parsed[df_parsed['success'] == True].copy()
    
    print(f"📊 Dataset: {len(df_success)} successful evaluations")
    print(f"   eval_by_0.05: {len(df_success[df_success['eval_type'] == 'eval_by_0.05'])}")
    print(f"   eval_percent: {len(df_success[df_success['eval_type'] == 'eval_percent'])}")
    print()
    
    # Basic F1 Score Statistics
    print("🎯 F1 SCORE OVERVIEW:")
    for metric in ['window_f1', 'word_f1', 'combined_iou']:
        values = df_success[metric]
        print(f"  {metric.replace('_', ' ').title():15}: Mean={values.mean():.4f}, Std={values.std():.4f}, Min={values.min():.4f}, Max={values.max():.4f}")
    print()
    
    # TOP PERFORMING CONFIGURATIONS
    print("🏆 TOP 10 CONFIGURATIONS BY WINDOW F1:")
    print("Rank | Window F1 | Word F1 | IoU   | Eval Type    | Window | Stride% | Configuration")
    print("-" * 90)
    
    top_10_window = df_success.nlargest(10, 'window_f1')
    for i, (_, row) in enumerate(top_10_window.iterrows(), 1):
        print(f"{i:4d} | {row['window_f1']:.4f}    | {row['word_f1']:.4f}  | {row['combined_iou']:.4f} | {row['eval_type']:12} | {row['window_numeric']:4.1f}s  | {row['stride_percentage']:5.1f}% | {row['window']}/{row['stride']}")
    
    print()
    print("🏆 TOP 10 CONFIGURATIONS BY WORD F1:")
    print("Rank | Word F1   | Window F1 | IoU   | Eval Type    | Window | Stride% | Configuration")
    print("-" * 90)
    
    top_10_word = df_success.nlargest(10, 'word_f1')
    for i, (_, row) in enumerate(top_10_word.iterrows(), 1):
        print(f"{i:4d} | {row['word_f1']:.4f}    | {row['window_f1']:.4f}  | {row['combined_iou']:.4f} | {row['eval_type']:12} | {row['window_numeric']:4.1f}s  | {row['stride_percentage']:5.1f}% | {row['window']}/{row['stride']}")
    
    print()
    
    # CORRELATION ANALYSIS
    print("🔍 CORRELATION ANALYSIS:")
    
    # Calculate correlations
    numeric_cols = ['window_numeric', 'stride_numeric', 'stride_percentage', 
                   'window_f1', 'word_f1', 'combined_iou']
    correlation_matrix = df_success[numeric_cols].corr()
    
    print("\n📈 KEY CORRELATIONS:")
    print("WINDOW SIZE EFFECTS:")
    print(f"  Window vs Window F1:      {correlation_matrix.loc['window_numeric', 'window_f1']:7.4f}")
    print(f"  Window vs Word F1:        {correlation_matrix.loc['window_numeric', 'word_f1']:7.4f}")
    print(f"  Window vs IoU:            {correlation_matrix.loc['window_numeric', 'combined_iou']:7.4f}")
    
    print("\nSTRIDE PERCENTAGE EFFECTS:")
    print(f"  Stride% vs Window F1:     {correlation_matrix.loc['stride_percentage', 'window_f1']:7.4f}")
    print(f"  Stride% vs Word F1:       {correlation_matrix.loc['stride_percentage', 'word_f1']:7.4f}")
    print(f"  Stride% vs IoU:           {correlation_matrix.loc['stride_percentage', 'combined_iou']:7.4f}")
    
    print("\nSTRIDE ABSOLUTE EFFECTS:")
    print(f"  Stride vs Window F1:      {correlation_matrix.loc['stride_numeric', 'window_f1']:7.4f}")
    print(f"  Stride vs Word F1:        {correlation_matrix.loc['stride_numeric', 'word_f1']:7.4f}")
    print(f"  Stride vs IoU:            {correlation_matrix.loc['stride_numeric', 'combined_iou']:7.4f}")
    print()
    
    # ANALYSIS BY WINDOW SIZE
    print("📏 PERFORMANCE BY WINDOW SIZE:")
    print("Window | Count | Window F1 (Mean±Std, Max) | Word F1 (Mean±Std, Max) | IoU (Mean±Std, Max)")
    print("-" * 85)
    
    for window in sorted(df_success['window_numeric'].unique()):
        window_data = df_success[df_success['window_numeric'] == window]
        wf1_mean, wf1_std, wf1_max = window_data['window_f1'].mean(), window_data['window_f1'].std(), window_data['window_f1'].max()
        word_mean, word_std, word_max = window_data['word_f1'].mean(), window_data['word_f1'].std(), window_data['word_f1'].max()
        iou_mean, iou_std, iou_max = window_data['combined_iou'].mean(), window_data['combined_iou'].std(), window_data['combined_iou'].max()
        
        print(f"{window:4.1f}s  | {len(window_data):5d} | {wf1_mean:.3f}±{wf1_std:.3f}, {wf1_max:.3f}    | {word_mean:.3f}±{word_std:.3f}, {word_max:.3f}    | {iou_mean:.3f}±{iou_std:.3f}, {iou_max:.3f}")
    
    print()
    
    # BEST CONFIGURATION PER WINDOW SIZE
    print("🎯 OPTIMAL CONFIGURATION PER WINDOW SIZE (by Window F1):")
    print("Window | Best Window F1 | Word F1 | IoU   | Optimal Stride% | Eval Type    | Full Configuration")
    print("-" * 95)
    
    best_per_window = df_success.loc[df_success.groupby('window_numeric')['window_f1'].idxmax()]
    for _, row in best_per_window.sort_values('window_numeric').iterrows():
        print(f"{row['window_numeric']:4.1f}s  | {row['window_f1']:10.4f}   | {row['word_f1']:.4f}  | {row['combined_iou']:.4f} | {row['stride_percentage']:10.1f}% | {row['eval_type']:12} | {row['window']}/{row['stride']}")
    
    print()
    
    # EVALUATION TYPE COMPARISON
    print("⚖️  EVALUATION TYPE COMPARISON:")
    eval_05 = df_success[df_success['eval_type'] == 'eval_by_0.05']
    eval_percent = df_success[df_success['eval_type'] == 'eval_percent']
    
    for metric in ['window_f1', 'word_f1', 'combined_iou']:
        print(f"\n{metric.replace('_', ' ').upper()}:")
        print(f"  eval_by_0.05:  Mean={eval_05[metric].mean():.4f}, Std={eval_05[metric].std():.4f}, Max={eval_05[metric].max():.4f}")
        print(f"  eval_percent:  Mean={eval_percent[metric].mean():.4f}, Std={eval_percent[metric].std():.4f}, Max={eval_percent[metric].max():.4f}")
        
        # Statistical significance test
        try:
            t_stat, p_value = stats.ttest_ind(eval_05[metric], eval_percent[metric])
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"  Difference:    t-stat={t_stat:.3f}, p-value={p_value:.6f} ({significance})")
        except:
            print(f"  Difference:    Could not compute statistical test")
    
    print()
    
    # STRIDE PERCENTAGE ANALYSIS
    print("📊 STRIDE PERCENTAGE IMPACT ANALYSIS:")
    
    # Group by stride percentage ranges
    stride_ranges = [(0, 25), (25, 50), (50, 75), (75, 100)]
    
    print("Stride Range | Count | Window F1 (Mean±Std) | Word F1 (Mean±Std) | IoU (Mean±Std)")
    print("-" * 75)
    
    for low, high in stride_ranges:
        range_data = df_success[(df_success['stride_percentage'] >= low) & (df_success['stride_percentage'] < high)]
        if len(range_data) > 0:
            wf1_mean, wf1_std = range_data['window_f1'].mean(), range_data['window_f1'].std()
            word_mean, word_std = range_data['word_f1'].mean(), range_data['word_f1'].std()
            iou_mean, iou_std = range_data['combined_iou'].mean(), range_data['combined_iou'].std()
            
            print(f"{low:3d}-{high:2d}%     | {len(range_data):5d} | {wf1_mean:.3f}±{wf1_std:.3f}       | {word_mean:.3f}±{word_std:.3f}       | {iou_mean:.3f}±{iou_std:.3f}")
    
    # Handle 100% case
    range_100 = df_success[df_success['stride_percentage'] >= 100]
    if len(range_100) > 0:
        wf1_mean, wf1_std = range_100['window_f1'].mean(), range_100['window_f1'].std()
        word_mean, word_std = range_100['word_f1'].mean(), range_100['word_f1'].std()
        iou_mean, iou_std = range_100['combined_iou'].mean(), range_100['combined_iou'].std()
        print(f"   100%     | {len(range_100):5d} | {wf1_mean:.3f}±{wf1_std:.3f}       | {word_mean:.3f}±{word_std:.3f}       | {iou_mean:.3f}±{iou_std:.3f}")
    
    print()
    
    # SAVE ANALYSIS RESULTS
    output_dir = Path("fixed_smart_parallel_results/correlation_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save complete analysis data
    df_success.to_csv(output_dir / "complete_analysis_data.csv", index=False)
    
    # Save correlation matrix
    correlation_matrix.to_csv(output_dir / "correlation_matrix.csv")
    
    # Save top configurations
    top_10_window.to_csv(output_dir / "top_10_window_f1_configs.csv", index=False)
    top_10_word.to_csv(output_dir / "top_10_word_f1_configs.csv", index=False)
    
    # Save best per window
    best_per_window.to_csv(output_dir / "best_config_per_window.csv", index=False)
    
    print(f"📁 Analysis results saved to: {output_dir}")
    
    # KEY INSIGHTS SUMMARY
    print("\n💡 KEY INSIGHTS AND RECOMMENDATIONS:")
    
    # Overall best
    best_overall_window = df_success.loc[df_success['window_f1'].idxmax()]
    best_overall_word = df_success.loc[df_success['word_f1'].idxmax()]
    
    print(f"\n🏆 BEST OVERALL PERFORMANCE:")
    print(f"   Best Window F1: {best_overall_window['window_f1']:.4f}")
    print(f"   → {best_overall_window['eval_type']} | {best_overall_window['window_numeric']}s window | {best_overall_window['stride_percentage']:.1f}% stride")
    print(f"   → Configuration: {best_overall_window['window']}/{best_overall_window['stride']}")
    
    print(f"\n   Best Word F1: {best_overall_word['word_f1']:.4f}")
    print(f"   → {best_overall_word['eval_type']} | {best_overall_word['window_numeric']}s window | {best_overall_word['stride_percentage']:.1f}% stride")
    print(f"   → Configuration: {best_overall_word['window']}/{best_overall_word['stride']}")
    
    # Correlation insights
    window_corr = correlation_matrix.loc['window_numeric', 'window_f1']
    stride_corr = correlation_matrix.loc['stride_percentage', 'window_f1']
    
    print(f"\n📈 CORRELATION INSIGHTS:")
    print(f"   Window size effect on Window F1: {window_corr:.4f}")
    if abs(window_corr) > 0.3:
        trend = "increases" if window_corr > 0 else "decreases"
        print(f"   → Window F1 {trend} with larger window sizes (moderate correlation)")
    else:
        print(f"   → Window size has weak correlation with Window F1 performance")
    
    print(f"   Stride percentage effect on Window F1: {stride_corr:.4f}")
    if abs(stride_corr) > 0.3:
        trend = "increases" if stride_corr > 0 else "decreases"
        print(f"   → Window F1 {trend} with larger stride percentages (moderate correlation)")
    else:
        print(f"   → Stride percentage has weak correlation with Window F1 performance")
    
    # Evaluation type recommendation
    eval_05_mean = eval_05['window_f1'].mean()
    eval_percent_mean = eval_percent['window_f1'].mean()
    
    print(f"\n🎯 EVALUATION TYPE RECOMMENDATION:")
    if eval_05_mean > eval_percent_mean:
        print(f"   → eval_by_0.05 performs better on average (Window F1: {eval_05_mean:.4f} vs {eval_percent_mean:.4f})")
    else:
        print(f"   → eval_percent performs better on average (Window F1: {eval_percent_mean:.4f} vs {eval_05_mean:.4f})")
    
    print(f"\n✨ ANALYSIS COMPLETE!")

if __name__ == "__main__":
    main()

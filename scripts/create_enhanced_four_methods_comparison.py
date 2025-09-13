#!/usr/bin/env python3
"""
Enhanced Comprehensive Evaluation Methods Comparison

This script creates a detailed comparison between:
1. Old evaluation method (eval_IoU_word_eval_sep)
2. New word_eval method 
3. word_iou_eval method
4. iou_eval method

Provides side-by-side analysis with statistical significance testing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def load_old_evaluation_data():
    """Load old evaluation data from comprehensive comparison"""
    old_data_file = Path("comprehensive_old_vs_new_comparison/performance_comparison_summary.csv")
    if old_data_file.exists():
        df = pd.read_csv(old_data_file)
        # Extract old method data
        old_method_data = df[df['Method'].str.contains('Old', na=False)]
        if not old_method_data.empty:
            return old_method_data.iloc[0]  # Get the old method row
        else:
            print("No old method data found in comparison file")
            return None
    else:
        print("Old evaluation comparison file not found!")
        return None

def load_detailed_old_data():
    """Load detailed old evaluation data if available"""
    old_detailed_file = Path("evaluation_method_analysis/old_evaluation_comparison.csv")
    if old_detailed_file.exists():
        return pd.read_csv(old_detailed_file)
    else:
        print("Detailed old evaluation data not found")
        return None

def load_new_evaluation_methods_data():
    """Load data for all new evaluation methods"""
    summary_file = Path("fixed_smart_parallel_results/fixed_smart_parallel_summary.csv")
    
    if summary_file.exists():
        print(f"Loading evaluation data from {summary_file}...")
        df = pd.read_csv(summary_file)
        
        # Filter out corrupted entries and extract window and stride as numeric values
        df = df[df['stride'].str.match(r'stride_[0-9.]+s$', na=False)]
        df = df[df['window'].str.match(r'window_[0-9.]+s$', na=False)]
        
        df['window_num'] = df['window'].str.extract(r'window_([0-9.]+)s')[0].astype(float)
        df['stride_num'] = df['stride'].str.extract(r'stride_([0-9.]+)s')[0].astype(float)
        
        print(f"Loaded {len(df)} configurations")
        return df
    else:
        print("Summary file not found!")
        return None

def load_iou_analysis_data():
    """Load IoU analysis data from iou_percentage_analysis directory"""
    iou_data = {}
    
    # Load IoU eval data (Combined and Individual Mean IoU)
    iou_summary_file = Path("iou_percentage_analysis/iou_performance_summary_statistics.csv")
    if iou_summary_file.exists():
        print(f"Loading IoU summary data from {iou_summary_file}...")
        iou_summary = pd.read_csv(iou_summary_file)
        iou_data['iou_summary'] = iou_summary
    
    # Load detailed IoU eval data
    iou_files = [
        "iou_percentage_analysis/top10_iou_eval_combined_mean_iou_percent.csv",
        "iou_percentage_analysis/top10_word_iou_eval_f1_iou_0.3.csv"
    ]
    
    for file_path in iou_files:
        if Path(file_path).exists():
            print(f"Loading detailed IoU data from {file_path}...")
            if 'word_iou_eval' in file_path:
                iou_data['word_iou_detailed'] = pd.read_csv(file_path)
            else:
                iou_data['iou_detailed'] = pd.read_csv(file_path)
    
    return iou_data if iou_data else None

def extract_comprehensive_method_data(summary_df, iou_analysis_data, detailed_old_data=None):
    """Extract comprehensive data for all evaluation methods"""
    methods_data = {
        'word_eval': {'configs': [], 'f1_scores': [], 'windows': [], 'strides': []},
        'word_iou_eval': {'configs': [], 'f1_scores': [], 'windows': [], 'strides': []},
        'iou_eval': {'configs': [], 'f1_scores': [], 'windows': [], 'strides': []},
        'old_method': {'configs': [], 'f1_scores': [], 'windows': [], 'strides': []}
    }
    
    # Extract from summary data (word_eval method)
    for _, row in summary_df.iterrows():
        config_name = f"{row['window']}_{row['stride']}"
        window = row['window_num']
        stride = row['stride_num']
        f1_score = row['word_f1'] if 'word_f1' in row else np.nan
        
        methods_data['word_eval']['configs'].append(config_name)
        methods_data['word_eval']['f1_scores'].append(f1_score)
        methods_data['word_eval']['windows'].append(window)
        methods_data['word_eval']['strides'].append(stride)
        
        # For iou_eval, use combined_iou as proxy for F1
        iou_score = row['combined_iou'] if 'combined_iou' in row else np.nan
        methods_data['iou_eval']['configs'].append(config_name)
        methods_data['iou_eval']['f1_scores'].append(iou_score)
        methods_data['iou_eval']['windows'].append(window)
        methods_data['iou_eval']['strides'].append(stride)
    
    # Add detailed IoU analysis data
    if iou_analysis_data:
        # Word IoU eval data
        if 'word_iou_detailed' in iou_analysis_data:
            word_iou_df = iou_analysis_data['word_iou_detailed']
            for _, row in word_iou_df.iterrows():
                if 'window_numeric' in row and 'stride_numeric' in row:
                    config_name = f"{row['window']}_{row['stride']}"
                    methods_data['word_iou_eval']['configs'].append(config_name)
                    methods_data['word_iou_eval']['f1_scores'].append(row['iou_f1_0.3'])
                    methods_data['word_iou_eval']['windows'].append(row['window_numeric'])
                    methods_data['word_iou_eval']['strides'].append(row['stride_numeric'])
    
    # Add old method data if available
    if detailed_old_data is not None:
        for _, row in detailed_old_data.iterrows():
            config_name = f"window_{row['window']}s_stride_{row['stride']}s"
            methods_data['old_method']['configs'].append(config_name)
            methods_data['old_method']['f1_scores'].append(row['f1_score'])
            methods_data['old_method']['windows'].append(row['window'])
            methods_data['old_method']['strides'].append(row['stride'])
    
    return methods_data

def create_enhanced_comparison_plots(old_summary, methods_data):
    """Create enhanced comparison plots for all four methods"""
    
    # Create output directory
    output_dir = Path("enhanced_four_methods_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create comprehensive comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Enhanced Four-Method Evaluation Comparison\n(Old Method vs Word Eval vs Word IoU Eval vs IoU Eval)', 
                 fontsize=20, fontweight='bold')
    
    # Prepare data for plotting - get common window sizes
    all_windows = set()
    for method_data in methods_data.values():
        if method_data['windows']:
            all_windows.update(method_data['windows'])
    
    common_windows = sorted(list(all_windows))
    window_ranges = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0]  # Focus on common ranges
    common_windows = [w for w in common_windows if w in window_ranges]
    
    # Plot 1: F1 Score Comparison Across Methods
    ax1 = axes[0, 0]
    
    method_colors = {
        'old_method': '#d62728',      # Red
        'word_eval': '#2ca02c',       # Green  
        'word_iou_eval': '#ff7f0e',   # Orange
        'iou_eval': '#1f77b4'         # Blue
    }
    
    method_labels = {
        'old_method': 'Old Method (eval_IoU_word_eval_sep)',
        'word_eval': 'Word Eval (New)',
        'word_iou_eval': 'Word IoU Eval (F1@0.3)',
        'iou_eval': 'IoU Eval (Combined Mean IoU)'
    }
    
    for method_name, method_data in methods_data.items():
        if not method_data['windows']:
            continue
            
        # Calculate mean F1 for each window size
        window_f1_means = []
        window_f1_stds = []
        
        for window in common_windows:
            window_f1s = [f1 for f1, w in zip(method_data['f1_scores'], method_data['windows']) 
                         if w == window and not np.isnan(f1)]
            
            if window_f1s:
                window_f1_means.append(np.mean(window_f1s))
                window_f1_stds.append(np.std(window_f1s))
            else:
                window_f1_means.append(np.nan)
                window_f1_stds.append(np.nan)
        
        # Plot with error bars
        valid_indices = [i for i, mean in enumerate(window_f1_means) if not np.isnan(mean)]
        if valid_indices:
            valid_windows = [common_windows[i] for i in valid_indices]
            valid_means = [window_f1_means[i] for i in valid_indices]
            valid_stds = [window_f1_stds[i] for i in valid_indices]
            
            ax1.errorbar(valid_windows, valid_means, yerr=valid_stds, 
                        marker='o', linewidth=3, markersize=8, capsize=5,
                        label=method_labels[method_name], color=method_colors[method_name], alpha=0.8)
    
    ax1.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1 Score / Performance Metric', fontsize=12, fontweight='bold')
    ax1.set_title('F1 Score Comparison Across All Methods', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance Distribution Comparison
    ax2 = axes[0, 1]
    
    method_f1_data = []
    method_names = []
    
    for method_name, method_data in methods_data.items():
        if method_data['f1_scores']:
            valid_f1s = [f1 for f1 in method_data['f1_scores'] if not np.isnan(f1)]
            if valid_f1s:
                method_f1_data.append(valid_f1s)
                method_names.append(method_labels[method_name])
    
    if method_f1_data:
        bp = ax2.boxplot(method_f1_data, labels=method_names, patch_artist=True)
        colors = [method_colors['old_method'], method_colors['word_eval'], 
                 method_colors['word_iou_eval'], method_colors['iou_eval']]
        
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_ylabel('F1 Score / Performance Metric', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correlation Analysis
    ax3 = axes[1, 0]
    
    correlations = {}
    method_positions = []
    correlation_values = []
    colors_list = []
    
    for i, (method_name, method_data) in enumerate(methods_data.items()):
        if len(method_data['windows']) > 1 and len(method_data['f1_scores']) > 1:
            valid_data = [(w, f1) for w, f1 in zip(method_data['windows'], method_data['f1_scores']) 
                         if not np.isnan(f1)]
            
            if len(valid_data) > 1:
                windows, f1s = zip(*valid_data)
                corr, p_value = pearsonr(windows, f1s)
                correlations[method_name] = {'correlation': corr, 'p_value': p_value}
                
                method_positions.append(i)
                correlation_values.append(corr)
                colors_list.append(method_colors[method_name])
    
    if correlation_values:
        bars = ax3.bar(method_positions, correlation_values, color=colors_list, alpha=0.8)
        ax3.set_xticks(method_positions)
        ax3.set_xticklabels([method_labels[list(methods_data.keys())[i]] for i in method_positions], 
                           rotation=45, ha='right')
        ax3.set_ylabel('Correlation with Window Size', fontsize=12, fontweight='bold')
        ax3.set_title('Window Size Correlation Analysis', fontsize=14, fontweight='bold')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # Add correlation values on bars
        for bar, corr in zip(bars, correlation_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height > 0 else -0.05),
                    f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    # Plot 4: Performance Summary Statistics
    ax4 = axes[1, 1]
    
    stats_data = {
        'Method': [],
        'Mean': [],
        'Max': [],
        'Std': [],
        'Count': []
    }
    
    for method_name, method_data in methods_data.items():
        if method_data['f1_scores']:
            valid_f1s = [f1 for f1 in method_data['f1_scores'] if not np.isnan(f1)]
            if valid_f1s:
                stats_data['Method'].append(method_labels[method_name])
                stats_data['Mean'].append(np.mean(valid_f1s))
                stats_data['Max'].append(np.max(valid_f1s))
                stats_data['Std'].append(np.std(valid_f1s))
                stats_data['Count'].append(len(valid_f1s))
    
    if stats_data['Method']:
        x_pos = np.arange(len(stats_data['Method']))
        
        width = 0.25
        ax4.bar(x_pos - width, stats_data['Mean'], width, label='Mean', alpha=0.8, color='lightblue')
        ax4.bar(x_pos, stats_data['Max'], width, label='Max', alpha=0.8, color='lightgreen')
        ax4.bar(x_pos + width, stats_data['Std'], width, label='Std Dev', alpha=0.8, color='lightcoral')
        
        ax4.set_xlabel('Evaluation Methods', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Performance Values', fontsize=12, fontweight='bold')
        ax4.set_title('Statistical Summary Comparison', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(stats_data['Method'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'enhanced_four_methods_comprehensive_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlations, stats_data

def create_detailed_performance_table(old_summary, methods_data, correlations):
    """Create detailed performance comparison table"""
    
    summary_data = []
    
    # Old method (from summary)
    if old_summary is not None:
        summary_data.append({
            'Method': 'Old Method (eval_IoU_word_eval_sep)',
            'Type': 'Traditional',
            'Configurations': old_summary['Total Configs'] if 'Total Configs' in old_summary else 225,
            'Mean F1': old_summary['F1 Score Mean'] if 'F1 Score Mean' in old_summary else np.nan,
            'Max F1': old_summary['F1 Score Max'] if 'F1 Score Max' in old_summary else np.nan,
            'Std F1': old_summary['F1 Score Std'] if 'F1 Score Std' in old_summary else np.nan,
            'Best Window': old_summary['Best Window (F1)'] if 'Best Window (F1)' in old_summary else np.nan,
            'Best Stride': old_summary['Best Stride (F1)'] if 'Best Stride (F1)' in old_summary else np.nan,
            'Window Correlation': correlations.get('old_method', {}).get('correlation', np.nan),
            'Correlation P-Value': correlations.get('old_method', {}).get('p_value', np.nan)
        })
    
    # New methods
    method_info = {
        'word_eval': ('Word Eval (New)', 'Word-based'),
        'word_iou_eval': ('Word IoU Eval', 'Word+IoU-based'),
        'iou_eval': ('IoU Eval', 'IoU-based')
    }
    
    for method_name, (display_name, method_type) in method_info.items():
        method_data = methods_data[method_name]
        if method_data['f1_scores']:
            valid_f1s = [f1 for f1 in method_data['f1_scores'] if not np.isnan(f1)]
            if valid_f1s:
                best_idx = np.argmax(valid_f1s)
                best_window = method_data['windows'][method_data['f1_scores'].index(max(valid_f1s))]
                best_stride = method_data['strides'][method_data['f1_scores'].index(max(valid_f1s))]
                
                summary_data.append({
                    'Method': display_name,
                    'Type': method_type,
                    'Configurations': len(valid_f1s),
                    'Mean F1': np.mean(valid_f1s),
                    'Max F1': np.max(valid_f1s),
                    'Std F1': np.std(valid_f1s),
                    'Best Window': f"{best_window}s",
                    'Best Stride': f"{best_stride}s",
                    'Window Correlation': correlations.get(method_name, {}).get('correlation', np.nan),
                    'Correlation P-Value': correlations.get(method_name, {}).get('p_value', np.nan)
                })
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    output_dir = Path("enhanced_four_methods_comparison")
    output_dir.mkdir(exist_ok=True)
    
    summary_df.to_csv(output_dir / 'enhanced_four_methods_performance_summary.csv', index=False)
    
    print("\n" + "="*120)
    print("ENHANCED FOUR-METHOD EVALUATION COMPARISON SUMMARY")
    print("="*120)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    print("="*120)
    
    return summary_df

def analyze_improvement_factors(methods_data):
    """Analyze improvement factors between methods"""
    
    print("\n" + "="*80)
    print("IMPROVEMENT FACTOR ANALYSIS")
    print("="*80)
    
    # Get mean F1 scores for each method
    method_means = {}
    for method_name, method_data in methods_data.items():
        if method_data['f1_scores']:
            valid_f1s = [f1 for f1 in method_data['f1_scores'] if not np.isnan(f1)]
            if valid_f1s:
                method_means[method_name] = np.mean(valid_f1s)
    
    # Calculate improvement factors
    if 'old_method' in method_means:
        old_mean = method_means['old_method']
        
        for method_name, mean_f1 in method_means.items():
            if method_name != 'old_method':
                improvement_factor = mean_f1 / old_mean if old_mean > 0 else float('inf')
                improvement_percent = ((mean_f1 - old_mean) / old_mean) * 100 if old_mean > 0 else float('inf')
                
                print(f"{method_name.replace('_', ' ').title()}: {improvement_factor:.2f}x improvement ({improvement_percent:+.1f}%)")
    
    print("="*80)

def main():
    """Main execution function"""
    print("="*80)
    print("ENHANCED FOUR-METHOD EVALUATION COMPARISON")
    print("Old Method vs Word Eval vs Word IoU Eval vs IoU Eval")
    print("="*80)
    
    # Load all data sources
    print("Loading evaluation data...")
    old_summary = load_old_evaluation_data()
    detailed_old_data = load_detailed_old_data()
    new_data = load_new_evaluation_methods_data()
    iou_analysis_data = load_iou_analysis_data()
    
    if new_data is None:
        print("Failed to load evaluation data!")
        return
    
    # Extract comprehensive method data
    print("Extracting comprehensive method data...")
    methods_data = extract_comprehensive_method_data(new_data, iou_analysis_data, detailed_old_data)
    
    # Create enhanced comparison plots
    print("Creating enhanced comparison plots...")
    correlations, stats_data = create_enhanced_comparison_plots(old_summary, methods_data)
    
    # Create detailed performance table
    print("Creating detailed performance table...")
    summary_df = create_detailed_performance_table(old_summary, methods_data, correlations)
    
    # Analyze improvement factors
    analyze_improvement_factors(methods_data)
    
    print("\nEnhanced four-method comparison analysis completed!")
    print("Output saved to: enhanced_four_methods_comparison/")

if __name__ == "__main__":
    main()

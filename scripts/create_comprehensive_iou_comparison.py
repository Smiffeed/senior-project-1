#!/usr/bin/env python3
"""
Comprehensive IoU Evaluation Methods Comparison

This script creates a comprehensive comparison between:
1. Old evaluation method (from note.txt files)
2. New word_eval method 
3. word_iou_eval method
4. iou_eval method

Focus on IoU-based evaluations and their performance patterns.
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
    """Load old evaluation data from CSV"""
    old_data_file = Path("evaluation_method_analysis/old_evaluation_comparison.csv")
    if old_data_file.exists():
        return pd.read_csv(old_data_file)
    else:
        print("Old evaluation data not found!")
        return None

def load_new_evaluation_data():
    """Load new evaluation data from summary CSV"""
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
    
    # Load IoU eval data
    iou_files = [
        "iou_percentage_analysis/top10_iou_eval_combined_mean_iou_percent.csv",
        "iou_percentage_analysis/iou_performance_summary_statistics.csv"
    ]
    
    for file_path in iou_files:
        if Path(file_path).exists():
            print(f"Loading IoU data from {file_path}...")
            iou_data['iou_eval'] = pd.read_csv(file_path)
            break
    
    # Load Word IoU eval data
    word_iou_files = [
        "iou_percentage_analysis/top10_word_iou_eval_f1_iou_0.3.csv"
    ]
    
    for file_path in word_iou_files:
        if Path(file_path).exists():
            print(f"Loading Word IoU data from {file_path}...")
            iou_data['word_iou_eval'] = pd.read_csv(file_path)
            break
    
    return iou_data if iou_data else None

def extract_method_specific_data(summary_df, iou_analysis_data=None):
    """Extract data for each evaluation method from available data"""
    methods_data = {
        'word_eval': {},
        'word_iou_eval': {},
        'iou_eval': {}
    }
    
    # Extract from summary data (basic metrics)
    for _, row in summary_df.iterrows():
        config_name = f"{row['window']}_{row['stride']}"
        window = row['window_num']
        stride = row['stride_num']
        
        # Basic data available in summary
        basic_data = {
            'window': window,
            'stride': stride,
            'window_f1': row['window_f1'] if 'window_f1' in row else np.nan,
            'word_f1': row['word_f1'] if 'word_f1' in row else np.nan,
            'combined_iou': row['combined_iou'] if 'combined_iou' in row else np.nan
        }
        
        # Store as word_eval (main method)
        methods_data['word_eval'][config_name] = {
            **basic_data,
            'f1_score': basic_data['word_f1']
        }
        
        # Store as iou_eval
        methods_data['iou_eval'][config_name] = {
            **basic_data,
            'combined_mean_iou': basic_data['combined_iou'] * 100 if not np.isnan(basic_data['combined_iou']) else np.nan  # Convert to percentage
        }
    
    # If IoU analysis data is available, extract more specific metrics
    if iou_analysis_data is not None:
        print("Extracting IoU analysis data...")
        
        # Extract IoU eval data
        if 'iou_eval' in iou_analysis_data:
            iou_df = iou_analysis_data['iou_eval']
            if 'window_numeric' in iou_df.columns:  # Top 10 format
                for _, row in iou_df.iterrows():
                    window = row['window_numeric']
                    stride = row['stride_numeric']
                    config_name = f"{row['window']}_{row['stride']}"
                    
                    if config_name in methods_data['iou_eval']:
                        methods_data['iou_eval'][config_name].update({
                            'iou_binary_f1_0.3': row['iou_f1_0.3'] if 'iou_f1_0.3' in row else np.nan,
                            'iou_binary_f1_0.5': row['iou_f1_0.5'] if 'iou_f1_0.5' in row else np.nan,
                            'iou_binary_f1_0.9': row['iou_f1_0.9'] if 'iou_f1_0.9' in row else np.nan,
                        })
        
        # Extract Word IoU eval data
        if 'word_iou_eval' in iou_analysis_data:
            word_iou_df = iou_analysis_data['word_iou_eval']
            for _, row in word_iou_df.iterrows():
                window = row['window_numeric']
                stride = row['stride_numeric']
                config_name = f"{row['window']}_{row['stride']}"
                
                methods_data['word_iou_eval'][config_name] = {
                    'window': window,
                    'stride': stride,
                    'f1_iou_0.3': row['iou_f1_0.3'] if 'iou_f1_0.3' in row else np.nan,
                    'precision_0.3': row['precision_0.3'] if 'precision_0.3' in row else np.nan,
                    'recall_0.3': row['recall_0.3'] if 'recall_0.3' in row else np.nan,
                }
    
    return methods_data

def create_comprehensive_comparison_plots(old_data, methods_data):
    """Create comprehensive comparison plots including IoU evaluations"""
    
    # Create output directory
    output_dir = Path("comprehensive_iou_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. F1 Score Comparison Across All Methods
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Comprehensive IoU Evaluation Methods Comparison', fontsize=20, fontweight='bold')
    
    # Prepare data for plotting
    window_sizes = []
    old_f1_scores = []
    word_eval_f1_scores = []
    word_iou_f1_scores = []
    iou_eval_f1_scores = []
    
    # Get common window sizes
    if old_data is not None:
        old_windows = sorted(old_data['window'].unique())
    else:
        old_windows = []
    
    new_windows = sorted([data['window'] for data in methods_data['word_eval'].values()])
    common_windows = sorted(set(old_windows).intersection(set(new_windows))) if old_data is not None else sorted(set(new_windows))
    
    for window in common_windows:
        # Old method data
        if old_data is not None:
            old_window_data = old_data[old_data['window'] == window]
            if not old_window_data.empty:
                old_f1 = old_window_data['f1_score'].mean()
                old_f1_scores.append(old_f1)
            else:
                old_f1_scores.append(np.nan)
        
        # New word_eval data
        word_eval_window_data = [data for data in methods_data['word_eval'].values() if data['window'] == window]
        if word_eval_window_data:
            word_f1 = np.mean([data['f1_score'] for data in word_eval_window_data if not np.isnan(data['f1_score'])])
            word_eval_f1_scores.append(word_f1)
        else:
            word_eval_f1_scores.append(np.nan)
        
        # Word IoU eval data (using F1 IoU 0.3 as representative)
        word_iou_window_data = [data for data in methods_data['word_iou_eval'].values() if data['window'] == window]
        if word_iou_window_data:
            word_iou_f1_values = [data['f1_iou_0.3'] for data in word_iou_window_data if 'f1_iou_0.3' in data and not np.isnan(data['f1_iou_0.3'])]
            if word_iou_f1_values:
                word_iou_f1 = np.mean(word_iou_f1_values)
                word_iou_f1_scores.append(word_iou_f1)
            else:
                word_iou_f1_scores.append(np.nan)
        else:
            word_iou_f1_scores.append(np.nan)
        
        # IoU eval data (using F1 IoU 0.3 as representative, fallback to combined_mean_iou)
        iou_window_data = [data for data in methods_data['iou_eval'].values() if data['window'] == window]
        if iou_window_data:
            # Try to get F1 IoU 0.3, otherwise use combined_mean_iou/100 as proxy
            iou_f1_values = []
            for data in iou_window_data:
                if 'iou_binary_f1_0.3' in data and not np.isnan(data['iou_binary_f1_0.3']):
                    iou_f1_values.append(data['iou_binary_f1_0.3'])
                elif 'combined_mean_iou' in data and not np.isnan(data['combined_mean_iou']):
                    # Use combined_mean_iou/100 as a proxy for F1 score
                    iou_f1_values.append(data['combined_mean_iou'] / 100)
            
            if iou_f1_values:
                iou_f1 = np.mean(iou_f1_values)
                iou_eval_f1_scores.append(iou_f1)
            else:
                iou_eval_f1_scores.append(np.nan)
        else:
            iou_eval_f1_scores.append(np.nan)
        
        window_sizes.append(window)
    
    # Plot 1: F1 Score Comparison
    ax1 = axes[0, 0]
    if old_data is not None and old_f1_scores:
        ax1.plot(window_sizes, old_f1_scores, 'o-', linewidth=3, markersize=8, label='Old Method', alpha=0.8)
    if word_eval_f1_scores:
        ax1.plot(window_sizes, word_eval_f1_scores, 's-', linewidth=3, markersize=8, label='Word Eval (New)', alpha=0.8)
    if word_iou_f1_scores:
        ax1.plot(window_sizes, word_iou_f1_scores, '^-', linewidth=3, markersize=8, label='Word IoU Eval (F1@0.3)', alpha=0.8)
    if iou_eval_f1_scores:
        ax1.plot(window_sizes, iou_eval_f1_scores, 'D-', linewidth=3, markersize=8, label='IoU Eval (F1@0.3)', alpha=0.8)
    
    ax1.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1 Score Comparison Across Methods', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: IoU Evaluation Methods Comparison (Different IoU Thresholds)
    ax2 = axes[0, 1]
    
    # Plot different IoU thresholds for word_iou_eval (only plot what we have)
    available_thresholds = []
    if methods_data['word_iou_eval']:
        sample_data = next(iter(methods_data['word_iou_eval'].values()))
        available_thresholds = [key for key in sample_data.keys() if key.startswith('f1_iou')]
    
    for threshold in available_thresholds:
        threshold_scores = []
        for window in common_windows:
            window_data = [data for data in methods_data['word_iou_eval'].values() if data['window'] == window]
            if window_data:
                scores = [data[threshold] for data in window_data if threshold in data and not np.isnan(data[threshold])]
                if scores:
                    threshold_scores.append(np.mean(scores))
                else:
                    threshold_scores.append(np.nan)
            else:
                threshold_scores.append(np.nan)
        
        if any(not np.isnan(score) for score in threshold_scores):
            ax2.plot(window_sizes, threshold_scores, 'o-', linewidth=2, markersize=6, 
                    label=f'Word IoU {threshold.replace("f1_iou_", "F1@")}', alpha=0.8)
    
    ax2.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Word IoU Evaluation - Different IoU Thresholds', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: IoU Evaluation Methods Comparison (IoU Binary F1)
    ax3 = axes[1, 0]
    
    # Plot available IoU thresholds for iou_eval
    available_iou_thresholds = []
    if methods_data['iou_eval']:
        sample_data = next(iter(methods_data['iou_eval'].values()))
        available_iou_thresholds = [key for key in sample_data.keys() if key.startswith('iou_binary_f1')]
    
    for threshold in available_iou_thresholds:
        threshold_scores = []
        for window in common_windows:
            window_data = [data for data in methods_data['iou_eval'].values() if data['window'] == window]
            if window_data:
                scores = [data[threshold] for data in window_data if threshold in data and not np.isnan(data[threshold])]
                if scores:
                    threshold_scores.append(np.mean(scores))
                else:
                    threshold_scores.append(np.nan)
            else:
                threshold_scores.append(np.nan)
        
        if any(not np.isnan(score) for score in threshold_scores):
            ax3.plot(window_sizes, threshold_scores, 's-', linewidth=2, markersize=6, 
                    label=f'IoU {threshold.replace("iou_binary_f1_", "F1@")}', alpha=0.8)
    
    ax3.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax3.set_title('IoU Evaluation - Different IoU Thresholds', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mean IoU Comparison
    ax4 = axes[1, 1]
    
    # Combined Mean IoU
    combined_iou_scores = []
    individual_iou_scores = []
    
    for window in common_windows:
        window_data = [data for data in methods_data['iou_eval'].values() if data['window'] == window]
        if window_data:
            # Combined Mean IoU
            combined_scores = [data['combined_mean_iou'] for data in window_data if 'combined_mean_iou' in data and not np.isnan(data['combined_mean_iou'])]
            if combined_scores:
                combined_iou_scores.append(np.mean(combined_scores))
            else:
                combined_iou_scores.append(np.nan)
                
            # Individual Mean IoU (if available)
            individual_scores = [data['individual_mean_iou'] for data in window_data if 'individual_mean_iou' in data and not np.isnan(data['individual_mean_iou'])]
            if individual_scores:
                individual_iou_scores.append(np.mean(individual_scores))
            else:
                individual_iou_scores.append(np.nan)
        else:
            combined_iou_scores.append(np.nan)
            individual_iou_scores.append(np.nan)
    
    if any(not np.isnan(score) for score in combined_iou_scores):
        ax4.plot(window_sizes, combined_iou_scores, 'o-', linewidth=3, markersize=8, 
                label='Combined Mean IoU', alpha=0.8)
    if any(not np.isnan(score) for score in individual_iou_scores):
        ax4.plot(window_sizes, individual_iou_scores, 's-', linewidth=3, markersize=8, 
                label='Individual Mean IoU', alpha=0.8)
    
    ax4.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Mean IoU (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Mean IoU Comparison', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_iou_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_summary_table(old_data, methods_data):
    """Create a performance summary table for all methods"""
    
    summary_data = []
    
    # Old method summary
    if old_data is not None:
        summary_data.append({
            'Method': 'Old Evaluation',
            'Type': 'Traditional',
            'Best F1': old_data['f1_score'].max(),
            'Mean F1': old_data['f1_score'].mean(),
            'Std F1': old_data['f1_score'].std(),
            'Best Window': old_data.loc[old_data['f1_score'].idxmax(), 'window'],
            'Best Stride': old_data.loc[old_data['f1_score'].idxmax(), 'stride'],
            'Configurations': len(old_data)
        })
    
    # Word eval summary
    word_eval_f1s = [data['f1_score'] for data in methods_data['word_eval'].values() if not np.isnan(data['f1_score'])]
    if word_eval_f1s:
        best_idx = np.argmax(word_eval_f1s)
        best_config = list(methods_data['word_eval'].values())[best_idx]
        
        summary_data.append({
            'Method': 'Word Eval (New)',
            'Type': 'Word-based',
            'Best F1': max(word_eval_f1s),
            'Mean F1': np.mean(word_eval_f1s),
            'Std F1': np.std(word_eval_f1s),
            'Best Window': best_config['window'],
            'Best Stride': best_config['stride'],
            'Configurations': len(word_eval_f1s)
        })
    
    # Word IoU eval summary (using F1@0.3)
    word_iou_f1s = []
    word_iou_configs = []
    for data in methods_data['word_iou_eval'].values():
        if 'f1_iou_0.3' in data and not np.isnan(data['f1_iou_0.3']):
            word_iou_f1s.append(data['f1_iou_0.3'])
            word_iou_configs.append(data)
    
    if word_iou_f1s:
        best_idx = np.argmax(word_iou_f1s)
        best_config = word_iou_configs[best_idx]
        
        summary_data.append({
            'Method': 'Word IoU Eval (F1@0.3)',
            'Type': 'Word+IoU-based',
            'Best F1': max(word_iou_f1s),
            'Mean F1': np.mean(word_iou_f1s),
            'Std F1': np.std(word_iou_f1s),
            'Best Window': best_config['window'],
            'Best Stride': best_config['stride'],
            'Configurations': len(word_iou_f1s)
        })
    
    # IoU eval summary (using Combined Mean IoU as primary metric)
    iou_scores = []
    iou_configs = []
    for data in methods_data['iou_eval'].values():
        if 'combined_mean_iou' in data and not np.isnan(data['combined_mean_iou']):
            iou_scores.append(data['combined_mean_iou'])
            iou_configs.append(data)
    
    if iou_scores:
        best_idx = np.argmax(iou_scores)
        best_config = iou_configs[best_idx]
        
        summary_data.append({
            'Method': 'IoU Eval (Combined Mean IoU)',
            'Type': 'IoU-based',
            'Best F1': max(iou_scores),  # Using IoU% as metric
            'Mean F1': np.mean(iou_scores),
            'Std F1': np.std(iou_scores),
            'Best Window': best_config['window'],
            'Best Stride': best_config['stride'],
            'Configurations': len(iou_scores)
        })
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    output_dir = Path("comprehensive_iou_comparison")
    output_dir.mkdir(exist_ok=True)
    
    summary_df.to_csv(output_dir / 'methods_performance_summary.csv', index=False)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE IoU EVALUATION METHODS COMPARISON SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    print("="*80)
    
    return summary_df

def analyze_correlation_patterns(methods_data):
    """Analyze correlation patterns for each method"""
    
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS - WINDOW SIZE EFFECTS")
    print("="*80)
    
    correlations = {}
    
    # Word eval correlations
    word_eval_data = methods_data['word_eval']
    if word_eval_data:
        windows = [data['window'] for data in word_eval_data.values()]
        f1_scores = [data['f1_score'] for data in word_eval_data.values() if not np.isnan(data['f1_score'])]
        windows_clean = [windows[i] for i in range(len(windows)) if not np.isnan(list(word_eval_data.values())[i]['f1_score'])]
        
        if len(windows_clean) > 1 and len(f1_scores) > 1:
            corr, p_value = pearsonr(windows_clean, f1_scores)
            correlations['Word Eval'] = {'correlation': corr, 'p_value': p_value}
            print(f"Word Eval (New Method): {corr:+.3f} (p={p_value:.2e})")
    
    # Word IoU eval correlations (F1@0.3)
    word_iou_data = methods_data['word_iou_eval']
    if word_iou_data:
        valid_data = [(data['window'], data['f1_iou_0.3']) for data in word_iou_data.values() 
                     if 'f1_iou_0.3' in data and not np.isnan(data['f1_iou_0.3'])]
        
        if len(valid_data) > 1:
            windows_clean, f1_scores = zip(*valid_data)
            corr, p_value = pearsonr(windows_clean, f1_scores)
            correlations['Word IoU Eval'] = {'correlation': corr, 'p_value': p_value}
            print(f"Word IoU Eval (F1@0.3): {corr:+.3f} (p={p_value:.2e})")
    
    # IoU eval correlations (use combined_mean_iou if F1 not available)
    iou_data = methods_data['iou_eval']
    if iou_data:
        valid_data = []
        for data in iou_data.values():
            if 'iou_binary_f1_0.3' in data and not np.isnan(data['iou_binary_f1_0.3']):
                valid_data.append((data['window'], data['iou_binary_f1_0.3']))
            elif 'combined_mean_iou' in data and not np.isnan(data['combined_mean_iou']):
                valid_data.append((data['window'], data['combined_mean_iou'] / 100))
        
        if len(valid_data) > 1:
            windows_clean, scores = zip(*valid_data)
            corr, p_value = pearsonr(windows_clean, scores)
            correlations['IoU Eval'] = {'correlation': corr, 'p_value': p_value}
            print(f"IoU Eval (Combined Mean IoU): {corr:+.3f} (p={p_value:.2e})")
    
    print("="*80)
    return correlations

def main():
    """Main execution function"""
    print("="*80)
    print("COMPREHENSIVE IoU EVALUATION METHODS COMPARISON")
    print("="*80)
    
    # Load data
    print("Loading evaluation data...")
    old_data = load_old_evaluation_data()
    new_data = load_new_evaluation_data()
    iou_analysis_data = load_iou_analysis_data()
    
    if new_data is None:
        print("Failed to load evaluation data!")
        return
    
    # Extract method-specific data
    print("Extracting method-specific data...")
    methods_data = extract_method_specific_data(new_data, iou_analysis_data)
    
    # Create comprehensive comparison plots
    print("Creating comprehensive comparison plots...")
    create_comprehensive_comparison_plots(old_data, methods_data)
    
    # Create performance summary
    print("Creating performance summary...")
    summary_df = create_performance_summary_table(old_data, methods_data)
    
    # Analyze correlation patterns
    correlations = analyze_correlation_patterns(methods_data)
    
    print("\nComprehensive IoU comparison analysis completed!")
    print("Output saved to: comprehensive_iou_comparison/")

if __name__ == "__main__":
    main()

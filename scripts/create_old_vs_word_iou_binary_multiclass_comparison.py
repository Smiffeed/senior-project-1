#!/usr/bin/env python3
"""
Old vs Word IoU Eval Comparison (Binary & Multiclass F1)
Create comparison plots showing both binary and multiclass F1 scores for word_iou_eval vs old method
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_old_evaluation_data():
    """Load old evaluation data from comprehensive comparison"""
    old_data_file = Path("comparison_graphs/eval_IoU_word_eval_sep_detailed.csv")
    if old_data_file.exists():
        old_data = pd.read_csv(old_data_file)
        print(f"✅ Loaded old evaluation data: {len(old_data)} configurations")
        return old_data
    else:
        print(f"❌ Old evaluation data not found at: {old_data_file}")
        return None

def load_word_iou_eval_detailed_data():
    """Load detailed Word IoU evaluation data with binary and multiclass F1"""
    
    word_iou_base_path = Path("fixed_smart_parallel_results/eval_by_0.05/eval_by_0.05/word_iou_eval")
    all_data = []
    
    if not word_iou_base_path.exists():
        print(f"❌ Word IoU eval path not found: {word_iou_base_path}")
        return None
    
    print("📊 Loading detailed Word IoU evaluation data...")
    
    # Iterate through all window/stride combinations
    for window_dir in word_iou_base_path.glob("window_*"):
        if not window_dir.is_dir():
            continue
            
        window_size = float(window_dir.name.replace('window_', '').replace('s', ''))
        
        for stride_dir in window_dir.glob("stride_*"):
            if not stride_dir.is_dir():
                continue
                
            stride_size = float(stride_dir.name.replace('stride_', '').replace('s', ''))
            
            # Load the word_iou_threshold_results.csv file
            results_file = stride_dir / "word_iou_threshold_results.csv"
            if results_file.exists():
                try:
                    threshold_df = pd.read_csv(results_file)
                    
                    # For each IoU threshold, create a record
                    for _, row in threshold_df.iterrows():
                        config_data = {
                            'window': window_size,
                            'stride': stride_size,
                            'iou_threshold': row['threshold'],
                            'binary_f1': row['binary_f1'],
                            'binary_precision': row['binary_precision'],
                            'binary_recall': row['binary_recall'],
                            'multiclass_f1': row['multiclass_f1'],
                            'multiclass_precision': row['multiclass_precision'],
                            'multiclass_recall': row['multiclass_recall'],
                            'config': f"window_{window_size}s_stride_{stride_size}s"
                        }
                        all_data.append(config_data)
                        
                except Exception as e:
                    print(f"⚠️  Error loading {results_file}: {e}")
                    continue
    
    if all_data:
        word_iou_df = pd.DataFrame(all_data)
        print(f"✅ Loaded Word IoU detailed data: {len(word_iou_df)} records")
        return word_iou_df
    else:
        print("❌ No Word IoU evaluation data found")
        return None

def create_binary_multiclass_comparison_plots(old_data, word_iou_data):
    """Create comparison plots showing binary and multiclass F1 scores"""
    
    if old_data is None or word_iou_data is None:
        print("❌ Cannot create plots - missing data")
        return
    
    # Set up the plot style
    plt.style.use('default')
    
    # Create figure with 6 subplots (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Old vs Word IoU Eval: Complete Binary & Multiclass F1 Comparison', fontsize=20, fontweight='bold')
    
    # Get common window sizes
    old_windows = sorted(old_data['window_numeric'].unique())
    word_iou_windows = sorted(word_iou_data['window'].unique())
    common_windows = sorted(set(old_windows).intersection(set(word_iou_windows)))
    
    print(f"ℹ️  Common windows for comparison: {common_windows}")
    
    # Use IoU threshold 0.3 as representative (good balance of precision/recall)
    iou_threshold = 0.3
    word_iou_filtered = word_iou_data[word_iou_data['iou_threshold'] == iou_threshold]
    
    # Prepare data for plotting
    old_f1_scores = []
    old_weighted_f1_scores = []
    old_macro_f1_scores = []
    word_iou_binary_f1_scores = []
    word_iou_multiclass_f1_scores = []
    
    for window in common_windows:
        # Old method data
        old_window_data = old_data[abs(old_data['window_numeric'] - window) < 0.05]
        if not old_window_data.empty:
            old_f1 = old_window_data['f1_score'].mean()
            old_weighted_f1 = old_window_data['weighted_f1'].mean()
            old_macro_f1 = old_window_data['macro_f1'].mean()
            old_f1_scores.append(old_f1)
            old_weighted_f1_scores.append(old_weighted_f1)
            old_macro_f1_scores.append(old_macro_f1)
        else:
            old_f1_scores.append(np.nan)
            old_weighted_f1_scores.append(np.nan)
            old_macro_f1_scores.append(np.nan)
        
        # Word IoU Binary F1 data
        word_iou_window_data = word_iou_filtered[abs(word_iou_filtered['window'] - window) < 0.05]
        if not word_iou_window_data.empty:
            binary_f1 = word_iou_window_data['binary_f1'].mean()
            multiclass_f1 = word_iou_window_data['multiclass_f1'].mean()
            word_iou_binary_f1_scores.append(binary_f1)
            word_iou_multiclass_f1_scores.append(multiclass_f1)
        else:
            word_iou_binary_f1_scores.append(np.nan)
            word_iou_multiclass_f1_scores.append(np.nan)
    
    # Plot 1: Old Evaluation Method - Binary F1
    ax1 = axes[0, 0]
    if old_f1_scores and any(not np.isnan(score) for score in old_f1_scores):
        ax1.plot(common_windows, old_f1_scores, 'o-', color='#FF6B6B', 
                linewidth=3, markersize=8, label='eval_IoU_word_eval_sep (Binary F1)')
        
        # Add confidence interval
        old_std_scores = []
        for window in common_windows:
            old_window_data = old_data[abs(old_data['window_numeric'] - window) < 0.05]
            if not old_window_data.empty and len(old_window_data) > 1:
                old_std_scores.append(old_window_data['f1_score'].std())
            else:
                old_std_scores.append(0.01)  # Small default
        
        ax1.fill_between(common_windows, 
                        np.array(old_f1_scores) - np.array(old_std_scores),
                        np.array(old_f1_scores) + np.array(old_std_scores),
                        alpha=0.3, color='#FF6B6B')
    
    ax1.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Binary F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title('Old Method - Binary F1 Score', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Old Evaluation Method - Weighted F1 (Multiclass)
    ax2 = axes[0, 1]
    if old_weighted_f1_scores and any(not np.isnan(score) for score in old_weighted_f1_scores):
        ax2.plot(common_windows, old_weighted_f1_scores, 's-', color='#FF8C94', 
                linewidth=3, markersize=8, label='eval_IoU_word_eval_sep (Weighted F1)')
        
        # Add confidence interval
        old_weighted_std_scores = []
        for window in common_windows:
            old_window_data = old_data[abs(old_data['window_numeric'] - window) < 0.05]
            if not old_window_data.empty and len(old_window_data) > 1:
                old_weighted_std_scores.append(old_window_data['weighted_f1'].std())
            else:
                old_weighted_std_scores.append(0.01)
        
        ax2.fill_between(common_windows, 
                        np.array(old_weighted_f1_scores) - np.array(old_weighted_std_scores),
                        np.array(old_weighted_f1_scores) + np.array(old_weighted_std_scores),
                        alpha=0.3, color='#FF8C94')
    
    ax2.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Weighted F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Old Method - Weighted F1 (Multiclass)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Old Evaluation Method - Macro F1 (Multiclass)
    ax3 = axes[0, 2]
    if old_macro_f1_scores and any(not np.isnan(score) for score in old_macro_f1_scores):
        ax3.plot(common_windows, old_macro_f1_scores, '^-', color='#FFB3BA', 
                linewidth=3, markersize=8, label='eval_IoU_word_eval_sep (Macro F1)')
        
        # Add confidence interval
        old_macro_std_scores = []
        for window in common_windows:
            old_window_data = old_data[abs(old_data['window_numeric'] - window) < 0.05]
            if not old_window_data.empty and len(old_window_data) > 1:
                old_macro_std_scores.append(old_window_data['macro_f1'].std())
            else:
                old_macro_std_scores.append(0.01)
        
        ax3.fill_between(common_windows, 
                        np.array(old_macro_f1_scores) - np.array(old_macro_std_scores),
                        np.array(old_macro_f1_scores) + np.array(old_macro_std_scores),
                        alpha=0.3, color='#FFB3BA')
    
    ax3.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Macro F1 Score', fontsize=12, fontweight='bold')
    ax3.set_title('Old Method - Macro F1 (Multiclass)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 2: Word IoU Binary F1
    ax2 = axes[0, 1]
    if word_iou_binary_f1_scores and any(not np.isnan(score) for score in word_iou_binary_f1_scores):
        ax2.plot(common_windows, word_iou_binary_f1_scores, 's-', color='#4ECDC4',
                linewidth=3, markersize=8, label='word_iou_eval Binary F1 (IoU@0.3)')
        
        # Add confidence interval
        word_iou_binary_std_scores = []
        for window in common_windows:
            window_data = word_iou_filtered[abs(word_iou_filtered['window'] - window) < 0.05]
            if not window_data.empty and len(window_data) > 1:
                word_iou_binary_std_scores.append(window_data['binary_f1'].std())
            else:
                word_iou_binary_std_scores.append(0.01)
        
        ax2.fill_between(common_windows,
                        np.array(word_iou_binary_f1_scores) - np.array(word_iou_binary_std_scores),
                        np.array(word_iou_binary_f1_scores) + np.array(word_iou_binary_std_scores),
                        alpha=0.3, color='#4ECDC4')
    
    ax2.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Binary F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Word IoU Eval - Binary F1 (IoU@0.3)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Word IoU Multiclass F1
    ax3 = axes[1, 0]
    if word_iou_multiclass_f1_scores and any(not np.isnan(score) for score in word_iou_multiclass_f1_scores):
        ax3.plot(common_windows, word_iou_multiclass_f1_scores, '^-', color='#45B7D1',
                linewidth=3, markersize=8, label='word_iou_eval Multiclass F1 (IoU@0.3)')
        
        # Add confidence interval
        word_iou_multiclass_std_scores = []
        for window in common_windows:
            window_data = word_iou_filtered[abs(word_iou_filtered['window'] - window) < 0.05]
            if not window_data.empty and len(window_data) > 1:
                word_iou_multiclass_std_scores.append(window_data['multiclass_f1'].std())
            else:
                word_iou_multiclass_std_scores.append(0.01)
        
        ax3.fill_between(common_windows,
                        np.array(word_iou_multiclass_f1_scores) - np.array(word_iou_multiclass_std_scores),
                        np.array(word_iou_multiclass_f1_scores) + np.array(word_iou_multiclass_std_scores),
                        alpha=0.3, color='#45B7D1')
    
    ax3.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Multiclass F1 Score', fontsize=12, fontweight='bold')
    ax3.set_title('Word IoU Eval - Multiclass F1 (IoU@0.3)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined Comparison - All Methods
    ax4 = axes[1, 1]
    
    if old_f1_scores and any(not np.isnan(score) for score in old_f1_scores):
        ax4.plot(common_windows, old_f1_scores, 'o-', color='#FF6B6B',
                linewidth=3, markersize=8, label='Old Method', alpha=0.8)
    
    if word_iou_binary_f1_scores and any(not np.isnan(score) for score in word_iou_binary_f1_scores):
        ax4.plot(common_windows, word_iou_binary_f1_scores, 's-', color='#4ECDC4',
                linewidth=3, markersize=8, label='Word IoU Binary F1', alpha=0.8)
    
    if word_iou_multiclass_f1_scores and any(not np.isnan(score) for score in word_iou_multiclass_f1_scores):
        ax4.plot(common_windows, word_iou_multiclass_f1_scores, '^-', color='#45B7D1',
                linewidth=3, markersize=8, label='Word IoU Multiclass F1', alpha=0.8)
    
    ax4.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax4.set_title('F1 Score Comparison: Old vs Word IoU (Binary & Multiclass)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("old_vs_word_iou_binary_multiclass_comparison")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / "old_vs_word_iou_binary_multiclass_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and display correlations
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS - WINDOW SIZE EFFECTS")
    print("="*80)
    
    # Old method correlation
    if old_f1_scores and len(old_f1_scores) > 1:
        valid_old = [(w, s) for w, s in zip(common_windows, old_f1_scores) if not np.isnan(s)]
        if len(valid_old) > 1:
            windows_old, scores_old = zip(*valid_old)
            corr_old, p_old = stats.pearsonr(windows_old, scores_old)
            print(f"Old Method (eval_IoU_word_eval_sep): {corr_old:+.3f} (p={p_old:.2e})")
    
    # Word IoU Binary F1 correlation
    if word_iou_binary_f1_scores and len(word_iou_binary_f1_scores) > 1:
        valid_binary = [(w, s) for w, s in zip(common_windows, word_iou_binary_f1_scores) if not np.isnan(s)]
        if len(valid_binary) > 1:
            windows_binary, scores_binary = zip(*valid_binary)
            corr_binary, p_binary = stats.pearsonr(windows_binary, scores_binary)
            print(f"Word IoU Binary F1: {corr_binary:+.3f} (p={p_binary:.2e})")
    
    # Word IoU Multiclass F1 correlation
    if word_iou_multiclass_f1_scores and len(word_iou_multiclass_f1_scores) > 1:
        valid_multiclass = [(w, s) for w, s in zip(common_windows, word_iou_multiclass_f1_scores) if not np.isnan(s)]
        if len(valid_multiclass) > 1:
            windows_multiclass, scores_multiclass = zip(*valid_multiclass)
            corr_multiclass, p_multiclass = stats.pearsonr(windows_multiclass, scores_multiclass)
            print(f"Word IoU Multiclass F1: {corr_multiclass:+.3f} (p={p_multiclass:.2e})")
    
    print("="*80)

def generate_detailed_performance_summary(old_data, word_iou_data):
    """Generate detailed performance summary for binary and multiclass metrics"""
    
    output_dir = Path("old_vs_word_iou_binary_multiclass_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # Use IoU threshold 0.3 for comparison
    iou_threshold = 0.3
    word_iou_filtered = word_iou_data[word_iou_data['iou_threshold'] == iou_threshold]
    
    summary_data = []
    
    # Old method summary
    if old_data is not None:
        summary_data.append({
            'Method': 'eval_IoU_word_eval_sep (Old)',
            'Type': 'Traditional',
            'Metric': 'F1 Score',
            'Best Score': old_data['f1_score'].max(),
            'Mean Score': old_data['f1_score'].mean(),
            'Std Score': old_data['f1_score'].std(),
            'Best Window': old_data.loc[old_data['f1_score'].idxmax(), 'window_numeric'],
            'Configurations': len(old_data)
        })
    
    # Word IoU Binary F1 summary
    if word_iou_filtered is not None and not word_iou_filtered.empty:
        summary_data.append({
            'Method': 'word_iou_eval (New)',
            'Type': 'Word+IoU-based',
            'Metric': 'Binary F1 (IoU@0.3)',
            'Best Score': word_iou_filtered['binary_f1'].max(),
            'Mean Score': word_iou_filtered['binary_f1'].mean(),
            'Std Score': word_iou_filtered['binary_f1'].std(),
            'Best Window': word_iou_filtered.loc[word_iou_filtered['binary_f1'].idxmax(), 'window'],
            'Configurations': len(word_iou_filtered)
        })
        
        # Word IoU Multiclass F1 summary
        summary_data.append({
            'Method': 'word_iou_eval (New)',
            'Type': 'Word+IoU-based',
            'Metric': 'Multiclass F1 (IoU@0.3)',
            'Best Score': word_iou_filtered['multiclass_f1'].max(),
            'Mean Score': word_iou_filtered['multiclass_f1'].mean(),
            'Std Score': word_iou_filtered['multiclass_f1'].std(),
            'Best Window': word_iou_filtered.loc[word_iou_filtered['multiclass_f1'].idxmax(), 'window'],
            'Configurations': len(word_iou_filtered)
        })
    
    # Create and save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'old_vs_word_iou_binary_multiclass_summary.csv', index=False)
        
        print("\n" + "="*90)
        print("OLD vs WORD IoU EVAL (BINARY & MULTICLASS) PERFORMANCE COMPARISON")
        print("="*90)
        print(summary_df.round(4).to_string(index=False))
        print("="*90)
        
        # Calculate improvements
        if len(summary_data) >= 2:
            old_mean = summary_data[0]['Mean Score']
            binary_mean = summary_data[1]['Mean Score']
            multiclass_mean = summary_data[2]['Mean Score'] if len(summary_data) > 2 else None
            
            binary_improvement = ((binary_mean - old_mean) / old_mean) * 100
            binary_factor = binary_mean / old_mean
            
            print(f"\nBINARY F1 PERFORMANCE IMPROVEMENT:")
            print(f"  Improvement: {binary_improvement:+.1f}%")
            print(f"  Performance Factor: {binary_factor:.2f}x")
            
            if multiclass_mean is not None:
                multiclass_improvement = ((multiclass_mean - old_mean) / old_mean) * 100
                multiclass_factor = multiclass_mean / old_mean
                
                print(f"\nMULTICLASS F1 PERFORMANCE IMPROVEMENT:")
                print(f"  Improvement: {multiclass_improvement:+.1f}%")
                print(f"  Performance Factor: {multiclass_factor:.2f}x")
            
            print("="*90)

def main():
    """Main execution function"""
    print("="*80)
    print("OLD vs WORD IoU EVAL COMPARISON (BINARY & MULTICLASS)")
    print("="*80)
    
    # Load data
    print("📊 Loading evaluation data...")
    old_data = load_old_evaluation_data()
    word_iou_data = load_word_iou_eval_detailed_data()
    
    if old_data is None or word_iou_data is None:
        print("❌ Failed to load required evaluation data!")
        return
    
    # Create comparison plots
    print("\n🎨 Creating binary & multiclass comparison plots...")
    create_binary_multiclass_comparison_plots(old_data, word_iou_data)
    
    # Generate performance summary
    print("\n📋 Generating detailed performance summary...")
    generate_detailed_performance_summary(old_data, word_iou_data)
    
    print("\n✅ Old vs Word IoU Eval (Binary & Multiclass) comparison completed!")
    print("📁 Output saved to: old_vs_word_iou_binary_multiclass_comparison/")

if __name__ == "__main__":
    main()

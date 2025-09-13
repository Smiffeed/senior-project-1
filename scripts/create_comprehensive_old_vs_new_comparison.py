#!/usr/bin/env python3
"""
Comprehensive Comparison: Old vs New Evaluation Methods
Compare eval_IoU_word_eval_sep (old) with current evaluation methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def load_old_evaluation_data():
    """Load the old evaluation data from CSV"""
    old_csv_path = Path("comparison_graphs/eval_IoU_word_eval_sep_detailed.csv")
    
    if not old_csv_path.exists():
        print("❌ Old evaluation CSV not found. Please run create_old_eval_comparison_graphs.py first.")
        return pd.DataFrame()
    
    df_old = pd.read_csv(old_csv_path)
    print(f"✅ Loaded old evaluation data: {len(df_old)} configurations")
    return df_old

def load_new_evaluation_data():
    """Load the new evaluation data from comprehensive analysis"""
    new_csv_path = Path("fixed_smart_parallel_results/fixed_smart_parallel_summary.csv")
    
    if not new_csv_path.exists():
        print("❌ New evaluation CSV not found.")
        return pd.DataFrame()
    
    df_new = pd.read_csv(new_csv_path)
    
    # Add window_numeric column for comparison
    df_new['window_numeric'] = df_new['window'].str.extract(r'window_(\d+\.?\d*)s').astype(float)
    df_new['stride_numeric'] = df_new['stride'].str.extract(r'stride_(\d+\.?\d*)s').astype(float)
    
    print(f"✅ Loaded new evaluation data: {len(df_new)} configurations")
    print(f"📊 Available metrics: {[col for col in df_new.columns if col not in ['eval_type', 'window', 'stride', 'success', 'duration_seconds', 'worker_id', 'window_numeric', 'stride_numeric']]}")
    return df_new

def create_comprehensive_comparison():
    """Create comprehensive comparison between old and new evaluation methods"""
    
    print("=== COMPREHENSIVE OLD vs NEW EVALUATION COMPARISON ===\n")
    
    # Load data
    df_old = load_old_evaluation_data()
    df_new = load_new_evaluation_data()
    
    if df_old.empty or df_new.empty:
        print("❌ Cannot create comparison - missing data")
        return
    
    # Create output directory
    output_dir = Path("comprehensive_old_vs_new_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # Process old data - group by window size
    df_old_stats = df_old.groupby('window_numeric').agg({
        'f1_score': ['mean', 'std', 'max'],
        'accuracy': ['mean', 'std', 'max'],
        'precision': ['mean', 'std', 'max'],
        'recall': ['mean', 'std', 'max'],
        'balanced_accuracy': ['mean', 'std', 'max']
    }).reset_index()
    
    # Flatten column names
    df_old_stats.columns = [f"old_{col[0]}_{col[1]}" if col[1] else col[0] for col in df_old_stats.columns]
    df_old_stats = df_old_stats.rename(columns={'window_numeric_': 'window_numeric'})
    
    # Process new data - group by window size for different metrics
    df_new_stats = df_new.groupby('window_numeric').agg({
        'word_f1': ['mean', 'std', 'max'],
        'window_f1': ['mean', 'std', 'max'],
        'combined_iou': ['mean', 'std', 'max']
    }).reset_index()
    
    # Flatten column names
    df_new_stats.columns = [f"new_{col[0]}_{col[1]}" if col[1] else col[0] for col in df_new_stats.columns]
    df_new_stats = df_new_stats.rename(columns={'window_numeric_': 'window_numeric'})
    
    # Create comparison plots
    create_f1_comparison_plot(df_old_stats, df_new_stats, output_dir)
    create_accuracy_comparison_plot(df_old_stats, df_new_stats, output_dir)
    create_performance_summary_table(df_old, df_new, output_dir)
    
    print(f"\n🎉 Comprehensive comparison created successfully!")
    print(f"📁 Output directory: {output_dir}")

def create_f1_comparison_plot(df_old_stats, df_new_stats, output_dir):
    """Create F1 score comparison plot"""
    
    plt.figure(figsize=(14, 10))
    
    # Plot old evaluation F1 scores
    if 'old_f1_score_mean' in df_old_stats.columns:
        plt.subplot(2, 2, 1)
        plt.plot(df_old_stats['window_numeric'], df_old_stats['old_f1_score_mean'], 
                marker='o', linewidth=3, markersize=8, color='#e74c3c', 
                label='eval_IoU_word_eval_sep (Old)', alpha=0.8)
        
        if 'old_f1_score_std' in df_old_stats.columns:
            plt.fill_between(df_old_stats['window_numeric'],
                           df_old_stats['old_f1_score_mean'] - df_old_stats['old_f1_score_std'],
                           df_old_stats['old_f1_score_mean'] + df_old_stats['old_f1_score_std'],
                           alpha=0.2, color='#e74c3c')
        
        plt.xlabel('Window Size (seconds)', fontweight='bold')
        plt.ylabel('F1 Score', fontweight='bold')
        plt.title('Old Evaluation Method - F1 Score', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Plot new evaluation Word F1 scores
    if not df_new_stats.empty and 'new_word_f1_mean' in df_new_stats.columns:
        plt.subplot(2, 2, 2)
        plt.plot(df_new_stats['window_numeric'], df_new_stats['new_word_f1_mean'], 
                marker='s', linewidth=3, markersize=8, color='#2ecc71', 
                label='word_eval (New)', alpha=0.8)
        
        if 'new_word_f1_std' in df_new_stats.columns:
            plt.fill_between(df_new_stats['window_numeric'],
                           df_new_stats['new_word_f1_mean'] - df_new_stats['new_word_f1_std'],
                           df_new_stats['new_word_f1_mean'] + df_new_stats['new_word_f1_std'],
                           alpha=0.2, color='#2ecc71')
        
        plt.xlabel('Window Size (seconds)', fontweight='bold')
        plt.ylabel('Word F1 Score', fontweight='bold')
        plt.title('New Evaluation Method - Word F1', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # Combined comparison
    plt.subplot(2, 1, 2)
    
    if 'old_f1_score_mean' in df_old_stats.columns:
        plt.plot(df_old_stats['window_numeric'], df_old_stats['old_f1_score_mean'], 
                marker='o', linewidth=3, markersize=8, color='#e74c3c', 
                label='eval_IoU_word_eval_sep (Old)', alpha=0.8)
    
    if not df_new_stats.empty and 'new_word_f1_mean' in df_new_stats.columns:
        # Only plot where both have data
        common_windows = set(df_old_stats['window_numeric']).intersection(set(df_new_stats['window_numeric']))
        if common_windows:
            old_subset = df_old_stats[df_old_stats['window_numeric'].isin(common_windows)]
            new_subset = df_new_stats[df_new_stats['window_numeric'].isin(common_windows)]
            
            plt.plot(new_subset['window_numeric'], new_subset['new_word_f1_mean'], 
                    marker='s', linewidth=3, markersize=8, color='#2ecc71', 
                    label='word_eval (New)', alpha=0.8)
    
    plt.xlabel('Window Size (seconds)', fontweight='bold')
    plt.ylabel('F1 Score', fontweight='bold')
    plt.title('F1 Score Comparison: Old vs New Methods', fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "f1_score_old_vs_new_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("💾 Saved: f1_score_old_vs_new_comparison.png")

def create_accuracy_comparison_plot(df_old_stats, df_new_stats, output_dir):
    """Create accuracy comparison plot"""
    
    plt.figure(figsize=(12, 8))
    
    # Plot old evaluation accuracy
    if 'old_accuracy_mean' in df_old_stats.columns:
        plt.plot(df_old_stats['window_numeric'], df_old_stats['old_accuracy_mean'], 
                marker='o', linewidth=3, markersize=8, color='#e74c3c', 
                label='eval_IoU_word_eval_sep (Old)', alpha=0.8)
        
        if 'old_accuracy_std' in df_old_stats.columns:
            plt.fill_between(df_old_stats['window_numeric'],
                           df_old_stats['old_accuracy_mean'] - df_old_stats['old_accuracy_std'],
                           df_old_stats['old_accuracy_mean'] + df_old_stats['old_accuracy_std'],
                           alpha=0.2, color='#e74c3c')
    
    plt.xlabel('Window Size (seconds)', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.title('Accuracy Comparison: Old vs New Methods', fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_old_vs_new_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("💾 Saved: accuracy_old_vs_new_comparison.png")

def create_performance_summary_table(df_old, df_new, output_dir):
    """Create performance summary comparison table"""
    
    summary_data = []
    
    # Old evaluation summary
    if not df_old.empty:
        old_summary = {
            'Method': 'eval_IoU_word_eval_sep (Old)',
            'Total Configs': len(df_old),
            'F1 Score Mean': df_old['f1_score'].mean(),
            'F1 Score Max': df_old['f1_score'].max(),
            'F1 Score Std': df_old['f1_score'].std(),
            'Accuracy Mean': df_old['accuracy'].mean(),
            'Accuracy Max': df_old['accuracy'].max(),
            'Precision Mean': df_old['precision'].mean(),
            'Recall Mean': df_old['recall'].mean(),
            'Best Window (F1)': df_old.loc[df_old['f1_score'].idxmax(), 'window'],
            'Best Stride (F1)': df_old.loc[df_old['f1_score'].idxmax(), 'stride']
        }
        summary_data.append(old_summary)
    
    # New evaluation summaries
    if not df_new.empty:
        # Word F1 evaluation
        if 'word_f1' in df_new.columns:
            best_idx = df_new['word_f1'].idxmax()
            word_f1_summary = {
                'Method': 'word_eval (New)',
                'Total Configs': len(df_new),
                'F1 Score Mean': df_new['word_f1'].mean(),
                'F1 Score Max': df_new['word_f1'].max(),
                'F1 Score Std': df_new['word_f1'].std(),
                'Accuracy Mean': np.nan,  # Not available in this dataset
                'Accuracy Max': np.nan,
                'Precision Mean': np.nan,
                'Recall Mean': np.nan,
                'Best Window (F1)': df_new.loc[best_idx, 'window'],
                'Best Stride (F1)': df_new.loc[best_idx, 'stride']
            }
            summary_data.append(word_f1_summary)
        
        # Window F1 evaluation  
        if 'window_f1' in df_new.columns:
            best_idx = df_new['window_f1'].idxmax()
            window_f1_summary = {
                'Method': 'window_eval (New)',
                'Total Configs': len(df_new),
                'F1 Score Mean': df_new['window_f1'].mean(),
                'F1 Score Max': df_new['window_f1'].max(),
                'F1 Score Std': df_new['window_f1'].std(),
                'Accuracy Mean': np.nan,
                'Accuracy Max': np.nan,
                'Precision Mean': np.nan,
                'Recall Mean': np.nan,
                'Best Window (F1)': df_new.loc[best_idx, 'window'],
                'Best Stride (F1)': df_new.loc[best_idx, 'stride']
            }
            summary_data.append(window_f1_summary)
            
        # Combined IoU evaluation
        if 'combined_iou' in df_new.columns:
            best_idx = df_new['combined_iou'].idxmax()
            iou_summary = {
                'Method': 'combined_iou (New)',
                'Total Configs': len(df_new),
                'F1 Score Mean': df_new['combined_iou'].mean(),  # Using IoU as F1 equivalent
                'F1 Score Max': df_new['combined_iou'].max(),
                'F1 Score Std': df_new['combined_iou'].std(),
                'Accuracy Mean': np.nan,
                'Accuracy Max': np.nan,
                'Precision Mean': np.nan,
                'Recall Mean': np.nan,
                'Best Window (F1)': df_new.loc[best_idx, 'window'],
                'Best Stride (F1)': df_new.loc[best_idx, 'stride']
            }
            summary_data.append(iou_summary)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.round(4)
    
    # Save to CSV
    summary_df.to_csv(output_dir / "performance_comparison_summary.csv", index=False)
    print("💾 Saved: performance_comparison_summary.csv")
    
    # Print summary
    print("\n📊 PERFORMANCE COMPARISON SUMMARY:")
    print("="*80)
    for _, row in summary_df.iterrows():
        print(f"\n{row['Method']}:")
        print(f"  Total Configs: {row['Total Configs']}")
        print(f"  F1 Score: {row['F1 Score Mean']:.4f} ± {row['F1 Score Std']:.4f} (max: {row['F1 Score Max']:.4f})")
        if not pd.isna(row['Accuracy Mean']):
            print(f"  Accuracy: {row['Accuracy Mean']:.4f} (max: {row['Accuracy Max']:.4f})")
        print(f"  Best Config: {row['Best Window (F1)']} + {row['Best Stride (F1)']}")
    
    return summary_df

if __name__ == "__main__":
    create_comprehensive_comparison()

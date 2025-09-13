#!/usr/bin/env python3
"""
Create Side-by-Side Comparison Graphs
Compare eval_IoU_word_eval_sep vs fixed_smart_parallel_results F1 scores
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import os

def load_fixed_smart_results():
    """Load results from fixed_smart_parallel_results correlation analysis"""
    
    # Try to load from correlation analysis
    data_path = Path("fixed_smart_parallel_results/correlation_analysis/complete_analysis_data.csv")
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        # Filter for word_iou_eval equivalent (which would be word_eval + iou data)
        # We'll use word_f1 as the comparable metric
        df_filtered = df[df['success'] == True].copy()
        
        # Group by window size
        grouped = df_filtered.groupby('window_numeric').agg({
            'word_f1': ['mean', 'std', 'count'],
            'window_f1': ['mean', 'std', 'count'],
            'combined_iou': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped.columns]
        grouped = grouped.rename(columns={'window_numeric_': 'window_numeric'})
        
        return grouped
    else:
        print(f"❌ Could not find fixed_smart_parallel_results data at {data_path}")
        return None

def load_old_eval_results():
    """Load the old eval results we just processed"""
    
    data_path = Path("comparison_graphs/eval_IoU_word_eval_sep_summary.csv")
    
    if data_path.exists():
        return pd.read_csv(data_path)
    else:
        print(f"❌ Could not find old eval results at {data_path}")
        return None

def create_side_by_side_comparison():
    """Create side-by-side comparison graphs"""
    
    print("=== CREATING SIDE-BY-SIDE COMPARISON GRAPHS ===\n")
    
    # Load both datasets
    df_new = load_fixed_smart_results()
    df_old = load_old_eval_results()
    
    if df_new is None or df_old is None:
        print("❌ Could not load comparison data")
        return
    
    print(f"📊 New eval data shape: {df_new.shape}")
    print(f"📊 Old eval data shape: {df_old.shape}")
    
    # Create output directory
    output_dir = Path("side_by_side_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    colors_new = '#2E86AB'  # Blue for new method
    colors_old = '#C73E1D'  # Red for old method
    
    # Define comparison pairs
    comparisons = [
        ('word_f1_mean', 'weighted_f1_mean', 'F1 Score Comparison', 'F1 Score'),
        ('word_f1_mean', 'overall_accuracy_mean', 'F1 vs Overall Accuracy', 'Score'),
        ('combined_iou_mean', 'balanced_accuracy_mean', 'IoU vs Balanced Accuracy', 'Score')
    ]
    
    for new_metric, old_metric, title, ylabel in comparisons:
        if new_metric not in df_new.columns or old_metric not in df_old.columns:
            print(f"⚠️ Skipping {title} - metrics not available")
            continue
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot new method
        valid_new = df_new.dropna(subset=[new_metric])
        ax.plot(valid_new['window_numeric'], valid_new[new_metric], 
               marker='o', linewidth=3, markersize=8, 
               color=colors_new, label=f'Fixed Smart Parallel ({new_metric.replace("_mean", "")})',
               alpha=0.8)
        
        # Add error bars for new method
        new_std_col = new_metric.replace('_mean', '_std')
        if new_std_col in df_new.columns:
            ax.fill_between(valid_new['window_numeric'], 
                           valid_new[new_metric] - valid_new[new_std_col],
                           valid_new[new_metric] + valid_new[new_std_col],
                           alpha=0.2, color=colors_new)
        
        # Plot old method
        valid_old = df_old.dropna(subset=[old_metric])
        ax.plot(valid_old['window_numeric'], valid_old[old_metric], 
               marker='s', linewidth=3, markersize=8, 
               color=colors_old, label=f'eval_IoU_word_eval_sep ({old_metric.replace("_mean", "")})',
               alpha=0.8)
        
        # Add error bars for old method
        old_std_col = old_metric.replace('_mean', '_std')
        if old_std_col in df_old.columns:
            ax.fill_between(valid_old['window_numeric'], 
                           valid_old[old_metric] - valid_old[old_std_col],
                           valid_old[old_metric] + valid_old[old_std_col],
                           alpha=0.2, color=colors_old)
        
        # Customize plot
        ax.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(f'{title}\n(Fixed Smart Parallel vs eval_IoU_word_eval_sep)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        
        # Set reasonable y-limits
        all_values = []
        if len(valid_new) > 0:
            all_values.extend(valid_new[new_metric].dropna())
        if len(valid_old) > 0:
            all_values.extend(valid_old[old_metric].dropna())
        
        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            margin = (y_max - y_min) * 0.1
            ax.set_ylim(max(0, y_min - margin), min(1, y_max + margin))
        
        # Save plot
        filename = output_dir / f"comparison_{new_metric}_{old_metric}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved: {filename}")
    
    # Create comprehensive comparison summary
    print(f"\n📊 COMPREHENSIVE COMPARISON SUMMARY:")
    print("="*80)
    
    print("🔵 FIXED SMART PARALLEL RESULTS:")
    if 'word_f1_mean' in df_new.columns:
        word_f1_values = df_new['word_f1_mean'].dropna()
        print(f"   Word F1: Mean={word_f1_values.mean():.4f}, Std={word_f1_values.std():.4f}, Range={word_f1_values.min():.4f}-{word_f1_values.max():.4f}")
    
    if 'combined_iou_mean' in df_new.columns:
        iou_values = df_new['combined_iou_mean'].dropna()
        print(f"   Combined IoU: Mean={iou_values.mean():.4f}, Std={iou_values.std():.4f}, Range={iou_values.min():.4f}-{iou_values.max():.4f}")
    
    print("\n🔴 eval_IoU_word_eval_sep RESULTS:")
    if 'weighted_f1_mean' in df_old.columns:
        weighted_f1_values = df_old['weighted_f1_mean'].dropna()
        print(f"   Weighted F1: Mean={weighted_f1_values.mean():.4f}, Std={weighted_f1_values.std():.4f}, Range={weighted_f1_values.min():.4f}-{weighted_f1_values.max():.4f}")
    
    if 'overall_accuracy_mean' in df_old.columns:
        accuracy_values = df_old['overall_accuracy_mean'].dropna()
        print(f"   Overall Accuracy: Mean={accuracy_values.mean():.4f}, Std={accuracy_values.std():.4f}, Range={accuracy_values.min():.4f}-{accuracy_values.max():.4f}")
    
    # Performance difference analysis
    print(f"\n📈 PERFORMANCE ANALYSIS:")
    if 'word_f1_mean' in df_new.columns and 'weighted_f1_mean' in df_old.columns:
        new_max = df_new['word_f1_mean'].max()
        old_max = df_old['weighted_f1_mean'].max()
        improvement = ((new_max - old_max) / old_max) * 100
        print(f"   Max F1 Score Improvement: {improvement:.2f}% ({old_max:.4f} → {new_max:.4f})")
    
    print(f"\n🎉 Side-by-side comparison complete!")
    print(f"📁 Results saved to: {output_dir}/")

if __name__ == "__main__":
    create_side_by_side_comparison()

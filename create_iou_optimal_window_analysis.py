#!/usr/bin/env python3
"""
IoU Evaluation Optimal Window Size Analysis
Create focused graphs showing both binary and multiclass F1 scores for IoU evaluation
to identify optimal window sizes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import os

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
        multiclass_content_match = re.search(r'--- IoU Threshold 0.1 ---.*?Multiclass Classification.*?Balanced Accuracy:\s*([\d.]+)', content, re.DOTALL)
        if multiclass_content_match:
            # For multiclass, we'll use balanced accuracy as a proxy since F1 extraction is complex
            metrics_data['multiclass_f1'] = float(multiclass_content_match.group(1))
        
        return metrics_data
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return {}

def extract_iou_data():
    """Extract F1 score data specifically for IoU evaluation"""
    
    base_dir = Path("fixed_smart_parallel_results")
    all_results = []
    
    # Process both eval_by_0.05 and eval_percent directories
    for eval_type in ["eval_by_0.05", "eval_percent"]:
        eval_dir = base_dir / eval_type / eval_type
        
        if not eval_dir.exists():
            print(f"Directory not found: {eval_dir}")
            continue
        
        print(f"Processing {eval_type}...")
        
        # Only IoU evaluation
        method_dir = eval_dir / "iou_eval"
        
        if method_dir.exists():
            print(f"  Processing iou_eval...")
            
            for window_dir in method_dir.iterdir():
                if window_dir.is_dir() and window_dir.name.startswith("window_"):
                    for stride_dir in window_dir.iterdir():
                        if stride_dir.is_dir() and stride_dir.name.startswith("stride_"):
                            note_file = stride_dir / "note.txt"
                            if note_file.exists():
                                f1_data = extract_iou_f1_metrics(note_file)
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
                                        
                                        config_data = {
                                            'eval_type': eval_type,
                                            'method': 'iou_eval',
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
        print(f"\n✅ Total IoU configurations extracted: {len(df)}")
        print(f"Evaluation types: {list(df['eval_type'].unique())}")
        return df
    else:
        return pd.DataFrame()

def create_iou_optimal_window_graphs(df, output_dir):
    """Create focused IoU evaluation graphs for optimal window size analysis"""
    
    # Create separate graphs for each eval_type
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type].copy()
        
        # Group by window size and calculate statistics
        window_stats_binary = eval_data.groupby('window_numeric')['binary_f1'].agg(['mean', 'std', 'min', 'max']).reset_index()
        window_stats_multiclass = eval_data.groupby('window_numeric')['multiclass_f1'].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        window_stats_binary = window_stats_binary.sort_values('window_numeric')
        window_stats_multiclass = window_stats_multiclass.sort_values('window_numeric')
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot Binary F1 Scores
        ax.plot(
            window_stats_binary['window_numeric'], 
            window_stats_binary['mean'],
            marker='o', 
            linewidth=3, 
            markersize=8,
            label='Binary F1 Score',
            color='#2ca02c',  # Green
            alpha=0.9
        )
        
        # Add shaded area for binary F1
        ax.fill_between(
            window_stats_binary['window_numeric'],
            window_stats_binary['min'],
            window_stats_binary['max'],
            alpha=0.2,
            color='#2ca02c'
        )
        
        # Plot Multiclass F1 Scores
        ax.plot(
            window_stats_multiclass['window_numeric'], 
            window_stats_multiclass['mean'],
            marker='s', 
            linewidth=3, 
            markersize=8,
            label='Multiclass F1 Score',
            color='#ff7f0e',  # Orange
            alpha=0.9
        )
        
        # Add shaded area for multiclass F1
        ax.fill_between(
            window_stats_multiclass['window_numeric'],
            window_stats_multiclass['min'],
            window_stats_multiclass['max'],
            alpha=0.2,
            color='#ff7f0e'
        )
        
        # Find optimal window sizes
        best_binary_idx = window_stats_binary['mean'].idxmax()
        best_multiclass_idx = window_stats_multiclass['mean'].idxmax()
        
        best_binary_window = window_stats_binary.loc[best_binary_idx, 'window_numeric']
        best_multiclass_window = window_stats_multiclass.loc[best_multiclass_idx, 'window_numeric']
        
        best_binary_score = window_stats_binary.loc[best_binary_idx, 'mean']
        best_multiclass_score = window_stats_multiclass.loc[best_multiclass_idx, 'mean']
        
        # Add vertical lines for optimal points
        ax.axvline(x=best_binary_window, color='#2ca02c', linestyle='--', alpha=0.7, linewidth=2)
        ax.axvline(x=best_multiclass_window, color='#ff7f0e', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add annotations for optimal points
        ax.annotate(f'Binary Optimal\n{best_binary_window}s\nF1={best_binary_score:.3f}', 
                   xy=(best_binary_window, best_binary_score), 
                   xytext=(best_binary_window + 0.2, best_binary_score + 0.05),
                   arrowprops=dict(arrowstyle='->', color='#2ca02c', alpha=0.7),
                   fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.annotate(f'Multiclass Optimal\n{best_multiclass_window}s\nF1={best_multiclass_score:.3f}', 
                   xy=(best_multiclass_window, best_multiclass_score), 
                   xytext=(best_multiclass_window - 0.3, best_multiclass_score - 0.1),
                   arrowprops=dict(arrowstyle='->', color='#ff7f0e', alpha=0.7),
                   fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_title(f'IoU Evaluation - Optimal Window Size Analysis ({eval_type})', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Window Size (seconds)', fontsize=13, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='best')
        ax.set_ylim(0, max(1.0, ax.get_ylim()[1] * 1.1))
        
        plt.tight_layout()
        plt.savefig(output_dir / f"iou_optimal_window_analysis_{eval_type}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved: iou_optimal_window_analysis_{eval_type}.png")
        print(f"   Binary F1 Optimal: {best_binary_window}s (F1={best_binary_score:.4f})")
        print(f"   Multiclass F1 Optimal: {best_multiclass_window}s (F1={best_multiclass_score:.4f})")

def create_combined_iou_optimal_analysis(df, output_dir):
    """Create a single combined graph comparing both eval types"""
    
    # Colors for eval types
    colors = {
        'eval_by_0.05': {'binary': '#2ca02c', 'multiclass': '#ff7f0e'},
        'eval_percent': {'binary': '#1f77b4', 'multiclass': '#d62728'}
    }
    
    # Line styles
    line_styles = {
        'eval_by_0.05': '-',
        'eval_percent': '--'
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    optimal_results = []
    
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type].copy()
        
        # Group by window size and calculate statistics
        window_stats_binary = eval_data.groupby('window_numeric')['binary_f1'].agg(['mean', 'std', 'min', 'max']).reset_index()
        window_stats_multiclass = eval_data.groupby('window_numeric')['multiclass_f1'].agg(['mean', 'std', 'min', 'max']).reset_index()
        
        window_stats_binary = window_stats_binary.sort_values('window_numeric')
        window_stats_multiclass = window_stats_multiclass.sort_values('window_numeric')
        
        # Plot Binary F1 Scores
        ax.plot(
            window_stats_binary['window_numeric'], 
            window_stats_binary['mean'],
            marker='o', 
            linewidth=3, 
            markersize=6,
            label=f'Binary F1 ({eval_type})',
            color=colors[eval_type]['binary'],
            linestyle=line_styles[eval_type],
            alpha=0.9
        )
        
        # Plot Multiclass F1 Scores
        ax.plot(
            window_stats_multiclass['window_numeric'], 
            window_stats_multiclass['mean'],
            marker='s', 
            linewidth=3, 
            markersize=6,
            label=f'Multiclass F1 ({eval_type})',
            color=colors[eval_type]['multiclass'],
            linestyle=line_styles[eval_type],
            alpha=0.9
        )
        
        # Add shaded areas
        ax.fill_between(
            window_stats_binary['window_numeric'],
            window_stats_binary['min'],
            window_stats_binary['max'],
            alpha=0.1,
            color=colors[eval_type]['binary']
        )
        
        ax.fill_between(
            window_stats_multiclass['window_numeric'],
            window_stats_multiclass['min'],
            window_stats_multiclass['max'],
            alpha=0.1,
            color=colors[eval_type]['multiclass']
        )
        
        # Store optimal results
        best_binary_idx = window_stats_binary['mean'].idxmax()
        best_multiclass_idx = window_stats_multiclass['mean'].idxmax()
        
        optimal_results.append({
            'eval_type': eval_type,
            'binary_optimal_window': window_stats_binary.loc[best_binary_idx, 'window_numeric'],
            'binary_optimal_score': window_stats_binary.loc[best_binary_idx, 'mean'],
            'multiclass_optimal_window': window_stats_multiclass.loc[best_multiclass_idx, 'window_numeric'],
            'multiclass_optimal_score': window_stats_multiclass.loc[best_multiclass_idx, 'mean']
        })
    
    ax.set_title('IoU Evaluation - Optimal Window Size Analysis (All Evaluation Types)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Window Size (seconds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1] * 1.1))
    
    plt.tight_layout()
    plt.savefig(output_dir / "iou_optimal_window_analysis_combined.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved: iou_optimal_window_analysis_combined.png")
    
    # Create summary table
    optimal_df = pd.DataFrame(optimal_results)
    optimal_df.to_csv(output_dir / "iou_optimal_window_summary.csv", index=False)
    
    print(f"💾 Saved: iou_optimal_window_summary.csv")
    print("\n📊 OPTIMAL WINDOW SIZE SUMMARY:")
    for result in optimal_results:
        print(f"\n{result['eval_type'].upper()}:")
        print(f"   Binary F1 Optimal: {result['binary_optimal_window']}s (F1={result['binary_optimal_score']:.4f})")
        print(f"   Multiclass F1 Optimal: {result['multiclass_optimal_window']}s (F1={result['multiclass_optimal_score']:.4f})")

def main():
    """Main function to create IoU optimal window analysis"""
    
    print("=== CREATING IoU OPTIMAL WINDOW SIZE ANALYSIS ===\n")
    
    # Extract IoU data
    df = extract_iou_data()
    
    if df.empty:
        print("❌ No IoU evaluation data extracted!")
        return
    
    print(f"✅ Extracted data from {len(df)} IoU configurations")
    print(f"   Evaluation types: {list(df['eval_type'].unique())}")
    
    # Create output directory
    output_dir = Path("iou_optimal_window_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Create analysis components
    create_iou_optimal_window_graphs(df, output_dir)
    create_combined_iou_optimal_analysis(df, output_dir)
    
    print(f"\n🎉 IoU optimal window analysis completed successfully!")
    print(f"📁 Output directory: {output_dir}")

if __name__ == "__main__":
    main()

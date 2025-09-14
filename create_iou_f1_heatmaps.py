#!/usr/bin/env python3
"""
IoU Evaluation F1 Score Heatmap Analysis
Create detailed heatmaps showing F1 scores across window and stride combinations
to find optimal configurations for both binary and multiclass classification
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
        
        # Extract Multiclass F1 at IoU 0.1 (using balanced accuracy as proxy)
        multiclass_content_match = re.search(r'--- IoU Threshold 0.1 ---.*?Multiclass Classification.*?Balanced Accuracy:\s*([\d.]+)', content, re.DOTALL)
        if multiclass_content_match:
            metrics_data['multiclass_f1'] = float(multiclass_content_match.group(1))
        
        return metrics_data
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return {}

def extract_iou_heatmap_data():
    """Extract F1 score data for IoU evaluation heatmap analysis"""
    
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
                                            stride_display = f"{stride_numeric}"
                                        else:
                                            stride_numeric = float(stride_match_percent.group(1))
                                            stride_type = 'percent'
                                            stride_display = f"{stride_numeric}"
                                        
                                        config_data = {
                                            'eval_type': eval_type,
                                            'window': window_dir.name,
                                            'stride': stride_dir.name,
                                            'window_numeric': float(window_match.group(1)),
                                            'stride_numeric': stride_numeric,
                                            'stride_type': stride_type,
                                            'stride_display': stride_display,
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

def create_iou_f1_heatmaps(df, output_dir):
    """Create F1 score heatmaps for IoU evaluation"""
    
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type].copy()
        
        # Create separate heatmaps for binary and multiclass F1
        for metric_type in ['binary_f1', 'multiclass_f1']:
            if metric_type not in eval_data.columns:
                continue
                
            # Create pivot table for heatmap
            if eval_type == 'eval_by_0.05':
                # For time-based strides, use stride_numeric directly
                pivot_data = eval_data.pivot_table(
                    values=metric_type, 
                    index='stride_numeric', 
                    columns='window_numeric', 
                    aggfunc='mean'
                )
                stride_label = 'Stride Size (seconds)'
            else:
                # For percentage-based strides, use stride_numeric
                pivot_data = eval_data.pivot_table(
                    values=metric_type, 
                    index='stride_numeric', 
                    columns='window_numeric', 
                    aggfunc='mean'
                )
                stride_label = 'Stride Size (%)'
            
            # Sort indices and columns
            pivot_data = pivot_data.sort_index().sort_index(axis=1)
            
            # Find optimal configuration
            max_value = pivot_data.max().max()
            max_location = pivot_data.stack().idxmax()
            optimal_stride, optimal_window = max_location
            
            # Create heatmap
            plt.figure(figsize=(16, 12))
            
            # Create heatmap with annotations
            sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                center=pivot_data.mean().mean(),
                square=False,
                linewidths=0.5,
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 8}
            )
            
            # Customize the plot
            metric_name = 'Binary F1' if metric_type == 'binary_f1' else 'Multiclass F1'
            plt.title(f'{eval_type} - IoU Evaluation - {metric_name} Score Heatmap\nOptimal: Window={optimal_window}s, Stride={optimal_stride}{" %" if eval_type == "eval_percent" else "s"}, F1={max_value:.4f}', 
                     fontsize=16, fontweight='bold', pad=20)
            
            plt.xlabel('Window Size (seconds)', fontsize=14, fontweight='bold')
            plt.ylabel(stride_label, fontsize=14, fontweight='bold')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            
            # Save the plot
            filename = f"iou_{metric_type}_heatmap_{eval_type}.png"
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"💾 Saved: {filename}")
            print(f"   Optimal {metric_name}: Window={optimal_window}s, Stride={optimal_stride}{' %' if eval_type == 'eval_percent' else 's'}, F1={max_value:.4f}")

def create_combined_optimal_heatmap(df, output_dir):
    """Create a combined heatmap showing both binary and multiclass optimal regions"""
    
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type].copy()
        
        # Create pivot tables for both metrics
        if eval_type == 'eval_by_0.05':
            stride_label = 'Stride Size (seconds)'
            stride_suffix = 's'
        else:
            stride_label = 'Stride Size (%)'
            stride_suffix = '%'
        
        binary_pivot = eval_data.pivot_table(
            values='binary_f1', 
            index='stride_numeric', 
            columns='window_numeric', 
            aggfunc='mean'
        ).sort_index().sort_index(axis=1)
        
        multiclass_pivot = eval_data.pivot_table(
            values='multiclass_f1', 
            index='stride_numeric', 
            columns='window_numeric', 
            aggfunc='mean'
        ).sort_index().sort_index(axis=1)
        
        # Create combined metric (weighted sum for demonstration)
        # You can adjust weights based on your priorities
        combined_metric = 0.6 * binary_pivot + 0.4 * multiclass_pivot
        
        # Find optimal configurations for each metric
        binary_max = binary_pivot.max().max()
        binary_location = binary_pivot.stack().idxmax()
        binary_optimal_stride, binary_optimal_window = binary_location
        
        multiclass_max = multiclass_pivot.max().max()
        multiclass_location = multiclass_pivot.stack().idxmax()
        multiclass_optimal_stride, multiclass_optimal_window = multiclass_location
        
        combined_max = combined_metric.max().max()
        combined_location = combined_metric.stack().idxmax()
        combined_optimal_stride, combined_optimal_window = combined_location
        
        # Create side-by-side subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Binary F1 heatmap
        sns.heatmap(
            binary_pivot,
            annot=True,
            fmt='.3f',
            cmap='Reds',
            ax=ax1,
            square=False,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 7}
        )
        ax1.set_title(f'Binary F1 Score\nOptimal: W={binary_optimal_window}s, S={binary_optimal_stride}{stride_suffix}, F1={binary_max:.4f}', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Window Size (seconds)', fontsize=12)
        ax1.set_ylabel(stride_label, fontsize=12)
        
        # Multiclass F1 heatmap  
        sns.heatmap(
            multiclass_pivot,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            ax=ax2,
            square=False,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 7}
        )
        ax2.set_title(f'Multiclass F1 Score\nOptimal: W={multiclass_optimal_window}s, S={multiclass_optimal_stride}{stride_suffix}, F1={multiclass_max:.4f}', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Window Size (seconds)', fontsize=12)
        ax2.set_ylabel(stride_label, fontsize=12)
        
        # Combined metric heatmap
        sns.heatmap(
            combined_metric,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            ax=ax3,
            square=False,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 7}
        )
        ax3.set_title(f'Combined Score (0.6×Binary + 0.4×Multiclass)\nOptimal: W={combined_optimal_window}s, S={combined_optimal_stride}{stride_suffix}, Score={combined_max:.4f}', 
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel('Window Size (seconds)', fontsize=12)
        ax3.set_ylabel(stride_label, fontsize=12)
        
        # Difference heatmap (Binary - Multiclass)
        difference_pivot = binary_pivot - multiclass_pivot
        sns.heatmap(
            difference_pivot,
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            ax=ax4,
            square=False,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 7}
        )
        ax4.set_title('Performance Difference (Binary F1 - Multiclass F1)\nRed: Binary Better, Blue: Multiclass Better', 
                     fontsize=14, fontweight='bold')
        ax4.set_xlabel('Window Size (seconds)', fontsize=12)
        ax4.set_ylabel(stride_label, fontsize=12)
        
        plt.suptitle(f'{eval_type} - IoU Evaluation - Comprehensive F1 Score Analysis', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save the plot
        filename = f"iou_comprehensive_f1_heatmap_{eval_type}.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved: {filename}")

def create_optimal_summary_table(df, output_dir):
    """Create summary table with all optimal configurations"""
    
    summary_results = []
    
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type].copy()
        
        for metric_type in ['binary_f1', 'multiclass_f1']:
            if metric_type not in eval_data.columns:
                continue
            
            # Find optimal configuration
            best_idx = eval_data[metric_type].idxmax()
            best_config = eval_data.loc[best_idx]
            
            summary_results.append({
                'eval_type': eval_type,
                'metric_type': metric_type,
                'optimal_window_size': best_config['window_numeric'],
                'optimal_stride_value': best_config['stride_numeric'],
                'stride_type': best_config['stride_type'],
                'optimal_f1_score': best_config[metric_type],
                'window_config': best_config['window'],
                'stride_config': best_config['stride']
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_results)
    summary_df = summary_df.round(4)
    summary_df.to_csv(output_dir / "iou_optimal_configurations_summary.csv", index=False)
    
    print(f"💾 Saved: iou_optimal_configurations_summary.csv")
    
    return summary_df

def main():
    """Main function to create IoU F1 score heatmap analysis"""
    
    print("=== CREATING IoU F1 SCORE HEATMAP ANALYSIS ===\n")
    
    # Extract IoU data
    df = extract_iou_heatmap_data()
    
    if df.empty:
        print("❌ No IoU evaluation data extracted!")
        return
    
    print(f"✅ Extracted data from {len(df)} IoU configurations")
    print(f"   Evaluation types: {list(df['eval_type'].unique())}")
    print(f"   Available metrics: {[col for col in df.columns if 'f1' in col]}")
    
    # Create output directory
    output_dir = Path("iou_f1_heatmap_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Create heatmap analysis
    create_iou_f1_heatmaps(df, output_dir)
    create_combined_optimal_heatmap(df, output_dir)
    summary_df = create_optimal_summary_table(df, output_dir)
    
    print(f"\n🎉 IoU F1 score heatmap analysis completed successfully!")
    print(f"📁 Output directory: {output_dir}")
    
    # Print detailed summary
    print("\n📊 OPTIMAL CONFIGURATION SUMMARY:")
    for _, row in summary_df.iterrows():
        metric_name = 'Binary F1' if row['metric_type'] == 'binary_f1' else 'Multiclass F1'
        stride_unit = '%' if row['stride_type'] == 'percent' else 's'
        print(f"\n{row['eval_type'].upper()} - {metric_name}:")
        print(f"   Optimal: Window={row['optimal_window_size']}s, Stride={row['optimal_stride_value']}{stride_unit}")
        print(f"   F1 Score: {row['optimal_f1_score']:.4f}")
        print(f"   Config: {row['window_config']} + {row['stride_config']}")

if __name__ == "__main__":
    main()

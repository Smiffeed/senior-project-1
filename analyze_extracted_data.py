#!/usr/bin/env python3
"""
Data Analysis and Visualization Script for Extracted Evaluation Data
Recreates the key visualizations from the extracted data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_extracted_data(data_dir="./extracted_evaluation_data"):
    """Load all extracted data files"""
    data_path = Path(data_dir)
    
    # Load main dataset
    complete_df = pd.read_csv(data_path / "complete_evaluation_data.csv")
    
    # Load heatmap data
    heatmap_data = {}
    for file in data_path.glob("*_heatmap.csv"):
        key = file.stem
        heatmap_data[key] = pd.read_csv(file, index_col=0)
    
    # Load line graph data
    line_data = {}
    for file in data_path.glob("*_line_graph_data.csv"):
        key = file.stem.replace("_line_graph_data", "")
        line_data[key] = pd.read_csv(file)
    
    # Load optimal configurations
    optimal_df = pd.read_csv(data_path / "optimal_configurations.csv")
    
    # Load summary statistics
    with open(data_path / "summary_statistics.json", 'r', encoding='utf-8') as f:
        summary_stats = json.load(f)
    
    return complete_df, heatmap_data, line_data, optimal_df, summary_stats

def recreate_binary_f1_heatmap(heatmap_data, eval_type="eval_percent"):
    """Recreate the Binary F1 Score Heatmap"""
    plt.figure(figsize=(14, 10))
    
    data_key = f"{eval_type}_binary_f1_heatmap"
    if data_key not in heatmap_data:
        print(f"No data found for {data_key}")
        return
    
    data = heatmap_data[data_key]
    
    # Find optimal value and position
    max_val = data.max().max()
    max_pos = np.where(data == max_val)
    if len(max_pos[0]) > 0:
        max_window = data.columns[max_pos[1][0]]
        max_stride = data.index[max_pos[0][0]]
    
    # Create heatmap
    sns.heatmap(data, 
                annot=True, 
                fmt='.3f',
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Binary F1 Score'},
                linewidths=0.5)
    
    plt.title(f'{eval_type} - IoU Evaluation - Binary F1 Score Heatmap\n'
              f'Optimal: Window={max_window}s, Stride={max_stride}%, F1={max_val:.4f}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Window Size (seconds)', fontsize=12)
    plt.ylabel('Stride Size (%)', fontsize=12)
    plt.tight_layout()
    
    return plt.gcf()

def recreate_iou_heatmap(heatmap_data, eval_type="eval_percent"):
    """Recreate the Combined Mean IoU Heatmap"""
    plt.figure(figsize=(14, 10))
    
    data_key = f"{eval_type}_combined_mean_iou_heatmap"
    if data_key not in heatmap_data:
        print(f"No data found for {data_key}")
        return
    
    data = heatmap_data[data_key]
    
    # Find optimal value and position
    max_val = data.max().max()
    max_pos = np.where(data == max_val)
    if len(max_pos[0]) > 0:
        max_window = data.columns[max_pos[1][0]]
        max_stride = data.index[max_pos[0][0]]
    
    # Create heatmap with values displayed
    sns.heatmap(data, 
                annot=True, 
                fmt='.2f',
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Combined Mean IoU (%)'},
                linewidths=0.5)
    
    plt.title(f'{eval_type} - IoU Evaluation - Combined Mean IoU Percentage\n'
              f'Optimal: Window={max_window}s, Stride={max_stride}%, IoU={max_val:.2f}%',
              fontsize=14, fontweight='bold')
    plt.xlabel('Window Size (seconds)', fontsize=12)
    plt.ylabel('Stride Size (%)', fontsize=12)
    plt.tight_layout()
    
    return plt.gcf()

def recreate_comparison_line_graphs(complete_df):
    """Recreate the method comparison line graphs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Separate data by evaluation type
    eval_percent_data = complete_df[complete_df['eval_type'] == 'eval_percent']
    eval_by_005_data = complete_df[complete_df['eval_type'] == 'eval_by_0']
    
    # Group by window size and calculate statistics
    def calculate_stats(df, metric):
        stats = df.groupby('window_size')[metric].agg(['mean', 'std']).reset_index()
        return stats
    
    # Binary F1 comparison
    if len(eval_percent_data) > 0:
        binary_stats_percent = calculate_stats(eval_percent_data, 'binary_f1')
        ax1.plot(binary_stats_percent['window_size'], binary_stats_percent['mean'], 
                'o-', label='Window Evaluation (eval_percent)', linewidth=2, markersize=6)
        ax1.fill_between(binary_stats_percent['window_size'], 
                        binary_stats_percent['mean'] - binary_stats_percent['std'],
                        binary_stats_percent['mean'] + binary_stats_percent['std'],
                        alpha=0.2)
    
    if len(eval_by_005_data) > 0:
        binary_stats_005 = calculate_stats(eval_by_005_data, 'binary_f1')
        ax1.plot(binary_stats_005['window_size'], binary_stats_005['mean'], 
                's--', label='Window Evaluation (eval_by_0.05)', linewidth=2, markersize=6)
        ax1.fill_between(binary_stats_005['window_size'], 
                        binary_stats_005['mean'] - binary_stats_005['std'],
                        binary_stats_005['mean'] + binary_stats_005['std'],
                        alpha=0.2)
    
    ax1.set_title('All Methods - Binary F1 Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Window Size (seconds)')
    ax1.set_ylabel('Binary F1 Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Multiclass F1 comparison
    if len(eval_percent_data) > 0:
        multi_stats_percent = calculate_stats(eval_percent_data, 'multiclass_f1')
        ax2.plot(multi_stats_percent['window_size'], multi_stats_percent['mean'], 
                'o-', label='Window Evaluation (eval_percent)', linewidth=2, markersize=6)
        ax2.fill_between(multi_stats_percent['window_size'], 
                        multi_stats_percent['mean'] - multi_stats_percent['std'],
                        multi_stats_percent['mean'] + multi_stats_percent['std'],
                        alpha=0.2)
    
    if len(eval_by_005_data) > 0:
        multi_stats_005 = calculate_stats(eval_by_005_data, 'multiclass_f1')
        ax2.plot(multi_stats_005['window_size'], multi_stats_005['mean'], 
                's--', label='Window Evaluation (eval_by_0.05)', linewidth=2, markersize=6)
        ax2.fill_between(multi_stats_005['window_size'], 
                        multi_stats_005['mean'] - multi_stats_005['std'],
                        multi_stats_005['mean'] + multi_stats_005['std'],
                        alpha=0.2)
    
    ax2.set_title('All Methods - Multiclass F1 Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Window Size (seconds)')
    ax2.set_ylabel('Multiclass F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def recreate_iou_line_graph(complete_df):
    """Recreate the IoU evaluation line graph"""
    plt.figure(figsize=(12, 8))
    
    # Separate data by evaluation type
    eval_percent_data = complete_df[complete_df['eval_type'] == 'eval_percent']
    eval_by_005_data = complete_df[complete_df['eval_type'] == 'eval_by_0']
    
    # Calculate statistics for each evaluation type
    def calculate_iou_stats(df):
        stats = df.groupby('window_size')['combined_mean_iou'].agg(['mean', 'std']).reset_index()
        return stats
    
    if len(eval_percent_data) > 0:
        iou_stats_percent = calculate_iou_stats(eval_percent_data)
        plt.plot(iou_stats_percent['window_size'], iou_stats_percent['mean'], 
                'o-', label='eval_percent', linewidth=2, markersize=6)
        plt.fill_between(iou_stats_percent['window_size'], 
                        iou_stats_percent['mean'] - iou_stats_percent['std'],
                        iou_stats_percent['mean'] + iou_stats_percent['std'],
                        alpha=0.2)
    
    if len(eval_by_005_data) > 0:
        iou_stats_005 = calculate_iou_stats(eval_by_005_data)
        plt.plot(iou_stats_005['window_size'], iou_stats_005['mean'], 
                's--', label='eval_by_0.05', linewidth=2, markersize=6)
        plt.fill_between(iou_stats_005['window_size'], 
                        iou_stats_005['mean'] - iou_stats_005['std'],
                        iou_stats_005['mean'] + iou_stats_005['std'],
                        alpha=0.2)
    
    plt.title('IoU Evaluation - Combined Mean IoU (Simple Average)', fontsize=14, fontweight='bold')
    plt.xlabel('Window Size (seconds)')
    plt.ylabel('Mean IoU (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def analyze_optimal_configurations(optimal_df):
    """Analyze and display optimal configurations"""
    print("🎯 OPTIMAL CONFIGURATIONS ANALYSIS")
    print("=" * 50)
    
    for _, row in optimal_df.iterrows():
        eval_type = row['eval_type']
        metric = row['metric']
        window_size = row['window_size']
        stride_value = row['stride_value']
        stride_type = row['stride_type']
        value = row['value']
        
        print(f"\n📊 {eval_type.upper()} - Best {metric.upper()}:")
        print(f"  Window Size: {window_size}s")
        print(f"  Stride: {stride_value}% ({stride_type})")
        print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
        
        if 'combined_mean_iou' in row and pd.notna(row['combined_mean_iou']):
            print(f"  Combined Mean IoU: {row['combined_mean_iou']:.2f}%")
        if 'binary_f1' in row and pd.notna(row['binary_f1']):
            print(f"  Binary F1: {row['binary_f1']:.3f}")

def create_comprehensive_report(complete_df, summary_stats):
    """Create a comprehensive analysis report"""
    print("\n📈 COMPREHENSIVE EVALUATION REPORT")
    print("=" * 60)
    
    print(f"\n🔍 DATASET OVERVIEW:")
    print(f"Total Configurations: {len(complete_df)}")
    
    for eval_type in complete_df['eval_type'].unique():
        subset = complete_df[complete_df['eval_type'] == eval_type]
        print(f"\n  📋 {eval_type.upper()}:")
        print(f"    Configurations: {len(subset)}")
        print(f"    Window Range: {subset['window_size'].min()}s - {subset['window_size'].max()}s")
        print(f"    Stride Range: {subset['stride_value'].min()}% - {subset['stride_value'].max()}%")
        
        print(f"\n    🎯 PERFORMANCE METRICS:")
        print(f"    Binary F1:        {subset['binary_f1'].min():.3f} - {subset['binary_f1'].max():.3f} (avg: {subset['binary_f1'].mean():.3f})")
        print(f"    Multiclass F1:     {subset['multiclass_f1'].min():.3f} - {subset['multiclass_f1'].max():.3f} (avg: {subset['multiclass_f1'].mean():.3f})")
        print(f"    Combined Mean IoU: {subset['combined_mean_iou'].min():.2f}% - {subset['combined_mean_iou'].max():.2f}% (avg: {subset['combined_mean_iou'].mean():.2f}%)")
    
    # Window size analysis
    print(f"\n🪟 WINDOW SIZE ANALYSIS:")
    window_analysis = complete_df.groupby('window_size').agg({
        'binary_f1': ['mean', 'max'],
        'multiclass_f1': ['mean', 'max'],
        'combined_mean_iou': ['mean', 'max']
    }).round(3)
    
    print("\nTop 5 windows by average Binary F1:")
    top_binary = complete_df.groupby('window_size')['binary_f1'].mean().sort_values(ascending=False).head()
    for window, f1 in top_binary.items():
        print(f"  {window}s: {f1:.3f}")
    
    print("\nTop 5 windows by average Combined Mean IoU:")
    top_iou = complete_df.groupby('window_size')['combined_mean_iou'].mean().sort_values(ascending=False).head()
    for window, iou in top_iou.items():
        print(f"  {window}s: {iou:.2f}%")

def save_visualizations(complete_df, heatmap_data, output_dir="./visualization_outputs"):
    """Save all recreated visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 SAVING VISUALIZATIONS TO: {output_dir}")
    
    # Save Binary F1 heatmaps
    for eval_type in ['eval_percent']:  # Only eval_percent has enough data
        fig = recreate_binary_f1_heatmap(heatmap_data, eval_type)
        if fig:
            fig.savefig(output_path / f'{eval_type}_binary_f1_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Saved {eval_type}_binary_f1_heatmap.png")
    
    # Save IoU heatmaps
    for eval_type in ['eval_percent']:
        fig = recreate_iou_heatmap(heatmap_data, eval_type)
        if fig:
            fig.savefig(output_path / f'{eval_type}_iou_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✅ Saved {eval_type}_iou_heatmap.png")
    
    # Save comparison line graphs
    fig = recreate_comparison_line_graphs(complete_df)
    fig.savefig(output_path / 'method_comparison_line_graphs.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✅ Saved method_comparison_line_graphs.png")
    
    # Save IoU line graph
    fig = recreate_iou_line_graph(complete_df)
    fig.savefig(output_path / 'iou_evaluation_line_graph.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✅ Saved iou_evaluation_line_graph.png")

def main():
    """Main function to analyze extracted data"""
    print("🎨 EVALUATION DATA VISUALIZATION AND ANALYSIS")
    print("=" * 60)
    
    # Load all data
    try:
        complete_df, heatmap_data, line_data, optimal_df, summary_stats = load_extracted_data()
        print("✅ Successfully loaded all extracted data")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Analyze optimal configurations
    analyze_optimal_configurations(optimal_df)
    
    # Create comprehensive report
    create_comprehensive_report(complete_df, summary_stats)
    
    # Save visualizations
    save_visualizations(complete_df, heatmap_data)
    
    print(f"\n🎉 ANALYSIS COMPLETED!")
    print(f"📊 Key Findings:")
    print(f"  • Best Binary F1: {complete_df['binary_f1'].max():.3f}")
    print(f"  • Best Multiclass F1: {complete_df['multiclass_f1'].max():.3f}")
    print(f"  • Best Combined Mean IoU: {complete_df['combined_mean_iou'].max():.2f}%")
    print(f"  • Total Configurations Analyzed: {len(complete_df)}")
    print(f"\n📁 Check ./visualization_outputs/ for recreated graphs")
    print(f"📁 Check ./extracted_evaluation_data/ for raw data files")

if __name__ == "__main__":
    main()
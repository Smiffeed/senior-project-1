#!/usr/bin/env python3
"""
Quick Data Sample Viewer
Shows examples of the extracted data structure
"""

import pandas as pd
import json
from pathlib import Path

def show_data_samples():
    """Display sample data from each extracted file"""
    data_dir = Path("./extracted_evaluation_data")
    
    print("🔍 EXTRACTED DATA SAMPLES")
    print("=" * 60)
    
    # 1. Complete Dataset Sample
    print("\n📊 1. COMPLETE EVALUATION DATA (sample)")
    print("-" * 40)
    df = pd.read_csv(data_dir / "complete_evaluation_data.csv")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 3 rows:")
    sample_cols = ['window_size', 'stride_value', 'eval_type', 'binary_f1', 'multiclass_f1', 'combined_mean_iou']
    print(df[sample_cols].head(3).to_string(index=False))
    
    # 2. Binary F1 Heatmap Sample
    print("\n📈 2. BINARY F1 HEATMAP DATA (sample)")
    print("-" * 40)
    heatmap = pd.read_csv(data_dir / "eval_percent_binary_f1_heatmap.csv", index_col=0)
    print(f"Shape: {heatmap.shape} (stride_values × window_sizes)")
    print(f"Stride range: {heatmap.index.min()}% - {heatmap.index.max()}%")
    print(f"Window range: {heatmap.columns.astype(float).min()}s - {heatmap.columns.astype(float).max()}s")
    print("\nSample (first 3 strides, first 5 windows):")
    print(heatmap.iloc[:3, :5].round(3).to_string())
    
    # 3. IoU Heatmap Sample
    print("\n🎯 3. COMBINED MEAN IOU HEATMAP DATA (sample)")
    print("-" * 40)
    iou_heatmap = pd.read_csv(data_dir / "eval_percent_combined_mean_iou_heatmap.csv", index_col=0)
    print(f"Shape: {iou_heatmap.shape}")
    print(f"IoU range: {iou_heatmap.min().min():.2f}% - {iou_heatmap.max().max():.2f}%")
    print("\nSample (first 3 strides, first 5 windows):")
    print(iou_heatmap.iloc[:3, :5].round(2).to_string())
    
    # 4. Optimal Configurations
    print("\n🏆 4. OPTIMAL CONFIGURATIONS")
    print("-" * 40)
    optimal = pd.read_csv(data_dir / "optimal_configurations.csv")
    for _, row in optimal.iterrows():
        print(f"Best {row['metric']}: Window={row['window_size']}s, Stride={row['stride_value']}%, Value={row['value']:.3f}")
    
    # 5. Class-specific IoU Sample
    print("\n🎯 5. CLASS-SPECIFIC IOU (เย็ด class sample)")
    print("-" * 40)
    class_iou = pd.read_csv(data_dir / "eval_percent_iou_เย็ด_heatmap.csv", index_col=0)
    print(f"Shape: {class_iou.shape}")
    print(f"IoU range for 'เย็ด': {class_iou.min().min():.2f}% - {class_iou.max().max():.2f}%")
    print("\nSample (first 3 strides, first 5 windows):")
    print(class_iou.iloc[:3, :5].round(2).to_string())
    
    # 6. Summary Statistics
    print("\n📋 6. SUMMARY STATISTICS")
    print("-" * 40)
    with open(data_dir / "summary_statistics.json", 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    for eval_type, data in stats.items():
        print(f"\n{eval_type.upper()}:")
        print(f"  Configurations: {data['total_configurations']}")
        print(f"  Window Sizes: {len(data['window_sizes'])} sizes ({data['window_sizes'][0]}s - {data['window_sizes'][-1]}s)")
        print(f"  Binary F1 Range: {data['performance_ranges']['binary_f1']['min']:.3f} - {data['performance_ranges']['binary_f1']['max']:.3f}")
        print(f"  IoU Range: {data['performance_ranges']['combined_mean_iou']['min']:.2f}% - {data['performance_ranges']['combined_mean_iou']['max']:.2f}%")
    
    # 7. Data Usage Examples
    print("\n💡 7. QUICK USAGE EXAMPLES")
    print("-" * 40)
    
    print("\n# Load and filter high-performance configs:")
    print("df = pd.read_csv('complete_evaluation_data.csv')")
    print("high_f1 = df[df['binary_f1'] > 0.38]")
    print(f"# Result: {len(df[df['binary_f1'] > 0.38])} configurations")
    
    print("\n# Load heatmap for direct plotting:")
    print("heatmap = pd.read_csv('eval_percent_binary_f1_heatmap.csv', index_col=0)")
    print("sns.heatmap(heatmap, annot=True)")
    
    print("\n# Find optimal window for specific stride:")
    stride_80 = df[df['stride_value'] == 80.0].sort_values('binary_f1', ascending=False).iloc[0]
    print(f"# Best window for 80% stride: {stride_80['window_size']}s (F1={stride_80['binary_f1']:.3f})")
    
    print("\n📁 All files available in: ./extracted_evaluation_data/")
    print("📊 Visualizations available in: ./visualization_outputs/")

if __name__ == "__main__":
    show_data_samples()
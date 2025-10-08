import os
import pandas as pd
import numpy as np
from pathlib import Path
import re

def extract_metrics_from_note(note_path):
    """Extract key metrics from note.txt file"""
    try:
        with open(note_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metrics using regex
        overall_iou_match = re.search(r'Overall Mean IoU: ([\d.]+)', content)
        binary_f1_match = re.search(r'Traditional Binary F1: ([\d.]+)', content)
        multiclass_f1_match = re.search(r'Traditional Multiclass F1: ([\d.]+)', content)
        best_threshold_match = re.search(r'Best IoU Threshold: ([\d.]+) \(F1=([\d.]+)\)', content)
        
        return {
            'overall_mean_iou': float(overall_iou_match.group(1)) if overall_iou_match else None,
            'traditional_binary_f1': float(binary_f1_match.group(1)) if binary_f1_match else None,
            'traditional_multiclass_f1': float(multiclass_f1_match.group(1)) if multiclass_f1_match else None,
            'best_iou_threshold': float(best_threshold_match.group(1)) if best_threshold_match else None,
            'best_iou_f1': float(best_threshold_match.group(2)) if best_threshold_match else None,
        }
    except Exception as e:
        print(f"Error processing {note_path}: {e}")
        return None

def analyze_comprehensive_results():
    base_dir = Path("comprehensive_iou_evaluation")
    results = []
    
    # Process eval_by_0.05 results
    eval_by_005_dir = base_dir / "eval_by_0.05" / "word"
    if eval_by_005_dir.exists():
        for window_dir in eval_by_005_dir.iterdir():
            if window_dir.is_dir():
                window_size = window_dir.name.replace('window_', '').replace('s', '')
                for stride_dir in window_dir.iterdir():
                    if stride_dir.is_dir():
                        stride_value = stride_dir.name.replace('stride_', '').replace('s', '')
                        note_file = stride_dir / "note.txt"
                        if note_file.exists():
                            metrics = extract_metrics_from_note(note_file)
                            if metrics:
                                results.append({
                                    'eval_type': 'eval_by_0.05',
                                    'window_size': float(window_size),
                                    'stride_value': float(stride_value),
                                    'stride_type': 'absolute',
                                    **metrics
                                })
    
    # Process eval_percent results
    eval_percent_dir = base_dir / "eval_percent" / "word"
    if eval_percent_dir.exists():
        for window_dir in eval_percent_dir.iterdir():
            if window_dir.is_dir():
                window_size = window_dir.name.replace('window_', '').replace('s', '')
                for stride_dir in window_dir.iterdir():
                    if stride_dir.is_dir():
                        stride_value = stride_dir.name.replace('stride_', '').replace('pct', '')
                        note_file = stride_dir / "note.txt"
                        if note_file.exists():
                            metrics = extract_metrics_from_note(note_file)
                            if metrics:
                                results.append({
                                    'eval_type': 'eval_percent',
                                    'window_size': float(window_size),
                                    'stride_value': float(stride_value),
                                    'stride_type': 'percentage',
                                    **metrics
                                })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    if df.empty:
        print("No results found!")
        return
    
    print(f"🔍 COMPREHENSIVE IoU EVALUATION ANALYSIS")
    print(f"=" * 60)
    print(f"📊 Total configurations analyzed: {len(df)}")
    print(f"📊 eval_by_0.05 configurations: {len(df[df['eval_type'] == 'eval_by_0.05'])}")
    print(f"📊 eval_percent configurations: {len(df[df['eval_type'] == 'eval_percent'])}")
    print()
    
    # Overall statistics
    print(f"📈 OVERALL STATISTICS")
    print(f"-" * 40)
    print(f"Overall Mean IoU: {df['overall_mean_iou'].mean():.3f} ± {df['overall_mean_iou'].std():.3f}")
    print(f"Traditional Binary F1: {df['traditional_binary_f1'].mean():.3f} ± {df['traditional_binary_f1'].std():.3f}")
    print(f"Traditional Multiclass F1: {df['traditional_multiclass_f1'].mean():.3f} ± {df['traditional_multiclass_f1'].std():.3f}")
    print(f"Best IoU F1: {df['best_iou_f1'].mean():.3f} ± {df['best_iou_f1'].std():.3f}")
    print()
    
    # Top 10 by Overall Mean IoU
    print(f"🏆 TOP 10 BY OVERALL MEAN IoU")
    print(f"-" * 80)
    top_iou = df.nlargest(10, 'overall_mean_iou')
    for i, (_, row) in enumerate(top_iou.iterrows(), 1):
        print(f"{i:2d}. {row['eval_type']:12s} | Window: {row['window_size']:4.1f}s | Stride: {row['stride_value']:5.1f}{row['stride_type'][0]} | IoU: {row['overall_mean_iou']:.3f} | Binary F1: {row['traditional_binary_f1']:.3f}")
    print()
    
    # Top 10 by Binary F1
    print(f"🎯 TOP 10 BY TRADITIONAL BINARY F1")
    print(f"-" * 80)
    top_f1 = df.nlargest(10, 'traditional_binary_f1')
    for i, (_, row) in enumerate(top_f1.iterrows(), 1):
        print(f"{i:2d}. {row['eval_type']:12s} | Window: {row['window_size']:4.1f}s | Stride: {row['stride_value']:5.1f}{row['stride_type'][0]} | Binary F1: {row['traditional_binary_f1']:.3f} | IoU: {row['overall_mean_iou']:.3f}")
    print()
    
    # Top 10 by Best IoU F1
    print(f"⚡ TOP 10 BY BEST IoU F1 (Optimized Threshold)")
    print(f"-" * 80)
    top_iou_f1 = df.nlargest(10, 'best_iou_f1')
    for i, (_, row) in enumerate(top_iou_f1.iterrows(), 1):
        print(f"{i:2d}. {row['eval_type']:12s} | Window: {row['window_size']:4.1f}s | Stride: {row['stride_value']:5.1f}{row['stride_type'][0]} | IoU F1: {row['best_iou_f1']:.3f} @ {row['best_iou_threshold']:.1f} | IoU: {row['overall_mean_iou']:.3f}")
    print()
    
    # Analysis by evaluation type
    print(f"📊 ANALYSIS BY EVALUATION TYPE")
    print(f"-" * 50)
    for eval_type in ['eval_by_0.05', 'eval_percent']:
        subset = df[df['eval_type'] == eval_type]
        print(f"{eval_type}:")
        print(f"  Configurations: {len(subset)}")
        print(f"  Mean IoU: {subset['overall_mean_iou'].mean():.3f} ± {subset['overall_mean_iou'].std():.3f}")
        print(f"  Binary F1: {subset['traditional_binary_f1'].mean():.3f} ± {subset['traditional_binary_f1'].std():.3f}")
        print(f"  Best Config (IoU): Window {subset.loc[subset['overall_mean_iou'].idxmax(), 'window_size']:.1f}s, Stride {subset.loc[subset['overall_mean_iou'].idxmax(), 'stride_value']:.1f}")
        print()
    
    # Window size analysis
    print(f"🔍 WINDOW SIZE ANALYSIS")
    print(f"-" * 40)
    window_analysis = df.groupby('window_size').agg({
        'overall_mean_iou': ['mean', 'std', 'count'],
        'traditional_binary_f1': ['mean', 'std'],
        'best_iou_f1': ['mean', 'std']
    }).round(3)
    
    for window_size in sorted(df['window_size'].unique()):
        subset = df[df['window_size'] == window_size]
        print(f"Window {window_size:.1f}s ({len(subset)} configs): IoU={subset['overall_mean_iou'].mean():.3f}±{subset['overall_mean_iou'].std():.3f}, Binary F1={subset['traditional_binary_f1'].mean():.3f}±{subset['traditional_binary_f1'].std():.3f}")
    
    # Save comprehensive results
    output_file = "comprehensive_iou_analysis_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Comprehensive results saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    df = analyze_comprehensive_results()
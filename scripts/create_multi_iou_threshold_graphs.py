#!/usr/bin/env python3
"""
Create Multi-IoU Threshold Comparison Graphs
Generate F1 score comparisons across IoU thresholds 0.3, 0.5, and 0.9
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import os

def extract_comprehensive_data():
    """Load data from the comprehensive analysis"""
    
    base_dir = Path("fixed_smart_parallel_results")
    
    all_results = []
    
    # Process both eval_by_0.05 and eval_percent directories
    for eval_type in ["eval_by_0.05", "eval_percent"]:
        eval_dir = base_dir / eval_type / eval_type
        
        if not eval_dir.exists():
            print(f"Directory not found: {eval_dir}")
            continue
        
        print(f"Processing {eval_type}...")
        
        # Method 2: Word-IoU Evaluation
        word_iou_eval_dir = eval_dir / "word_iou_eval"
        if word_iou_eval_dir.exists():
            for window_dir in word_iou_eval_dir.iterdir():
                if window_dir.is_dir() and window_dir.name.startswith("window_"):
                    for stride_dir in window_dir.iterdir():
                        if stride_dir.is_dir() and stride_dir.name.startswith("stride_"):
                            note_file = stride_dir / "note.txt"
                            if note_file.exists():
                                scores = extract_word_iou_f1_all_thresholds(note_file)
                                if scores:
                                    config_data = {
                                        'eval_type': eval_type,
                                        'method': 'word_iou_eval',
                                        'window': window_dir.name,
                                        'stride': stride_dir.name,
                                        'window_numeric': float(re.search(r'window_(\d+\.?\d*)s', window_dir.name).group(1)),
                                        **scores
                                    }
                                    all_results.append(config_data)
        
        # Method 3: IoU Evaluation  
        iou_eval_dir = eval_dir / "iou_eval"
        if iou_eval_dir.exists():
            for window_dir in iou_eval_dir.iterdir():
                if window_dir.is_dir() and window_dir.name.startswith("window_"):
                    for stride_dir in window_dir.iterdir():
                        if stride_dir.is_dir() and stride_dir.name.startswith("stride_"):
                            note_file = stride_dir / "note.txt"
                            if note_file.exists():
                                scores = extract_iou_f1_all_thresholds(note_file)
                                if scores:
                                    config_data = {
                                        'eval_type': eval_type,
                                        'method': 'iou_eval',
                                        'window': window_dir.name,
                                        'stride': stride_dir.name,
                                        'window_numeric': float(re.search(r'window_(\d+\.?\d*)s', window_dir.name).group(1)),
                                        **scores
                                    }
                                    all_results.append(config_data)
    
    return pd.DataFrame(all_results)

def extract_word_iou_f1_all_thresholds(note_file):
    """Extract F1 scores for multiple IoU thresholds from word-IoU evaluation"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        f1_scores = {}
        
        # Extract F1 scores for IoU thresholds 0.1, 0.3, 0.5, 0.7, 0.9
        thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
        
        for threshold in thresholds:
            # Binary F1
            pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?F1-Score:\s*([\d.]+)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                f1_scores[f'binary_f1_iou_{threshold}'] = float(match.group(1))
            
            # Multiclass F1  
            pattern = rf'--- IoU Threshold {threshold} ---.*?Multiclass Classification.*?F1-Score:\s*([\d.]+)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                f1_scores[f'multiclass_f1_iou_{threshold}'] = float(match.group(1))
        
        return f1_scores
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return {}

def extract_iou_f1_all_thresholds(note_file):
    """Extract F1 scores for multiple IoU thresholds from IoU evaluation"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        f1_scores = {}
        
        # Extract F1 scores for IoU thresholds 0.1, 0.3, 0.5, 0.7, 0.9
        thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
        
        for threshold in thresholds:
            # Binary F1
            pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?F1-Score:\s*([\d.]+)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                f1_scores[f'iou_binary_f1_{threshold}'] = float(match.group(1))
            
            # Multiclass F1 from classification report (weighted avg)
            pattern = rf'--- IoU Threshold {threshold} ---.*?Multiclass Classification.*?weighted avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                f1_scores[f'iou_multiclass_f1_{threshold}'] = float(match.group(1))
        
        return f1_scores
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return {}

def calculate_window_stats_multi_threshold(df, thresholds, metric_prefix):
    """Calculate statistics grouped by window size for multiple thresholds"""
    
    window_sizes = sorted(df['window_numeric'].unique())
    stats = []
    
    for window_size in window_sizes:
        window_data = df[df['window_numeric'] == window_size]
        
        stat_dict = {'window_size': window_size}
        
        for threshold in thresholds:
            for f1_type in ['binary', 'multiclass']:
                if metric_prefix == 'word_iou':
                    col_name = f'{f1_type}_f1_iou_{threshold}'
                else:  # iou_eval
                    col_name = f'iou_{f1_type}_f1_{threshold}'
                
                if col_name in window_data.columns:
                    values = window_data[col_name].dropna()
                    if len(values) > 0:
                        stat_dict.update({
                            f'{col_name}_mean': values.mean(),
                            f'{col_name}_std': values.std(),
                            f'{col_name}_count': len(values),
                            f'{col_name}_min': values.min(),
                            f'{col_name}_max': values.max()
                        })
        
        stats.append(stat_dict)
    
    return pd.DataFrame(stats)

def create_multi_threshold_graphs():
    """Create comparison graphs for IoU thresholds 0.3, 0.5, and 0.9"""
    
    print("=== CREATING MULTI-IoU THRESHOLD COMPARISON GRAPHS ===\n")
    
    # Extract data
    df = extract_comprehensive_data()
    
    if df.empty:
        print("❌ No data extracted!")
        return
    
    print(f"✅ Extracted data from {len(df)} configurations")
    
    # Create output directory
    output_dir = Path("multi_iou_threshold_graphs")
    output_dir.mkdir(exist_ok=True)
    
    # Target IoU thresholds
    target_thresholds = ['0.3', '0.5', '0.9']
    
    # Set style
    plt.style.use('seaborn-v0_8')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Process each method
    methods = [
        ('word_iou_eval', 'Word-IoU Evaluation', 'word_iou'),
        ('iou_eval', 'IoU Evaluation', 'iou_eval')
    ]
    
    for method, method_name, metric_prefix in methods:
        method_data = df[df['method'] == method].copy()
        
        if method_data.empty:
            print(f"⚠️ No data found for {method_name}")
            continue
        
        print(f"📊 Processing {method_name}: {len(method_data)} configurations")
        
        # Calculate statistics
        stats_df = calculate_window_stats_multi_threshold(method_data, target_thresholds, metric_prefix)
        
        if stats_df.empty:
            print(f"⚠️ No statistics calculated for {method_name}")
            continue
        
        # Create Binary F1 comparison
        create_threshold_comparison_plot(
            stats_df, target_thresholds, metric_prefix, 'binary',
            f"{method_name} - Binary F1 Comparison Across IoU Thresholds",
            f"{method_name.lower().replace(' ', '_').replace('-', '_')}_binary_f1_thresholds.png",
            output_dir, colors
        )
        
        # Create Multiclass F1 comparison
        create_threshold_comparison_plot(
            stats_df, target_thresholds, metric_prefix, 'multiclass',
            f"{method_name} - Multiclass F1 Comparison Across IoU Thresholds",
            f"{method_name.lower().replace(' ', '_').replace('-', '_')}_multiclass_f1_thresholds.png",
            output_dir, colors
        )
    
    # Create combined comparison
    create_combined_threshold_comparison(df, target_thresholds, output_dir, colors)
    
    print(f"\n🎉 Multi-IoU threshold graphs created successfully!")
    print(f"📁 Output directory: {output_dir}")

def create_threshold_comparison_plot(stats_df, thresholds, metric_prefix, f1_type, title, filename, output_dir, colors):
    """Create a single threshold comparison plot"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, threshold in enumerate(thresholds):
        if metric_prefix == 'word_iou':
            col_name = f'{f1_type}_f1_iou_{threshold}_mean'
            std_col_name = f'{f1_type}_f1_iou_{threshold}_std'
        else:  # iou_eval
            col_name = f'iou_{f1_type}_f1_{threshold}_mean'
            std_col_name = f'iou_{f1_type}_f1_{threshold}_std'
        
        if col_name in stats_df.columns:
            valid_data = stats_df.dropna(subset=[col_name])
            
            ax.plot(valid_data['window_size'], valid_data[col_name], 
                   marker='o', linewidth=3, markersize=8, 
                   color=colors[i], label=f'IoU {threshold}', alpha=0.8)
            
            # Add error bars if available
            if std_col_name in stats_df.columns:
                ax.fill_between(valid_data['window_size'], 
                               valid_data[col_name] - valid_data[std_col_name],
                               valid_data[col_name] + valid_data[std_col_name],
                               alpha=0.2, color=colors[i])
    
    ax.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{f1_type.capitalize()} F1 Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved: {filename}")

def create_combined_threshold_comparison(df, thresholds, output_dir, colors):
    """Create combined comparison showing both methods"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    methods = [
        ('word_iou_eval', 'Word-IoU Evaluation', 'word_iou'),
        ('iou_eval', 'IoU Evaluation', 'iou_eval')
    ]
    
    # Binary F1 comparisons
    for method_idx, (method, method_name, metric_prefix) in enumerate(methods):
        method_data = df[df['method'] == method].copy()
        if method_data.empty:
            continue
            
        stats_df = calculate_window_stats_multi_threshold(method_data, thresholds, metric_prefix)
        
        ax = ax1 if method_idx == 0 else ax2
        
        for i, threshold in enumerate(thresholds):
            if metric_prefix == 'word_iou':
                col_name = f'binary_f1_iou_{threshold}_mean'
                std_col_name = f'binary_f1_iou_{threshold}_std'
            else:
                col_name = f'iou_binary_f1_{threshold}_mean'
                std_col_name = f'iou_binary_f1_{threshold}_std'
            
            if col_name in stats_df.columns:
                valid_data = stats_df.dropna(subset=[col_name])
                
                ax.plot(valid_data['window_size'], valid_data[col_name], 
                       marker='o', linewidth=3, markersize=6, 
                       color=colors[i], label=f'IoU {threshold}', alpha=0.8)
                
                if std_col_name in stats_df.columns:
                    ax.fill_between(valid_data['window_size'], 
                                   valid_data[col_name] - valid_data[std_col_name],
                                   valid_data[col_name] + valid_data[std_col_name],
                                   alpha=0.2, color=colors[i])
        
        ax.set_xlabel('Window Size (seconds)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Binary F1 Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{method_name} - Binary F1', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    # Multiclass F1 comparisons
    for method_idx, (method, method_name, metric_prefix) in enumerate(methods):
        method_data = df[df['method'] == method].copy()
        if method_data.empty:
            continue
            
        stats_df = calculate_window_stats_multi_threshold(method_data, thresholds, metric_prefix)
        
        ax = ax3 if method_idx == 0 else ax4
        
        for i, threshold in enumerate(thresholds):
            if metric_prefix == 'word_iou':
                col_name = f'multiclass_f1_iou_{threshold}_mean'
                std_col_name = f'multiclass_f1_iou_{threshold}_std'
            else:
                col_name = f'iou_multiclass_f1_{threshold}_mean'
                std_col_name = f'iou_multiclass_f1_{threshold}_std'
            
            if col_name in stats_df.columns:
                valid_data = stats_df.dropna(subset=[col_name])
                
                ax.plot(valid_data['window_size'], valid_data[col_name], 
                       marker='s', linewidth=3, markersize=6, 
                       color=colors[i], label=f'IoU {threshold}', alpha=0.8)
                
                if std_col_name in stats_df.columns:
                    ax.fill_between(valid_data['window_size'], 
                                   valid_data[col_name] - valid_data[std_col_name],
                                   valid_data[col_name] + valid_data[std_col_name],
                                   alpha=0.2, color=colors[i])
        
        ax.set_xlabel('Window Size (seconds)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Multiclass F1 Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{method_name} - Multiclass F1', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "combined_multi_iou_threshold_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved: combined_multi_iou_threshold_comparison.png")

if __name__ == "__main__":
    create_multi_threshold_graphs()

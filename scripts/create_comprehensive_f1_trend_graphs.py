#!/usr/bin/env python3
"""
Create comprehensive F1 trend graphs for all evaluation methods (excluding Window F1)
Generates separate trend graphs for:
1. Word-level Evaluation F1
2. Word-IoU Evaluation F1 (multiple thresholds)
3. IoU Evaluation F1 (multiple thresholds)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import re
import os

def extract_f1_from_note_files():
    """Extract F1 scores from all evaluation method note files"""
    base_dir = Path("fixed_smart_parallel_results")
    
    all_results = []
    
    # Process both eval_by_0.05 and eval_percent directories
    for eval_type in ["eval_by_0.05", "eval_percent"]:
        eval_dir = base_dir / eval_type / eval_type
        
        if not eval_dir.exists():
            print(f"Directory not found: {eval_dir}")
            continue
            
        print(f"Processing {eval_type}...")
        
        # Method 1: Word Evaluation
        word_eval_dir = eval_dir / "word_eval"
        if word_eval_dir.exists():
            for window_dir in word_eval_dir.iterdir():
                if window_dir.is_dir() and window_dir.name.startswith("window_"):
                    for stride_dir in window_dir.iterdir():
                        if stride_dir.is_dir() and stride_dir.name.startswith("stride_"):
                            note_file = stride_dir / "note.txt"
                            if note_file.exists():
                                f1_scores = extract_word_eval_f1(note_file)
                                if f1_scores:
                                    f1_scores.update({
                                        'eval_type': eval_type,
                                        'window': window_dir.name,
                                        'stride': stride_dir.name,
                                        'method': 'word_eval'
                                    })
                                    all_results.append(f1_scores)
        
        # Method 2: Word-IoU Evaluation  
        word_iou_dir = eval_dir / "word_iou_eval"
        if word_iou_dir.exists():
            for window_dir in word_iou_dir.iterdir():
                if window_dir.is_dir() and window_dir.name.startswith("window_"):
                    for stride_dir in window_dir.iterdir():
                        if stride_dir.is_dir() and stride_dir.name.startswith("stride_"):
                            note_file = stride_dir / "note.txt"
                            if note_file.exists():
                                f1_scores = extract_word_iou_eval_f1(note_file)
                                if f1_scores:
                                    f1_scores.update({
                                        'eval_type': eval_type,
                                        'window': window_dir.name,
                                        'stride': stride_dir.name,
                                        'method': 'word_iou_eval'
                                    })
                                    all_results.append(f1_scores)
        
        # Method 3: IoU Evaluation
        iou_dir = eval_dir / "iou_eval"
        if iou_dir.exists():
            for window_dir in iou_dir.iterdir():
                if window_dir.is_dir() and window_dir.name.startswith("window_"):
                    for stride_dir in window_dir.iterdir():
                        if stride_dir.is_dir() and stride_dir.name.startswith("stride_"):
                            note_file = stride_dir / "note.txt"
                            if note_file.exists():
                                f1_scores = extract_iou_eval_f1(note_file)
                                if f1_scores:
                                    f1_scores.update({
                                        'eval_type': eval_type,
                                        'window': window_dir.name,
                                        'stride': stride_dir.name,
                                        'method': 'iou_eval'
                                    })
                                    all_results.append(f1_scores)
    
    return pd.DataFrame(all_results)

def extract_word_eval_f1(note_file):
    """Extract F1 scores from word evaluation note file"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        f1_scores = {}
        
        # Extract binary F1 score
        binary_match = re.search(r'Binary Classification.*?F1-Score:\s*([\d.]+)', content, re.DOTALL)
        if binary_match:
            f1_scores['binary_f1'] = float(binary_match.group(1))
        
        # Extract multiclass F1 score
        multiclass_match = re.search(r'Multiclass Classification.*?F1-Score:\s*([\d.]+)', content, re.DOTALL)
        if multiclass_match:
            f1_scores['multiclass_f1'] = float(multiclass_match.group(1))
        
        return f1_scores
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return None

def extract_word_iou_eval_f1(note_file):
    """Extract F1 scores from word-IoU evaluation note file"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        f1_scores = {}
        
        # Extract F1 scores for different IoU thresholds
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
        return None

def extract_iou_eval_f1(note_file):
    """Extract F1 scores from IoU evaluation note file"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        f1_scores = {}
        
        # Extract combined mean IoU
        iou_match = re.search(r'Combined Mean IoU:\s*([\d.]+)', content)
        if iou_match:
            f1_scores['combined_mean_iou'] = float(iou_match.group(1))
        
        # Extract F1 scores for different IoU thresholds
        thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
        
        for threshold in thresholds:
            # Binary F1
            pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?F1-Score:\s*([\d.]+)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                f1_scores[f'iou_binary_f1_{threshold}'] = float(match.group(1))
            
            # Multiclass F1 from classification report
            pattern = rf'--- IoU Threshold {threshold} ---.*?Multiclass Classification.*?weighted avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                f1_scores[f'iou_multiclass_f1_{threshold}'] = float(match.group(1))
            
            # Also extract multiclass accuracy 
            pattern = rf'--- IoU Threshold {threshold} ---.*?Multiclass Classification.*?Accuracy:\s*([\d.]+)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                f1_scores[f'iou_multiclass_acc_{threshold}'] = float(match.group(1))
        
        return f1_scores
    except Exception as e:
        print(f"Error reading {note_file}: {e}")
        return None

def parse_window_stride(df):
    """Parse window and stride values to numeric format"""
    df_parsed = df.copy()
    
    # Parse window values (e.g., "window_0.3s" -> 0.3)
    df_parsed['window_numeric'] = df_parsed['window'].str.extract(r'window_(\d+\.?\d*)s').astype(float)
    
    # Parse stride values
    stride_numeric = []
    stride_percentage = []
    
    for idx, row in df_parsed.iterrows():
        stride_val = row['stride']
        if row['eval_type'] == 'eval_by_0.05':
            # For eval_by_0.05: stride_0.125s -> 0.125
            numeric_val = float(re.search(r'stride_(\d+\.?\d*)', stride_val).group(1))
            stride_numeric.append(numeric_val)
            # Calculate percentage relative to window
            percentage = (numeric_val / row['window_numeric']) * 100
            stride_percentage.append(percentage)
        else:
            # For eval_percent: stride_30.0% -> 30.0
            if '%' in stride_val:
                percentage = float(re.search(r'stride_(\d+\.?\d*)%', stride_val).group(1))
                stride_percentage.append(percentage)
                # Calculate absolute value
                numeric_val = (percentage / 100) * row['window_numeric']
                stride_numeric.append(numeric_val)
            else:
                # Handle special cases
                percentage = float(re.search(r'stride_(\d+\.?\d*)', stride_val).group(1))
                stride_percentage.append(percentage)
                numeric_val = (percentage / 100) * row['window_numeric']
                stride_numeric.append(numeric_val)
    
    df_parsed['stride_numeric'] = stride_numeric
    df_parsed['stride_percentage'] = stride_percentage
    
    return df_parsed

def create_method_comparison_graphs(df):
    """Create trend graphs comparing all evaluation methods"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    output_dir = Path("fixed_smart_parallel_results/comprehensive_f1_trend_graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique window sizes
    window_sizes = sorted(df['window_numeric'].unique())
    
    # Group data by method
    methods_data = {}
    
    # 1. Word Evaluation
    word_eval_data = df[df['method'] == 'word_eval'].copy()
    if not word_eval_data.empty:
        methods_data['Word Evaluation'] = calculate_window_stats(word_eval_data, ['binary_f1', 'multiclass_f1'])
    
    # 2. Word-IoU Evaluation (use IoU 0.5 as representative threshold)
    word_iou_data = df[df['method'] == 'word_iou_eval'].copy()
    if not word_iou_data.empty:
        methods_data['Word-IoU Evaluation'] = calculate_window_stats(word_iou_data, ['binary_f1_iou_0.5', 'multiclass_f1_iou_0.5'])
    
    # 3. IoU Evaluation (use IoU 0.5 threshold for both binary and multiclass F1)
    iou_data = df[df['method'] == 'iou_eval'].copy()
    if not iou_data.empty:
        methods_data['IoU Evaluation'] = calculate_window_stats(iou_data, ['iou_binary_f1_0.5', 'iou_multiclass_f1_0.5'])
    
    # Create comprehensive comparison graph
    create_method_comparison_plot(methods_data, window_sizes, output_dir)
    
    # Create individual method graphs
    create_individual_method_graphs(methods_data, window_sizes, output_dir)
    
    # Create IoU threshold comparison graphs
    create_iou_threshold_graphs(df, window_sizes, output_dir)
    
    return methods_data

def calculate_window_stats(data, metrics):
    """Calculate statistics for each window size"""
    window_sizes = sorted(data['window_numeric'].unique())
    stats = []
    
    for window in window_sizes:
        window_data = data[data['window_numeric'] == window]
        
        stat_dict = {'window_size': window, 'count': len(window_data)}
        
        for metric in metrics:
            if metric in window_data.columns:
                values = window_data[metric].dropna()
                if len(values) > 0:
                    stat_dict.update({
                        f'{metric}_mean': values.mean(),
                        f'{metric}_max': values.max(),
                        f'{metric}_min': values.min(),
                        f'{metric}_std': values.std()
                    })
        
        stats.append(stat_dict)
    
    return pd.DataFrame(stats)

def create_method_comparison_plot(methods_data, window_sizes, output_dir):
    """Create comparison plot of all methods"""
    
    plt.figure(figsize=(15, 10))
    
    # Create subplots for binary and multiclass/IoU
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # ============ BINARY F1 COMPARISON ============
    color_idx = 0
    for method_name, stats_df in methods_data.items():
        if stats_df.empty:
            continue
            
        # Find binary F1 column
        binary_cols = [col for col in stats_df.columns if 'binary_f1' in col and '_mean' in col]
        if binary_cols:
            binary_col = binary_cols[0]
            ax1.plot(stats_df['window_size'], stats_df[binary_col], 
                    marker='o', linewidth=3, markersize=8, 
                    color=colors[color_idx % len(colors)], 
                    label=f'{method_name}')
            
            # Add error bars if std available
            std_col = binary_col.replace('_mean', '_std')
            if std_col in stats_df.columns:
                ax1.fill_between(stats_df['window_size'], 
                               stats_df[binary_col] - stats_df[std_col],
                               stats_df[binary_col] + stats_df[std_col],
                               alpha=0.2, color=colors[color_idx % len(colors)])
        
        color_idx += 1
    
    ax1.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Binary F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title('Binary F1 Score Comparison Across Evaluation Methods', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10)
    ax1.set_xticks(window_sizes)
    
    # ============ MULTICLASS F1 COMPARISON (F1 SCORES ONLY) ============
    color_idx = 0
    for method_name, stats_df in methods_data.items():
        if stats_df.empty:
            continue
            
        # Find multiclass F1 columns for each method
        multiclass_f1_cols = []
        
        if 'Word Evaluation' in method_name:
            # Word Evaluation: look for standard multiclass F1
            multiclass_f1_cols = [col for col in stats_df.columns if 
                                 'multiclass_f1' in col and '_mean' in col and 'iou' not in col.lower()]
        elif 'Word-IoU Evaluation' in method_name:
            # Word-IoU Evaluation: look for IoU threshold multiclass F1
            multiclass_f1_cols = [col for col in stats_df.columns if 
                                 'multiclass_f1_iou' in col and '_mean' in col]
        elif 'IoU Evaluation' in method_name:
            # IoU Evaluation: look for multiclass F1 from IoU thresholds
            multiclass_f1_cols = [col for col in stats_df.columns if 
                                 'iou_multiclass_f1' in col and '_mean' in col]
        
        if multiclass_f1_cols:
            col = multiclass_f1_cols[0]  # Use first available F1 column
            
            # Determine appropriate label based on method
            if 'Word Evaluation' in method_name:
                label_name = f"{method_name} (Multiclass F1)"
            elif 'Word-IoU Evaluation' in method_name:
                label_name = f"{method_name} (Multiclass F1)"
            elif 'IoU Evaluation' in method_name:
                label_name = f"{method_name} (Multiclass F1)"
                
            ax2.plot(stats_df['window_size'], stats_df[col], 
                    marker='s', linewidth=3, markersize=8, 
                    color=colors[color_idx % len(colors)], 
                    label=label_name)
            
            # Add error bars if std available
            std_col = col.replace('_mean', '_std')
            if std_col in stats_df.columns:
                ax2.fill_between(stats_df['window_size'], 
                               stats_df[col] - stats_df[std_col],
                               stats_df[col] + stats_df[std_col],
                               alpha=0.2, color=colors[color_idx % len(colors)])
        
        color_idx += 1
    
    ax2.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Multiclass F1 Score Comparison Across Evaluation Methods', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    ax2.set_xticks(window_sizes)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_method_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_individual_method_graphs(methods_data, window_sizes, output_dir):
    """Create individual graphs for each method"""
    
    for method_name, stats_df in methods_data.items():
        if stats_df.empty:
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Plot all available metrics for this method
        metric_cols = [col for col in stats_df.columns if '_mean' in col and col != 'window_size']
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A2B']
        
        for i, col in enumerate(metric_cols):
            metric_name = col.replace('_mean', '').replace('_', ' ').title()
            plt.plot(stats_df['window_size'], stats_df[col], 
                    marker='o', linewidth=3, markersize=8, 
                    color=colors[i % len(colors)], 
                    label=metric_name)
            
            # Add error bars
            std_col = col.replace('_mean', '_std')
            if std_col in stats_df.columns:
                plt.fill_between(stats_df['window_size'], 
                               stats_df[col] - stats_df[std_col],
                               stats_df[col] + stats_df[std_col],
                               alpha=0.2, color=colors[i % len(colors)])
        
        plt.xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title(f'{method_name} - Performance Trends', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=10)
        plt.xticks(window_sizes)
        
        # Save individual method graph
        safe_filename = method_name.lower().replace(' ', '_').replace('-', '_')
        plt.savefig(output_dir / f"{safe_filename}_trends.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_iou_threshold_graphs(df, window_sizes, output_dir):
    """Create graphs comparing different IoU thresholds"""
    
    thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
    
    # Word-IoU Threshold Comparison
    word_iou_data = df[df['method'] == 'word_iou_eval'].copy()
    if not word_iou_data.empty:
        plt.figure(figsize=(15, 10))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds)))
        
        # Binary F1 by IoU threshold
        for i, threshold in enumerate(thresholds):
            col_name = f'binary_f1_iou_{threshold}'
            if col_name in word_iou_data.columns:
                stats = calculate_window_stats(word_iou_data, [col_name])
                if not stats.empty:
                    ax1.plot(stats['window_size'], stats[f'{col_name}_mean'], 
                            marker='o', linewidth=2, markersize=6,
                            color=colors[i], label=f'IoU ≥ {threshold}')
        
        ax1.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Binary F1 Score', fontsize=12, fontweight='bold')
        ax1.set_title('Word-IoU Evaluation: Binary F1 by IoU Threshold', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        ax1.set_xticks(window_sizes)
        
        # Multiclass F1 by IoU threshold
        for i, threshold in enumerate(thresholds):
            col_name = f'multiclass_f1_iou_{threshold}'
            if col_name in word_iou_data.columns:
                stats = calculate_window_stats(word_iou_data, [col_name])
                if not stats.empty:
                    ax2.plot(stats['window_size'], stats[f'{col_name}_mean'], 
                            marker='s', linewidth=2, markersize=6,
                            color=colors[i], label=f'IoU ≥ {threshold}')
        
        ax2.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Multiclass F1 Score', fontsize=12, fontweight='bold')
        ax2.set_title('Word-IoU Evaluation: Multiclass F1 by IoU Threshold', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        ax2.set_xticks(window_sizes)
        
        plt.tight_layout()
        plt.savefig(output_dir / "word_iou_threshold_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_table(methods_data, output_dir):
    """Create summary tables for all methods"""
    
    summary_file = output_dir / "comprehensive_f1_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("=== COMPREHENSIVE F1 SCORE ANALYSIS (EXCLUDING WINDOW F1) ===\n\n")
        
        for method_name, stats_df in methods_data.items():
            if stats_df.empty:
                continue
                
            f.write(f"=== {method_name.upper()} ===\n")
            f.write("-" * 50 + "\n")
            
            # Find best performing configurations
            metric_cols = [col for col in stats_df.columns if '_mean' in col and col != 'window_size']
            
            for col in metric_cols:
                if not stats_df[col].isna().all():
                    best_idx = stats_df[col].idxmax()
                    metric_name = col.replace('_mean', '').replace('_', ' ').title()
                    
                    f.write(f"{metric_name}:\n")
                    f.write(f"  Best Window: {stats_df.loc[best_idx, 'window_size']}s\n")
                    f.write(f"  Best Score: {stats_df.loc[best_idx, col]:.4f}\n")
                    
                    # Calculate correlation
                    corr = stats_df['window_size'].corr(stats_df[col])
                    if abs(corr) > 0.3:
                        trend = "Positive" if corr > 0 else "Negative"
                        strength = "Strong" if abs(corr) > 0.7 else "Moderate"
                        f.write(f"  Correlation: {corr:.3f} ({strength} {trend})\n")
                    
                    f.write("\n")
            
            f.write("\n")
    
    print(f"📄 Comprehensive summary saved to: {summary_file}")

def main():
    print("=== CREATING COMPREHENSIVE F1 TREND GRAPHS (EXCLUDING WINDOW F1) ===\n")
    
    # Extract F1 scores from all evaluation methods
    print("📊 Extracting F1 scores from evaluation results...")
    df_raw = extract_f1_from_note_files()
    
    if df_raw.empty:
        print("❌ No evaluation data found!")
        return
    
    print(f"✅ Extracted data from {len(df_raw)} configurations")
    print(f"📋 Methods found: {df_raw['method'].unique()}")
    print(f"📋 Evaluation types: {df_raw['eval_type'].unique()}")
    
    # Parse window and stride values
    df = parse_window_stride(df_raw)
    
    # Create comprehensive graphs
    print("📈 Creating comprehensive trend graphs...")
    methods_data = create_method_comparison_graphs(df)
    
    # Create summary
    output_dir = Path("fixed_smart_parallel_results/comprehensive_f1_trend_graphs")
    create_summary_table(methods_data, output_dir)
    
    print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"📁 All graphs and analysis saved to: {output_dir}")
    print(f"📊 Generated files:")
    print(f"   • comprehensive_method_comparison.png (All methods comparison)")
    print(f"   • word_evaluation_trends.png (Word evaluation only)")
    print(f"   • word_iou_evaluation_trends.png (Word-IoU evaluation only)")
    print(f"   • iou_evaluation_trends.png (IoU evaluation only)")
    print(f"   • word_iou_threshold_comparison.png (IoU threshold comparison)")
    print(f"   • comprehensive_f1_summary.txt (Detailed analysis)")
    
    # Print key findings
    print(f"\n💡 KEY FINDINGS (Methods analyzed):")
    for method_name, stats_df in methods_data.items():
        if not stats_df.empty:
            print(f"🔍 {method_name}: {len(stats_df)} window sizes analyzed")

if __name__ == "__main__":
    main()

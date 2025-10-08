#!/usr/bin/env python3
"""
📈 IoU Evaluation F1 Score Line Graph Generator
Create line graphs showing F1 scores vs window sizes from comprehensive IoU evaluation results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
import re

def extract_f1_scores_from_results(base_dir="comprehensive_iou_evaluation"):
    """Extract F1 scores from all evaluation result files"""
    
    results = []
    base_path = Path(base_dir)
    
    print("📊 Extracting F1 scores from IoU evaluation results...")
    
    # Process both eval_by_0.05 and eval_percent
    for eval_type in ['eval_by_0.05', 'eval_percent']:
        eval_path = base_path / eval_type / 'word'
        
        if not eval_path.exists():
            print(f"⚠️  Path not found: {eval_path}")
            continue
        
        # Process each window size directory
        for window_dir in eval_path.iterdir():
            if not window_dir.is_dir() or not window_dir.name.startswith('window_'):
                continue
            
            # Extract window size
            window_match = re.match(r'window_(\d+\.?\d*)s', window_dir.name)
            if not window_match:
                continue
            window_size = float(window_match.group(1))
            
            # Process each stride directory
            for stride_dir in window_dir.iterdir():
                if not stride_dir.is_dir() or not stride_dir.name.startswith('stride_'):
                    continue
                
                # Extract stride info
                stride_match = re.match(r'stride_(\d+\.?\d*)(s|%|pct)', stride_dir.name)
                if not stride_match:
                    continue
                
                stride_value = float(stride_match.group(1))
                stride_type = 'absolute' if stride_match.group(2) == 's' else 'percentage'
                
                # Read the note.txt file to extract metrics
                note_file = stride_dir / 'note.txt'
                if not note_file.exists():
                    continue
                
                try:
                    with open(note_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract metrics
                    overall_mean_iou = None
                    binary_f1 = None
                    multiclass_f1 = None
                    best_iou_threshold_f1 = None
                    
                    # Parse metrics
                    for line in content.split('\n'):
                        if 'Overall Mean IoU:' in line:
                            try:
                                iou_part = line.split(':')[1].split('(')[0].strip()
                                overall_mean_iou = float(iou_part)
                            except:
                                pass
                        elif 'Traditional Binary F1:' in line:
                            try:
                                binary_f1 = float(line.split(':')[1].strip())
                            except:
                                pass
                        elif 'Traditional Multiclass F1:' in line:
                            try:
                                multiclass_f1 = float(line.split(':')[1].strip())
                            except:
                                pass
                    
                    # Extract best IoU threshold F1 from the threshold table
                    threshold_lines = []
                    in_threshold_section = False
                    for line in content.split('\n'):
                        if '=== IoU THRESHOLD ANALYSIS ===' in line:
                            in_threshold_section = True
                            continue
                        elif in_threshold_section and line.strip().startswith('==='):
                            break
                        elif in_threshold_section and '|' in line and not 'Threshold' in line and not '-------' in line:
                            threshold_lines.append(line)
                    
                    # Find best F1 from thresholds
                    best_f1 = 0.0
                    for tline in threshold_lines:
                        try:
                            parts = [p.strip() for p in tline.split('|')]
                            if len(parts) >= 4:
                                f1_score = float(parts[3])
                                if f1_score > best_f1:
                                    best_f1 = f1_score
                        except:
                            continue
                    
                    best_iou_threshold_f1 = best_f1 if best_f1 > 0 else None
                    
                    # Store results
                    if overall_mean_iou is not None and binary_f1 is not None:
                        results.append({
                            'eval_type': eval_type,
                            'window_size': window_size,
                            'stride_value': stride_value,
                            'stride_type': stride_type,
                            'stride_label': f"{stride_value}{'%' if stride_type == 'percentage' else 's'}",
                            'overall_mean_iou': overall_mean_iou,
                            'binary_f1': binary_f1,
                            'multiclass_f1': multiclass_f1 if multiclass_f1 is not None else 0.0,
                            'best_iou_threshold_f1': best_iou_threshold_f1 if best_iou_threshold_f1 is not None else 0.0
                        })
                
                except Exception as e:
                    print(f"⚠️  Error processing {note_file}: {e}")
                    continue
    
    print(f"✅ Extracted {len(results)} configurations")
    return pd.DataFrame(results)

def create_f1_line_graphs(df, output_dir="iou_f1_line_graphs"):
    """Create comprehensive F1 score line graphs"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Binary F1 vs Window Size (by eval_type)
    plt.figure(figsize=(12, 8))
    
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type]
        
        # Group by window size and calculate mean, min, max
        window_stats = eval_data.groupby('window_size')['binary_f1'].agg(['mean', 'min', 'max', 'std']).reset_index()
        
        label = 'Fixed Time Stride' if eval_type == 'eval_by_0.05' else 'Percentage Stride'
        color = 'blue' if eval_type == 'eval_by_0.05' else 'red'
        
        plt.plot(window_stats['window_size'], window_stats['mean'], 
                marker='o', linewidth=2, markersize=6, label=f'{label} (Mean)', color=color)
        
        # Add error bars for standard deviation
        plt.fill_between(window_stats['window_size'], 
                        window_stats['mean'] - window_stats['std'], 
                        window_stats['mean'] + window_stats['std'], 
                        alpha=0.2, color=color)
    
    plt.xlabel('Window Size (seconds)', fontsize=12)
    plt.ylabel('Binary F1 Score', fontsize=12)
    plt.title('IoU Evaluation: Binary F1 Score vs Window Size\n(Traditional Classification Metrics)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/binary_f1_vs_window_size.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Best IoU Threshold F1 vs Window Size
    plt.figure(figsize=(12, 8))
    
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type]
        
        # Group by window size and calculate mean
        window_stats = eval_data.groupby('window_size')['best_iou_threshold_f1'].agg(['mean', 'std']).reset_index()
        
        label = 'Fixed Time Stride' if eval_type == 'eval_by_0.05' else 'Percentage Stride'
        color = 'green' if eval_type == 'eval_by_0.05' else 'orange'
        
        plt.plot(window_stats['window_size'], window_stats['mean'], 
                marker='s', linewidth=2, markersize=6, label=f'{label} (Mean)', color=color)
        
        # Add error bars
        plt.fill_between(window_stats['window_size'], 
                        window_stats['mean'] - window_stats['std'], 
                        window_stats['mean'] + window_stats['std'], 
                        alpha=0.2, color=color)
    
    plt.xlabel('Window Size (seconds)', fontsize=12)
    plt.ylabel('Best IoU Threshold F1 Score', fontsize=12)
    plt.title('IoU Evaluation: Best IoU Threshold F1 Score vs Window Size\n(Optimal IoU Threshold Performance)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/best_iou_f1_vs_window_size.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Mean IoU vs Window Size
    plt.figure(figsize=(12, 8))
    
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type]
        
        # Group by window size and calculate mean
        window_stats = eval_data.groupby('window_size')['overall_mean_iou'].agg(['mean', 'std']).reset_index()
        
        label = 'Fixed Time Stride' if eval_type == 'eval_by_0.05' else 'Percentage Stride'
        color = 'purple' if eval_type == 'eval_by_0.05' else 'brown'
        
        plt.plot(window_stats['window_size'], window_stats['mean'], 
                marker='^', linewidth=2, markersize=6, label=f'{label} (Mean)', color=color)
        
        # Add error bars
        plt.fill_between(window_stats['window_size'], 
                        window_stats['mean'] - window_stats['std'], 
                        window_stats['mean'] + window_stats['std'], 
                        alpha=0.2, color=color)
    
    plt.xlabel('Window Size (seconds)', fontsize=12)
    plt.ylabel('Mean IoU', fontsize=12)
    plt.title('IoU Evaluation: Mean IoU vs Window Size\n(Temporal Overlap Quality)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mean_iou_vs_window_size.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Combined comparison graph
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Binary F1
    ax = axes[0, 0]
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type]
        window_stats = eval_data.groupby('window_size')['binary_f1'].mean().reset_index()
        label = 'Fixed Time' if eval_type == 'eval_by_0.05' else 'Percentage'
        ax.plot(window_stats['window_size'], window_stats['binary_f1'], 
               marker='o', linewidth=2, label=label)
    ax.set_xlabel('Window Size (s)')
    ax.set_ylabel('Binary F1 Score')
    ax.set_title('Binary F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Best IoU Threshold F1
    ax = axes[0, 1]
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type]
        window_stats = eval_data.groupby('window_size')['best_iou_threshold_f1'].mean().reset_index()
        label = 'Fixed Time' if eval_type == 'eval_by_0.05' else 'Percentage'
        ax.plot(window_stats['window_size'], window_stats['best_iou_threshold_f1'], 
               marker='s', linewidth=2, label=label)
    ax.set_xlabel('Window Size (s)')
    ax.set_ylabel('Best IoU Threshold F1')
    ax.set_title('Best IoU Threshold F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean IoU
    ax = axes[1, 0]
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type]
        window_stats = eval_data.groupby('window_size')['overall_mean_iou'].mean().reset_index()
        label = 'Fixed Time' if eval_type == 'eval_by_0.05' else 'Percentage'
        ax.plot(window_stats['window_size'], window_stats['overall_mean_iou'], 
               marker='^', linewidth=2, label=label)
    ax.set_xlabel('Window Size (s)')
    ax.set_ylabel('Mean IoU')
    ax.set_title('Mean IoU')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Multiclass F1
    ax = axes[1, 1]
    for eval_type in df['eval_type'].unique():
        eval_data = df[df['eval_type'] == eval_type]
        window_stats = eval_data.groupby('window_size')['multiclass_f1'].mean().reset_index()
        label = 'Fixed Time' if eval_type == 'eval_by_0.05' else 'Percentage'
        ax.plot(window_stats['window_size'], window_stats['multiclass_f1'], 
               marker='d', linewidth=2, label=label)
    ax.set_xlabel('Window Size (s)')
    ax.set_ylabel('Multiclass F1 Score')
    ax.set_title('Multiclass F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('IoU Evaluation: Comprehensive Performance vs Window Size', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comprehensive_f1_vs_window_size.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_dir

def generate_summary_statistics(df, output_dir="iou_f1_line_graphs"):
    """Generate summary statistics"""
    
    # Best configurations by different metrics
    best_configs = {}
    
    # Best Binary F1
    best_binary_f1 = df.loc[df['binary_f1'].idxmax()]
    best_configs['binary_f1'] = best_binary_f1
    
    # Best IoU Threshold F1
    best_iou_f1 = df.loc[df['best_iou_threshold_f1'].idxmax()]
    best_configs['best_iou_threshold_f1'] = best_iou_f1
    
    # Best Mean IoU
    best_mean_iou = df.loc[df['overall_mean_iou'].idxmax()]
    best_configs['mean_iou'] = best_mean_iou
    
    # Window size analysis
    window_analysis = df.groupby('window_size').agg({
        'binary_f1': ['mean', 'std', 'max'],
        'best_iou_threshold_f1': ['mean', 'std', 'max'],
        'overall_mean_iou': ['mean', 'std', 'max']
    }).round(4)
    
    # Save summary
    summary_path = f"{output_dir}/f1_analysis_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=== IoU EVALUATION F1 SCORE ANALYSIS SUMMARY ===\n\n")
        
        f.write("📊 BEST CONFIGURATIONS:\n")
        f.write("-" * 40 + "\n")
        
        f.write(f"🏆 Best Binary F1 Score: {best_configs['binary_f1']['binary_f1']:.3f}\n")
        f.write(f"   Configuration: {best_configs['binary_f1']['eval_type']}, window_{best_configs['binary_f1']['window_size']}s, stride_{best_configs['binary_f1']['stride_label']}\n")
        f.write(f"   Mean IoU: {best_configs['binary_f1']['overall_mean_iou']:.3f}\n\n")
        
        f.write(f"🎯 Best IoU Threshold F1 Score: {best_configs['best_iou_threshold_f1']['best_iou_threshold_f1']:.3f}\n")
        f.write(f"   Configuration: {best_configs['best_iou_threshold_f1']['eval_type']}, window_{best_configs['best_iou_threshold_f1']['window_size']}s, stride_{best_configs['best_iou_threshold_f1']['stride_label']}\n")
        f.write(f"   Mean IoU: {best_configs['best_iou_threshold_f1']['overall_mean_iou']:.3f}\n\n")
        
        f.write(f"🔗 Best Mean IoU: {best_configs['mean_iou']['overall_mean_iou']:.3f}\n")
        f.write(f"   Configuration: {best_configs['mean_iou']['eval_type']}, window_{best_configs['mean_iou']['window_size']}s, stride_{best_configs['mean_iou']['stride_label']}\n")
        f.write(f"   Binary F1: {best_configs['mean_iou']['binary_f1']:.3f}\n\n")
        
        f.write("📈 WINDOW SIZE ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write("Window | Binary F1 (Mean±Std) | IoU F1 (Mean±Std) | Mean IoU (Mean±Std)\n")
        f.write("-------|---------------------|-------------------|--------------------\n")
        
        for window_size in sorted(df['window_size'].unique()):
            stats = window_analysis.loc[window_size]
            f.write(f" {window_size:4.1f}s | {stats[('binary_f1', 'mean')]:.3f}±{stats[('binary_f1', 'std')]:.3f} | "
                   f"{stats[('best_iou_threshold_f1', 'mean')]:.3f}±{stats[('best_iou_threshold_f1', 'std')]:.3f} | "
                   f"{stats[('overall_mean_iou', 'mean')]:.3f}±{stats[('overall_mean_iou', 'std')]:.3f}\n")
    
    print(f"📋 Summary saved to: {summary_path}")
    return summary_path

def main():
    """Main function"""
    print("📈 Generating IoU Evaluation F1 Score Line Graphs")
    print("=" * 50)
    
    # Extract data
    df = extract_f1_scores_from_results()
    
    if df.empty:
        print("❌ No data found! Please check if comprehensive_iou_evaluation directory exists.")
        return
    
    print(f"📊 Processing {len(df)} configurations:")
    print(f"   • eval_by_0.05: {len(df[df['eval_type'] == 'eval_by_0.05'])} configs")
    print(f"   • eval_percent: {len(df[df['eval_type'] == 'eval_percent'])} configs")
    print(f"   • Window sizes: {sorted(df['window_size'].unique())}")
    
    # Create graphs
    output_dir = create_f1_line_graphs(df)
    print(f"\n📊 Line graphs created in: {output_dir}")
    
    # Generate summary
    summary_path = generate_summary_statistics(df, output_dir)
    
    # Save processed data
    df.to_csv(f"{output_dir}/iou_f1_analysis_data.csv", index=False)
    print(f"📁 Analysis data saved to: {output_dir}/iou_f1_analysis_data.csv")
    
    print(f"\n🎉 Analysis complete!")
    print(f"📁 All results in: {output_dir}")
    print(f"📊 Generated graphs:")
    print(f"   • binary_f1_vs_window_size.png")
    print(f"   • best_iou_f1_vs_window_size.png") 
    print(f"   • mean_iou_vs_window_size.png")
    print(f"   • comprehensive_f1_vs_window_size.png")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Create Line Graphs for eval_IoU_word_eval_sep F1 Scores
Extract and visualize F1 scores from the older evaluation pipeline for comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import os

def parse_window_stride_from_path(config_path):
    """Parse window and stride from configuration path"""
    # Extract window (e.g., "window_1.2s" -> 1.2)
    window_match = re.search(r'window_(\d+\.?\d*)s', config_path)
    window_numeric = float(window_match.group(1)) if window_match else None
    
    # Extract stride 
    stride_match = re.search(r'stride_(\d+\.?\d*)s', config_path)
    if stride_match:
        stride_val = float(stride_match.group(1))
        stride_numeric = stride_val
        stride_percentage = (stride_val / window_numeric * 100) if window_numeric else 100
        return window_numeric, stride_numeric, stride_percentage
    
    return window_numeric, None, None

def extract_f1_from_old_eval(notes_file):
    """Extract F1 scores and performance metrics from the old evaluation format based on table structure"""
    if not os.path.exists(notes_file):
        return {}
    
    try:
        with open(notes_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        scores = {}
        
        # Extract metrics based on the tabular data structure shown in the image
        # Looking for patterns that match the table format with decimal values
        
        # Extract F1 scores (looking for patterns like "F1: 0.xxxx" or similar)
        f1_patterns = [
            r'F1[- ]*[Ss]core:?\s*([\d.]+)',
            r'F1:?\s*([\d.]+)',
            r'f1-score\s+([\d.]+)',
        ]
        
        for pattern in f1_patterns:
            f1_match = re.search(pattern, content, re.IGNORECASE)
            if f1_match:
                scores['f1_score'] = float(f1_match.group(1))
                break
        
        # Extract accuracy metrics
        accuracy_patterns = [
            r'Overall Accuracy[^:]*:\s*([\d.]+)',
            r'Accuracy:?\s*([\d.]+)',
            r'accuracy\s+([\d.]+)',
        ]
        
        for pattern in accuracy_patterns:
            acc_match = re.search(pattern, content, re.IGNORECASE)
            if acc_match:
                scores['accuracy'] = float(acc_match.group(1))
                break
        
        # Extract precision and recall
        precision_match = re.search(r'Precision:?\s*([\d.]+)', content, re.IGNORECASE)
        if precision_match:
            scores['precision'] = float(precision_match.group(1))
            
        recall_match = re.search(r'Recall:?\s*([\d.]+)', content, re.IGNORECASE)
        if recall_match:
            scores['recall'] = float(recall_match.group(1))
        
        # Extract balanced accuracy
        balanced_acc_match = re.search(r'Balanced Accuracy:?\s*([\d.]+)', content, re.IGNORECASE)
        if balanced_acc_match:
            scores['balanced_accuracy'] = float(balanced_acc_match.group(1))
        
        # Extract weighted averages from classification report
        weighted_patterns = [
            r'weighted avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
            r'weighted average.*?(\d+\.\d+)',
        ]
        
        for pattern in weighted_patterns:
            weighted_match = re.search(pattern, content, re.IGNORECASE)
            if weighted_match:
                scores['weighted_f1'] = float(weighted_match.group(1))
                break
        
        # Extract macro averages
        macro_patterns = [
            r'macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
            r'macro average.*?(\d+\.\d+)',
        ]
        
        for pattern in macro_patterns:
            macro_match = re.search(pattern, content, re.IGNORECASE)
            if macro_match:
                scores['macro_f1'] = float(macro_match.group(1))
                break
        
        # Extract any decimal values that might represent performance metrics
        # This will help capture metrics from the tabular format shown in the image
        decimal_values = re.findall(r'\b(\d+\.\d{2,4})\b', content)
        if decimal_values:
            # Convert to float and filter reasonable performance metric values (0.0 to 1.0 or 0 to 100)
            valid_metrics = []
            for val in decimal_values:
                float_val = float(val)
                if 0.0 <= float_val <= 1.0:  # Normalized metrics
                    valid_metrics.append(float_val)
                elif 0.0 <= float_val <= 100.0:  # Percentage metrics
                    valid_metrics.append(float_val / 100.0)  # Normalize to 0-1
            
            if valid_metrics:
                # Use statistics from extracted metrics
                scores['mean_performance'] = np.mean(valid_metrics)
                scores['max_performance'] = np.max(valid_metrics)
                scores['min_performance'] = np.min(valid_metrics)
        
        # If no specific metrics found, try to extract from any numerical table format
        if not scores:
            # Look for lines with multiple decimal values (table rows)
            table_rows = re.findall(r'((?:\d+\.\d+\s+){2,})', content)
            if table_rows:
                all_values = []
                for row in table_rows:
                    values = [float(x) for x in re.findall(r'\d+\.\d+', row)]
                    all_values.extend([v for v in values if 0.0 <= v <= 1.0])
                
                if all_values:
                    scores['table_mean'] = np.mean(all_values)
                    scores['table_max'] = np.max(all_values)
        
        return scores
        
    except Exception as e:
        print(f"Error reading {notes_file}: {e}")
        return {}

def extract_old_eval_data():
    """Extract data from eval_IoU_word_eval_sep directory"""
    
    base_dir = Path("evaluation_results/eval_IoU_word_eval_sep/eval_by_0.05")
    
    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return []
    
    print(f"📊 Processing eval_IoU_word_eval_sep data...")
    
    all_configs = []
    
    # Traverse window directories
    for window_dir in base_dir.iterdir():
        if not window_dir.is_dir() or not window_dir.name.startswith("window_"):
            continue
            
        # Traverse stride directories  
        for stride_dir in window_dir.iterdir():
            if not stride_dir.is_dir() or not stride_dir.name.startswith("stride_"):
                continue
            
            # Look for notes file
            notes_file = stride_dir / "note.txt"
            if not notes_file.exists():
                continue
            
            # Extract configuration info
            config_path = f"{window_dir.name}/{stride_dir.name}"
            window_numeric, stride_numeric, stride_percentage = parse_window_stride_from_path(config_path)
            
            # Extract F1 scores
            scores = extract_f1_from_old_eval(notes_file)
            
            if scores:
                config_data = {
                    'eval_type': 'eval_by_0.05_old',
                    'method': 'eval_IoU_word_eval_sep',
                    'window': window_dir.name,
                    'stride': stride_dir.name,
                    'window_numeric': window_numeric,
                    'stride_numeric': stride_numeric,
                    'stride_percentage': stride_percentage,
                    'configuration': config_path,
                    **scores  # Add all extracted scores
                }
                all_configs.append(config_data)
    
    print(f"✅ Found {len(all_configs)} configurations for eval_IoU_word_eval_sep")
    return all_configs

def create_comparison_graphs():
    """Create comparison line graphs between old and new evaluation methods"""
    
    # Extract data from old evaluation
    old_eval_data = extract_old_eval_data()
    
    if not old_eval_data:
        print("❌ No data found in eval_IoU_word_eval_sep")
        return
    
    # Convert to DataFrame
    df_old = pd.DataFrame(old_eval_data)
    
    # Create output directory
    output_dir = Path("comparison_graphs")
    output_dir.mkdir(exist_ok=True)
    
    print(f"📊 Creating comparison graphs...")
    print(f"📊 Old eval data shape: {df_old.shape}")
    print(f"📊 Metrics available: {[col for col in df_old.columns if col not in ['eval_type', 'method', 'window', 'stride', 'window_numeric', 'stride_numeric', 'stride_percentage', 'configuration']]}")
    
    # Group by window size for analysis - dynamically detect numeric columns
    numeric_columns = []
    for col in df_old.columns:
        if col not in ['eval_type', 'method', 'window', 'stride', 'window_numeric', 'stride_numeric', 'stride_percentage', 'configuration']:
            if df_old[col].dtype in ['float64', 'int64'] and not df_old[col].isna().all():
                numeric_columns.append(col)
    
    print(f"📊 Detected numeric metrics: {numeric_columns}")
    
    # Create aggregation dictionary dynamically
    agg_dict = {}
    for col in numeric_columns:
        agg_dict[col] = ['mean', 'std', 'count']
    
    if agg_dict:
        df_old_grouped = df_old.groupby('window_numeric').agg(agg_dict).reset_index()
    else:
        print("⚠️ No numeric metrics found for aggregation")
        return
    
    # Flatten column names
    df_old_grouped.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_old_grouped.columns]
    df_old_grouped = df_old_grouped.rename(columns={'window_numeric_': 'window_numeric'})
    
    # Create multiple comparison graphs based on extracted metrics
    # First check what metrics are available
    available_metrics = []
    for col in df_old.columns:
        if col.endswith('_mean') and col != 'window_numeric_':
            metric_name = col.replace('_mean', '')
            available_metrics.append(metric_name)
    
    # Define metric mappings with better titles
    metric_titles = {
        'f1_score': 'F1 Score',
        'accuracy': 'Accuracy',
        'precision': 'Precision', 
        'recall': 'Recall',
        'balanced_accuracy': 'Balanced Accuracy',
        'weighted_f1': 'Weighted F1 Score',
        'macro_f1': 'Macro F1 Score',
        'mean_performance': 'Mean Performance',
        'max_performance': 'Max Performance',
        'table_mean': 'Table Mean Performance'
    }
    
    # Create list of metrics to plot
    metrics_to_plot = []
    for metric in available_metrics:
        title = metric_titles.get(metric, metric.replace('_', ' ').title())
        metrics_to_plot.append((metric, title))
    
    # Fallback metrics if none detected
    if not metrics_to_plot:
        metrics_to_plot = [
            ('f1_score', 'F1 Score'),
            ('accuracy', 'Accuracy'),
            ('precision', 'Precision'),
            ('recall', 'Recall'),
            ('balanced_accuracy', 'Balanced Accuracy')
        ]
    
    # Set style
    plt.style.use('seaborn-v0_8')
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD']
    
    for idx, (metric, title) in enumerate(metrics_to_plot):
        if f'{metric}_mean' not in df_old_grouped.columns:
            print(f"⚠️ Metric {metric} not available, skipping...")
            continue
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot old evaluation method
        ax.plot(df_old_grouped['window_numeric'], df_old_grouped[f'{metric}_mean'], 
               marker='o', linewidth=3, markersize=8, 
               color=colors[idx % len(colors)], 
               label=f'eval_IoU_word_eval_sep ({title})')
        
        # Add error bars
        if f'{metric}_std' in df_old_grouped.columns:
            ax.fill_between(df_old_grouped['window_numeric'], 
                           df_old_grouped[f'{metric}_mean'] - df_old_grouped[f'{metric}_std'],
                           df_old_grouped[f'{metric}_mean'] + df_old_grouped[f'{metric}_std'],
                           alpha=0.2, color=colors[idx % len(colors)])
        
        # Customize plot
        ax.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} vs Window Size\n(eval_IoU_word_eval_sep Method)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Set reasonable y-limits
        y_values = df_old_grouped[f'{metric}_mean'].dropna()
        if len(y_values) > 0:
            y_min, y_max = y_values.min(), y_values.max()
            margin = (y_max - y_min) * 0.1
            ax.set_ylim(max(0, y_min - margin), min(1, y_max + margin))
        
        # Save plot
        filename = output_dir / f"old_eval_{metric}_vs_window_size.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved: {filename}")
    
    # Create summary statistics
    print(f"\n📊 SUMMARY STATISTICS FOR eval_IoU_word_eval_sep:")
    print("="*60)
    for metric, title in metrics_to_plot:
        if f'{metric}_mean' in df_old_grouped.columns:
            values = df_old_grouped[f'{metric}_mean'].dropna()
            if len(values) > 0:
                print(f"{title}:")
                print(f"  Mean: {values.mean():.4f}")
                print(f"  Std:  {values.std():.4f}")
                print(f"  Min:  {values.min():.4f}")
                print(f"  Max:  {values.max():.4f}")
                print(f"  Range: {values.min():.4f} - {values.max():.4f}")
                print()
    
    # Save raw data
    csv_file = output_dir / "eval_IoU_word_eval_sep_summary.csv"
    df_old_grouped.to_csv(csv_file, index=False)
    print(f"💾 Saved summary data: {csv_file}")
    
    # Create detailed data file
    detailed_csv = output_dir / "eval_IoU_word_eval_sep_detailed.csv"
    df_old.to_csv(detailed_csv, index=False)
    print(f"💾 Saved detailed data: {detailed_csv}")
    
    print(f"\n🎉 Comparison graphs created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total configurations analyzed: {len(df_old)}")
    print(f"📊 Window sizes: {sorted(df_old['window_numeric'].unique())}")

if __name__ == "__main__":
    print("=== CREATING COMPARISON GRAPHS FOR eval_IoU_word_eval_sep ===\n")
    create_comparison_graphs()

#!/usr/bin/env python3
"""
Fresh Correlation Analysis - Recalculate correlations between window size, stride, and F1 scores
Provides accurate correlation coefficients for all evaluation methods
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

def load_comprehensive_data():
    """Load data from both existing analysis and comprehensive extraction"""
    
    # Load original correlation data
    data_path = Path("fixed_smart_parallel_results/correlation_analysis/complete_analysis_data.csv")
    if data_path.exists():
        df_original = pd.read_csv(data_path)
        print(f"✅ Loaded original data: {len(df_original)} configurations")
    else:
        df_original = pd.DataFrame()
        print("❌ Original correlation data not found")
    
    return df_original

def extract_comprehensive_f1_data():
    """Extract F1 data from all evaluation method files"""
    base_dir = Path("fixed_smart_parallel_results")
    
    all_results = []
    
    # Process both eval_by_0.05 and eval_percent directories
    for eval_type in ["eval_by_0.05", "eval_percent"]:
        eval_dir = base_dir / eval_type / eval_type
        
        if not eval_dir.exists():
            continue
            
        print(f"Processing {eval_type}...")
        
        # Extract from all evaluation methods
        methods = {
            'word_eval': extract_word_eval_f1,
            'word_iou_eval': extract_word_iou_eval_f1,
            'iou_eval': extract_iou_eval_f1
        }
        
        for method_name, extract_func in methods.items():
            method_dir = eval_dir / method_name
            if not method_dir.exists():
                continue
                
            for window_dir in method_dir.iterdir():
                if window_dir.is_dir() and window_dir.name.startswith("window_"):
                    for stride_dir in window_dir.iterdir():
                        if stride_dir.is_dir() and stride_dir.name.startswith("stride_"):
                            note_file = stride_dir / "note.txt"
                            if note_file.exists():
                                f1_scores = extract_func(note_file)
                                if f1_scores:
                                    f1_scores.update({
                                        'eval_type': eval_type,
                                        'window': window_dir.name,
                                        'stride': stride_dir.name,
                                        'method': method_name
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
            f1_scores['word_binary_f1'] = float(binary_match.group(1))
        
        # Extract multiclass F1 score
        multiclass_match = re.search(r'Multiclass Classification.*?F1-Score:\s*([\d.]+)', content, re.DOTALL)
        if multiclass_match:
            f1_scores['word_multiclass_f1'] = float(multiclass_match.group(1))
        
        return f1_scores
    except Exception as e:
        return None

def extract_word_iou_eval_f1(note_file):
    """Extract F1 scores from word-IoU evaluation note file"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        f1_scores = {}
        
        # Extract F1 scores for key IoU thresholds
        thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
        
        for threshold in thresholds:
            # Binary F1
            pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?F1-Score:\s*([\d.]+)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                f1_scores[f'word_iou_binary_f1_{threshold.replace(".", "_")}'] = float(match.group(1))
            
            # Multiclass F1  
            pattern = rf'--- IoU Threshold {threshold} ---.*?Multiclass Classification.*?F1-Score:\s*([\d.]+)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                f1_scores[f'word_iou_multiclass_f1_{threshold.replace(".", "_")}'] = float(match.group(1))
        
        return f1_scores
    except Exception as e:
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
        
        # Extract F1 scores for key IoU thresholds
        thresholds = ['0.1', '0.3', '0.5', '0.7', '0.9']
        
        for threshold in thresholds:
            # Binary F1
            pattern = rf'--- IoU Threshold {threshold} ---.*?Binary Classification.*?F1-Score:\s*([\d.]+)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                f1_scores[f'iou_binary_f1_{threshold.replace(".", "_")}'] = float(match.group(1))
        
        return f1_scores
    except Exception as e:
        return None

def parse_window_stride_comprehensive(df):
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

def calculate_comprehensive_correlations(df_original, df_comprehensive):
    """Calculate correlations for all available metrics"""
    
    correlations = {}
    
    print("\n" + "="*80)
    print("COMPREHENSIVE CORRELATION ANALYSIS")
    print("="*80)
    
    # 1. Original data correlations (window_f1, word_f1, combined_iou)
    if not df_original.empty:
        print("\n🔵 ORIGINAL DATA CORRELATIONS:")
        print("-" * 50)
        
        metrics = ['window_f1', 'word_f1', 'combined_iou']
        
        for metric in metrics:
            if metric in df_original.columns:
                # Pearson correlation with window size
                corr_window, p_val_window = pearsonr(df_original['window_numeric'], df_original[metric])
                correlations[f'{metric}_vs_window'] = corr_window
                
                # Pearson correlation with stride
                corr_stride, p_val_stride = pearsonr(df_original['stride_numeric'], df_original[metric])
                correlations[f'{metric}_vs_stride'] = corr_stride
                
                # Spearman correlation (rank-based, more robust)
                spear_window, _ = spearmanr(df_original['window_numeric'], df_original[metric])
                spear_stride, _ = spearmanr(df_original['stride_numeric'], df_original[metric])
                
                print(f"📊 {metric.replace('_', ' ').title()}:")
                print(f"   vs Window Size: {corr_window:+.3f} (Pearson), {spear_window:+.3f} (Spearman)")
                print(f"   vs Stride Size: {corr_stride:+.3f} (Pearson), {spear_stride:+.3f} (Spearman)")
                
                # Interpretation
                if abs(corr_window) >= 0.7:
                    strength = "Strong"
                elif abs(corr_window) >= 0.3:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                
                direction = "Positive" if corr_window > 0 else "Negative"
                print(f"   Trend: {strength} {direction} correlation with window size")
                print()
    
    # 2. Comprehensive data correlations (all methods)
    if not df_comprehensive.empty:
        print("\n🔴 COMPREHENSIVE METHOD CORRELATIONS:")
        print("-" * 50)
        
        # Group by method and calculate correlations
        for method in df_comprehensive['method'].unique():
            method_data = df_comprehensive[df_comprehensive['method'] == method]
            
            print(f"\n📋 {method.replace('_', ' ').upper()} METHOD:")
            
            # Find F1 columns for this method
            f1_columns = [col for col in method_data.columns if 'f1' in col.lower() or 'iou' in col.lower()]
            f1_columns = [col for col in f1_columns if col not in ['eval_type', 'window', 'stride', 'method']]
            
            for metric in f1_columns:
                if method_data[metric].notna().sum() > 5:  # Need at least 6 data points
                    # Window correlation
                    corr_window, p_val = pearsonr(method_data['window_numeric'], method_data[metric])
                    spear_window, _ = spearmanr(method_data['window_numeric'], method_data[metric])
                    
                    # Stride correlation
                    corr_stride, _ = pearsonr(method_data['stride_numeric'], method_data[metric])
                    spear_stride, _ = spearmanr(method_data['stride_numeric'], method_data[metric])
                    
                    correlations[f'{method}_{metric}_vs_window'] = corr_window
                    correlations[f'{method}_{metric}_vs_stride'] = corr_stride
                    
                    print(f"   {metric.replace('_', ' ').title()}:")
                    print(f"     vs Window: {corr_window:+.3f} (Pearson), {spear_window:+.3f} (Spearman)")
                    print(f"     vs Stride: {corr_stride:+.3f} (Pearson), {spear_stride:+.3f} (Spearman)")
                    
                    # Significance
                    if p_val < 0.001:
                        sig = "***"
                    elif p_val < 0.01:
                        sig = "**"
                    elif p_val < 0.05:
                        sig = "*"
                    else:
                        sig = "ns"
                    
                    print(f"     Significance: p={p_val:.4f} {sig}")
    
    return correlations

def create_correlation_matrix_heatmap(df_original, df_comprehensive):
    """Create correlation matrix heatmaps"""
    
    output_dir = Path("fixed_smart_parallel_results/fresh_correlation_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Original data correlation matrix
    if not df_original.empty:
        plt.figure(figsize=(10, 8))
        
        # Select numeric columns for correlation
        numeric_cols = ['window_numeric', 'stride_numeric', 'stride_percentage', 
                       'window_f1', 'word_f1', 'combined_iou']
        
        # Filter to available columns
        available_cols = [col for col in numeric_cols if col in df_original.columns]
        
        if len(available_cols) > 2:
            corr_matrix = df_original[available_cols].corr()
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, mask=mask, cbar_kws={"shrink": .8},
                       fmt='.3f', annot_kws={'size': 10})
            
            plt.title('Original Data Correlation Matrix\n(Window F1, Word F1, Combined IoU)', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_dir / "original_correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 2. Method-specific correlation matrices
    if not df_comprehensive.empty:
        methods = df_comprehensive['method'].unique()
        
        for method in methods:
            method_data = df_comprehensive[df_comprehensive['method'] == method]
            
            # Get numeric F1 columns
            f1_cols = [col for col in method_data.columns if 
                      ('f1' in col.lower() or 'iou' in col.lower()) and 
                      method_data[col].dtype in ['float64', 'int64']]
            
            if len(f1_cols) > 1:
                plt.figure(figsize=(12, 8))
                
                # Create correlation matrix including window/stride
                corr_cols = ['window_numeric', 'stride_numeric'] + f1_cols
                available_corr_cols = [col for col in corr_cols if col in method_data.columns]
                
                if len(available_corr_cols) > 2:
                    corr_matrix = method_data[available_corr_cols].corr()
                    
                    # Create heatmap
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                               square=True, mask=mask, cbar_kws={"shrink": .8},
                               fmt='.3f', annot_kws={'size': 8})
                    
                    plt.title(f'{method.replace("_", " ").title()} Method Correlation Matrix', 
                             fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    safe_method = method.replace('_', '_')
                    plt.savefig(output_dir / f"{safe_method}_correlation_matrix.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()

def save_correlation_results(correlations, df_original, df_comprehensive):
    """Save detailed correlation results"""
    
    output_dir = Path("fixed_smart_parallel_results/fresh_correlation_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save correlation summary
    with open(output_dir / "fresh_correlation_summary.txt", 'w') as f:
        f.write("=== FRESH CORRELATION ANALYSIS RESULTS ===\n\n")
        
        f.write("DATA OVERVIEW:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Original Data Configurations: {len(df_original) if not df_original.empty else 0}\n")
        f.write(f"Comprehensive Data Configurations: {len(df_comprehensive) if not df_comprehensive.empty else 0}\n")
        
        if not df_comprehensive.empty:
            f.write(f"Methods Found: {', '.join(df_comprehensive['method'].unique())}\n")
            f.write(f"Evaluation Types: {', '.join(df_comprehensive['eval_type'].unique())}\n")
        
        f.write(f"Total Unique Correlations Calculated: {len(correlations)}\n\n")
        
        f.write("CORRELATION COEFFICIENTS:\n")
        f.write("-" * 40 + "\n")
        
        # Sort correlations by strength
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for corr_name, corr_value in sorted_corr:
            strength = "Strong" if abs(corr_value) >= 0.7 else "Moderate" if abs(corr_value) >= 0.3 else "Weak"
            direction = "Positive" if corr_value > 0 else "Negative"
            
            f.write(f"{corr_name}: {corr_value:+.3f} ({strength} {direction})\n")
        
        f.write(f"\nSTRONG CORRELATIONS (|r| >= 0.7):\n")
        f.write("-" * 40 + "\n")
        
        strong_corr = [(name, val) for name, val in correlations.items() if abs(val) >= 0.7]
        for corr_name, corr_value in strong_corr:
            f.write(f"{corr_name}: {corr_value:+.3f}\n")
        
        if not strong_corr:
            f.write("No strong correlations found (|r| >= 0.7)\n")
    
    # Save raw correlation data
    corr_df = pd.DataFrame([
        {'correlation_name': name, 'correlation_value': value, 'absolute_value': abs(value)}
        for name, value in correlations.items()
    ])
    corr_df = corr_df.sort_values('absolute_value', ascending=False)
    corr_df.to_csv(output_dir / "fresh_correlations_data.csv", index=False)
    
    print(f"\n📊 Correlation results saved to: {output_dir}")
    print(f"📄 Summary: fresh_correlation_summary.txt")
    print(f"📊 Raw data: fresh_correlations_data.csv")
    print(f"🔥 Heatmaps: *_correlation_matrix.png")

def main():
    print("=== FRESH CORRELATION ANALYSIS ===\n")
    
    # Load original correlation data
    print("📊 Loading original correlation data...")
    df_original = load_comprehensive_data()
    
    # Extract comprehensive F1 data from all methods
    print("📊 Extracting comprehensive F1 data from evaluation files...")
    df_comprehensive = extract_comprehensive_f1_data()
    
    if df_comprehensive.empty and df_original.empty:
        print("❌ No data found for correlation analysis!")
        return
    
    # Parse window/stride values for comprehensive data
    if not df_comprehensive.empty:
        df_comprehensive = parse_window_stride_comprehensive(df_comprehensive)
        print(f"✅ Extracted comprehensive data: {len(df_comprehensive)} configurations")
        print(f"📋 Methods: {df_comprehensive['method'].unique()}")
    
    # Calculate correlations
    print("\n📈 Calculating comprehensive correlations...")
    correlations = calculate_comprehensive_correlations(df_original, df_comprehensive)
    
    # Create correlation heatmaps
    print("\n🔥 Creating correlation heatmaps...")
    create_correlation_matrix_heatmap(df_original, df_comprehensive)
    
    # Save results
    print("\n💾 Saving correlation results...")
    save_correlation_results(correlations, df_original, df_comprehensive)
    
    print("\n🎉 FRESH CORRELATION ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: fixed_smart_parallel_results/fresh_correlation_analysis/")

if __name__ == "__main__":
    main()

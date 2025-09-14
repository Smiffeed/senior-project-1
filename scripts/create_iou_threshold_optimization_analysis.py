#!/usr/bin/env python3
"""
IoU Threshold Optimization Analysis
Create comprehensive analysis to determine optimal IoU thresholds for word_iou_eval and iou_eval methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_word_iou_eval_threshold_data():
    """Load word_iou_eval threshold data from all configurations"""
    base_path = Path("fixed_smart_parallel_results/eval_by_0.05/eval_by_0.05/word_iou_eval")
    
    if not base_path.exists():
        print(f"❌ Word IoU eval path not found: {base_path}")
        return None
    
    all_data = []
    config_count = 0
    
    print("📊 Loading word_iou_eval threshold data...")
    
    # Traverse all window/stride combinations
    for window_dir in base_path.glob("window_*"):
        if window_dir.is_dir():
            window_value = float(window_dir.name.replace('window_', '').replace('s', ''))
            
            for stride_dir in window_dir.glob("stride_*"):
                if stride_dir.is_dir():
                    stride_value = float(stride_dir.name.replace('stride_', '').replace('s', ''))
                    
                    # Look for threshold results file
                    threshold_file = stride_dir / "word_iou_threshold_results.csv"
                    if threshold_file.exists():
                        try:
                            threshold_data = pd.read_csv(threshold_file)
                            threshold_data['window'] = window_value
                            threshold_data['stride'] = stride_value
                            threshold_data['method'] = 'word_iou_eval'
                            all_data.append(threshold_data)
                            config_count += 1
                        except Exception as e:
                            print(f"⚠️ Error loading {threshold_file}: {e}")
                            continue
    
    if all_data:
        consolidated_data = pd.concat(all_data, ignore_index=True)
        print(f"✅ Loaded word_iou_eval data: {config_count} configurations, {len(consolidated_data)} records")
        return consolidated_data
    else:
        print("❌ No word_iou_eval threshold data found")
        return None

def load_iou_eval_threshold_data():
    """Load iou_eval threshold data from all configurations"""
    base_path = Path("fixed_smart_parallel_results/eval_by_0.05/eval_by_0.05/iou_eval")
    
    if not base_path.exists():
        print(f"❌ IoU eval path not found: {base_path}")
        return None
    
    all_data = []
    config_count = 0
    
    print("📊 Loading iou_eval threshold data...")
    
    # Traverse all window/stride combinations
    for window_dir in base_path.glob("window_*"):
        if window_dir.is_dir():
            window_value = float(window_dir.name.replace('window_', '').replace('s', ''))
            
            for stride_dir in window_dir.glob("stride_*"):
                if stride_dir.is_dir():
                    stride_value = float(stride_dir.name.replace('stride_', '').replace('s', ''))
                    
                    # Look for threshold reference file
                    threshold_file = stride_dir / "iou_threshold_reference.csv"
                    if threshold_file.exists():
                        try:
                            threshold_data = pd.read_csv(threshold_file)
                            threshold_data['window'] = window_value
                            threshold_data['stride'] = stride_value
                            threshold_data['method'] = 'iou_eval'
                            all_data.append(threshold_data)
                            config_count += 1
                        except Exception as e:
                            print(f"⚠️ Error loading {threshold_file}: {e}")
                            continue
    
    if all_data:
        consolidated_data = pd.concat(all_data, ignore_index=True)
        print(f"✅ Loaded iou_eval data: {config_count} configurations, {len(consolidated_data)} records")
        return consolidated_data
    else:
        print("❌ No iou_eval threshold data found")
        return None

def analyze_threshold_performance(word_iou_data, iou_data):
    """Analyze F1 performance across different IoU thresholds"""
    
    # Combine data for analysis
    combined_data = []
    
    if word_iou_data is not None:
        combined_data.append(word_iou_data)
    
    if iou_data is not None:
        combined_data.append(iou_data)
    
    if not combined_data:
        print("❌ No data available for analysis")
        return None
    
    all_data = pd.concat(combined_data, ignore_index=True)
    
    # Identify available threshold columns and F1 score columns
    threshold_cols = [col for col in all_data.columns if 'threshold' in col.lower()]
    f1_cols = [col for col in all_data.columns if 'f1' in col.lower()]
    
    print(f"📊 Found threshold columns: {threshold_cols}")
    print(f"📊 Found F1 columns: {f1_cols}")
    
    return all_data

def create_threshold_optimization_plots(data):
    """Create comprehensive threshold optimization analysis plots"""
    
    if data is None or data.empty:
        print("❌ No data available for plotting")
        return
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('IoU Threshold Optimization Analysis: word_iou_eval vs iou_eval', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Get unique methods
    methods = data['method'].unique()
    print(f"📊 Methods found: {methods}")
    
    # Determine available F1 metrics and thresholds
    # Check for common F1 score columns
    f1_metrics = []
    if 'binary_f1' in data.columns:
        f1_metrics.append('binary_f1')
    if 'multiclass_f1' in data.columns:
        f1_metrics.append('multiclass_f1')
    if 'f1_score' in data.columns:
        f1_metrics.append('f1_score')
    if 'f1' in data.columns:
        f1_metrics.append('f1')
    if 'weighted_f1' in data.columns:
        f1_metrics.append('weighted_f1')
    
    # Check for threshold column
    threshold_col = None
    for col in ['iou_threshold', 'threshold', 'IoU_threshold']:
        if col in data.columns:
            threshold_col = col
            break
    
    if threshold_col is None:
        print("❌ No threshold column found")
        return
    
    print(f"📊 Using threshold column: {threshold_col}")
    print(f"📊 Available F1 metrics: {f1_metrics}")
    
    # Get threshold values
    thresholds = sorted(data[threshold_col].unique())
    print(f"📊 Threshold values: {thresholds}")
    
    colors = {'word_iou_eval': '#4ECDC4', 'iou_eval': '#45B7D1'}
    
    # Plot 1: Binary F1 vs Threshold (if available)
    ax1 = axes[0, 0]
    if 'binary_f1' in f1_metrics:
        for method in methods:
            method_data = data[data['method'] == method]
            threshold_means = []
            threshold_stds = []
            
            for threshold in thresholds:
                threshold_data = method_data[method_data[threshold_col] == threshold]
                if not threshold_data.empty:
                    threshold_means.append(threshold_data['binary_f1'].mean())
                    threshold_stds.append(threshold_data['binary_f1'].std())
                else:
                    threshold_means.append(np.nan)
                    threshold_stds.append(np.nan)
            
            # Remove NaN values for plotting
            valid_indices = ~np.isnan(threshold_means)
            valid_thresholds = np.array(thresholds)[valid_indices]
            valid_means = np.array(threshold_means)[valid_indices]
            valid_stds = np.array(threshold_stds)[valid_indices]
            
            if len(valid_thresholds) > 0:
                ax1.plot(valid_thresholds, valid_means, 'o-', 
                        color=colors.get(method, '#333333'), linewidth=3, 
                        markersize=8, label=f'{method} Binary F1', alpha=0.8)
                
                # Add confidence intervals
                ax1.fill_between(valid_thresholds, 
                               valid_means - valid_stds,
                               valid_means + valid_stds,
                               color=colors.get(method, '#333333'), alpha=0.2)
    
    ax1.set_xlabel('IoU Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Binary F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title('Binary F1 Score vs IoU Threshold', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Multiclass F1 vs Threshold (if available)
    ax2 = axes[0, 1]
    if 'multiclass_f1' in f1_metrics:
        for method in methods:
            method_data = data[data['method'] == method]
            threshold_means = []
            threshold_stds = []
            
            for threshold in thresholds:
                threshold_data = method_data[method_data[threshold_col] == threshold]
                if not threshold_data.empty:
                    threshold_means.append(threshold_data['multiclass_f1'].mean())
                    threshold_stds.append(threshold_data['multiclass_f1'].std())
                else:
                    threshold_means.append(np.nan)
                    threshold_stds.append(np.nan)
            
            # Remove NaN values for plotting
            valid_indices = ~np.isnan(threshold_means)
            valid_thresholds = np.array(thresholds)[valid_indices]
            valid_means = np.array(threshold_means)[valid_indices]
            valid_stds = np.array(threshold_stds)[valid_indices]
            
            if len(valid_thresholds) > 0:
                ax2.plot(valid_thresholds, valid_means, 's-', 
                        color=colors.get(method, '#333333'), linewidth=3, 
                        markersize=8, label=f'{method} Multiclass F1', alpha=0.8)
                
                # Add confidence intervals
                ax2.fill_between(valid_thresholds, 
                               valid_means - valid_stds,
                               valid_means + valid_stds,
                               color=colors.get(method, '#333333'), alpha=0.2)
    
    ax2.set_xlabel('IoU Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Multiclass F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Multiclass F1 Score vs IoU Threshold', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Combined Comparison
    ax3 = axes[0, 2]
    for method in methods:
        method_data = data[data['method'] == method]
        
        # Use best available F1 metric
        primary_metric = f1_metrics[0] if f1_metrics else None
        
        if primary_metric:
            threshold_means = []
            for threshold in thresholds:
                threshold_data = method_data[method_data[threshold_col] == threshold]
                if not threshold_data.empty:
                    threshold_means.append(threshold_data[primary_metric].mean())
                else:
                    threshold_means.append(np.nan)
            
            # Remove NaN values for plotting
            valid_indices = ~np.isnan(threshold_means)
            valid_thresholds = np.array(thresholds)[valid_indices]
            valid_means = np.array(threshold_means)[valid_indices]
            
            if len(valid_thresholds) > 0:
                ax3.plot(valid_thresholds, valid_means, '^-', 
                        color=colors.get(method, '#333333'), linewidth=3, 
                        markersize=8, label=f'{method} {primary_metric}', alpha=0.8)
    
    ax3.set_xlabel('IoU Threshold', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax3.set_title('Combined F1 Performance vs IoU Threshold', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Threshold Distribution Analysis
    ax4 = axes[1, 0]
    for method in methods:
        method_data = data[data['method'] == method]
        primary_metric = f1_metrics[0] if f1_metrics else None
        
        if primary_metric:
            # Create box plot showing F1 distribution for each threshold
            threshold_groups = []
            threshold_labels = []
            
            for threshold in thresholds:
                threshold_data = method_data[method_data[threshold_col] == threshold]
                if not threshold_data.empty:
                    threshold_groups.append(threshold_data[primary_metric].values)
                    threshold_labels.append(f'{threshold}')
            
            if threshold_groups:
                positions = np.arange(len(threshold_labels))
                bp = ax4.boxplot(threshold_groups, positions=positions, 
                               patch_artist=True, widths=0.6)
                
                # Color the boxes
                for patch in bp['boxes']:
                    patch.set_facecolor(colors.get(method, '#333333'))
                    patch.set_alpha(0.7)
    
    ax4.set_xlabel('IoU Threshold', fontsize=12, fontweight='bold')
    ax4.set_ylabel('F1 Score Distribution', fontsize=12, fontweight='bold')
    ax4.set_title('F1 Score Distribution by Threshold', fontsize=14, fontweight='bold')
    if threshold_labels:
        ax4.set_xticks(range(len(threshold_labels)))
        ax4.set_xticklabels(threshold_labels)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Optimal Threshold Analysis
    ax5 = axes[1, 1]
    
    optimal_thresholds = {}
    
    for method in methods:
        method_data = data[data['method'] == method]
        primary_metric = f1_metrics[0] if f1_metrics else None
        
        if primary_metric:
            threshold_means = []
            for threshold in thresholds:
                threshold_data = method_data[method_data[threshold_col] == threshold]
                if not threshold_data.empty:
                    threshold_means.append(threshold_data[primary_metric].mean())
                else:
                    threshold_means.append(0)
            
            # Find optimal threshold
            if threshold_means:
                best_idx = np.argmax(threshold_means)
                optimal_threshold = thresholds[best_idx]
                optimal_score = threshold_means[best_idx]
                optimal_thresholds[method] = (optimal_threshold, optimal_score)
                
                ax5.bar(method, optimal_score, color=colors.get(method, '#333333'), 
                       alpha=0.7, label=f'Threshold: {optimal_threshold}')
    
    ax5.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Best F1 Score', fontsize=12, fontweight='bold')
    ax5.set_title('Optimal F1 Performance by Method', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Threshold Sensitivity Analysis
    ax6 = axes[1, 2]
    
    for method in methods:
        method_data = data[data['method'] == method]
        primary_metric = f1_metrics[0] if f1_metrics else None
        
        if primary_metric:
            threshold_means = []
            threshold_stds = []
            
            for threshold in thresholds:
                threshold_data = method_data[method_data[threshold_col] == threshold]
                if not threshold_data.empty:
                    threshold_means.append(threshold_data[primary_metric].mean())
                    threshold_stds.append(threshold_data[primary_metric].std())
                else:
                    threshold_means.append(0)
                    threshold_stds.append(0)
            
            # Calculate coefficient of variation (CV = std/mean)
            cv_values = []
            for mean_val, std_val in zip(threshold_means, threshold_stds):
                if mean_val > 0:
                    cv_values.append(std_val / mean_val)
                else:
                    cv_values.append(0)
            
            ax6.plot(thresholds, cv_values, 'o-', 
                    color=colors.get(method, '#333333'), linewidth=3, 
                    markersize=8, label=f'{method} CV', alpha=0.8)
    
    ax6.set_xlabel('IoU Threshold', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Coefficient of Variation', fontsize=12, fontweight='bold')
    ax6.set_title('Threshold Sensitivity (Lower = More Stable)', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("iou_threshold_optimization")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / "iou_threshold_optimization_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return optimal thresholds for summary
    return optimal_thresholds

def generate_threshold_summary(data, optimal_thresholds):
    """Generate comprehensive threshold optimization summary"""
    
    output_dir = Path("iou_threshold_optimization")
    output_dir.mkdir(exist_ok=True)
    
    # Get available metrics
    f1_metrics = []
    if 'binary_f1' in data.columns:
        f1_metrics.append('binary_f1')
    if 'multiclass_f1' in data.columns:
        f1_metrics.append('multiclass_f1')
    if 'f1_score' in data.columns:
        f1_metrics.append('f1_score')
    if 'f1' in data.columns:
        f1_metrics.append('f1')
    if 'weighted_f1' in data.columns:
        f1_metrics.append('weighted_f1')
    
    threshold_col = None
    for col in ['iou_threshold', 'threshold', 'IoU_threshold']:
        if col in data.columns:
            threshold_col = col
            break
    
    if threshold_col is None:
        print("❌ No threshold column found for summary")
        return
    
    summary_data = []
    
    # Analyze each method and threshold combination
    methods = data['method'].unique()
    thresholds = sorted(data[threshold_col].unique())
    
    for method in methods:
        method_data = data[data['method'] == method]
        
        for metric in f1_metrics:
            for threshold in thresholds:
                threshold_data = method_data[method_data[threshold_col] == threshold]
                
                if not threshold_data.empty:
                    summary_data.append({
                        'Method': method,
                        'Metric': metric,
                        'IoU_Threshold': threshold,
                        'Mean_F1': threshold_data[metric].mean(),
                        'Max_F1': threshold_data[metric].max(),
                        'Std_F1': threshold_data[metric].std(),
                        'Count_Configs': len(threshold_data),
                        'CV': threshold_data[metric].std() / threshold_data[metric].mean() if threshold_data[metric].mean() > 0 else np.nan
                    })
    
    # Create and save summary DataFrame
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round(4)
        summary_df.to_csv(output_dir / 'threshold_optimization_summary.csv', index=False)
        
        print("\n" + "="*100)
        print("IoU THRESHOLD OPTIMIZATION ANALYSIS SUMMARY")
        print("="*100)
        
        # Show top performing threshold for each method-metric combination
        for method in methods:
            for metric in f1_metrics:
                method_metric_data = summary_df[
                    (summary_df['Method'] == method) & 
                    (summary_df['Metric'] == metric)
                ]
                
                # Filter out NaN values
                method_metric_data = method_metric_data.dropna(subset=['Mean_F1'])
                
                if not method_metric_data.empty:
                    best_row = method_metric_data.loc[method_metric_data['Mean_F1'].idxmax()]
                    
                    print(f"\n🎯 {method.upper()} - {metric.upper()}:")
                    print(f"   Optimal Threshold: {best_row['IoU_Threshold']}")
                    print(f"   Best Mean F1: {best_row['Mean_F1']:.4f}")
                    print(f"   Max F1: {best_row['Max_F1']:.4f}")
                    print(f"   Stability (CV): {best_row['CV']:.4f}")
                    print(f"   Configurations: {best_row['Count_Configs']}")
        
        print("\n" + "="*100)
        print("RECOMMENDATION SUMMARY")
        print("="*100)
        
        # Provide recommendations
        for method in methods:
            if f1_metrics:
                primary_metric = f1_metrics[0]
                method_data = summary_df[
                    (summary_df['Method'] == method) & 
                    (summary_df['Metric'] == primary_metric)
                ]
                
                # Filter out NaN values
                method_data = method_data.dropna(subset=['Mean_F1', 'CV'])
                
                if not method_data.empty:
                    best_performance = method_data.loc[method_data['Mean_F1'].idxmax()]
                    most_stable = method_data.loc[method_data['CV'].idxmin()]
                    
                    print(f"\n📊 {method.upper()} RECOMMENDATIONS:")
                    print(f"   🏆 Best Performance: IoU Threshold = {best_performance['IoU_Threshold']} (F1 = {best_performance['Mean_F1']:.4f})")
                    print(f"   🎯 Most Stable: IoU Threshold = {most_stable['IoU_Threshold']} (CV = {most_stable['CV']:.4f})")
                    
                    # Balanced recommendation
                    method_data['score'] = method_data['Mean_F1'] * (1 - method_data['CV'])  # Balance performance and stability
                    balanced_best = method_data.loc[method_data['score'].idxmax()]
                    print(f"   ⚖️ Balanced Choice: IoU Threshold = {balanced_best['IoU_Threshold']} (F1 = {balanced_best['Mean_F1']:.4f}, CV = {balanced_best['CV']:.4f})")
        
        print("="*100)

def main():
    """Main execution function"""
    print("="*80)
    print("IoU THRESHOLD OPTIMIZATION ANALYSIS")
    print("="*80)
    
    # Load data
    print("📊 Loading threshold data for both methods...")
    word_iou_data = load_word_iou_eval_threshold_data()
    iou_data = load_iou_eval_threshold_data()
    
    # Analyze threshold performance
    print("\n🔍 Analyzing threshold performance...")
    combined_data = analyze_threshold_performance(word_iou_data, iou_data)
    
    if combined_data is not None:
        # Create optimization plots
        print("\n🎨 Creating threshold optimization plots...")
        optimal_thresholds = create_threshold_optimization_plots(combined_data)
        
        # Generate summary
        print("\n📋 Generating optimization summary...")
        generate_threshold_summary(combined_data, optimal_thresholds)
        
        print("\n✅ IoU threshold optimization analysis completed!")
        print("📁 Output saved to: iou_threshold_optimization/")
    else:
        print("❌ No data available for analysis")

if __name__ == "__main__":
    main()

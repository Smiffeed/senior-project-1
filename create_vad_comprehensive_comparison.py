#!/usr/bin/env python3
"""
📊 VAD vs Comprehensive Results Comparison
Create a focused comparison between VAD refined results and comprehensive evaluation results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication quality
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_process_vad_data():
    """Load VAD refinement summary"""
    vad_path = Path("vad_refined_results/vad_refinement_summary.csv")
    
    if not vad_path.exists():
        raise FileNotFoundError(f"VAD summary not found: {vad_path}")
    
    df = pd.read_csv(vad_path)
    df = df[df['success'] == True].copy()
    
    # Extract window size as float
    df['window_size'] = pd.to_numeric(df['window_size'], errors='coerce')
    
    # Parse stride information
    df['stride_value'] = df['stride_info'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['stride_unit'] = df['stride_info'].str.extract(r'(\w+)$')
    
    return df

def load_and_process_comprehensive_data():
    """Load comprehensive evaluation summary"""
    comp_path = Path("fixed_smart_parallel_results/fixed_smart_parallel_summary.csv")
    
    if not comp_path.exists():
        raise FileNotFoundError(f"Comprehensive summary not found: {comp_path}")
    
    df = pd.read_csv(comp_path)
    df = df[df['success'] == True].copy()
    
    # Extract window size
    df['window_size'] = df['window'].str.extract(r'(\d+\.?\d*)').astype(float)
    
    # Parse stride information
    df['stride_value'] = df['stride'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['stride_unit'] = df['stride'].str.extract(r'(\w+)$')
    
    # Map evaluation types
    df['eval_type'] = df.apply(lambda row: 'eval_percent' if '%' in row['stride'] else 'eval_by_0.05', axis=1)
    
    # The comprehensive results have word_f1 which is similar to our binary_f1
    # and combined_iou already exists
    df['binary_f1'] = df['word_f1']  # Word F1 is binary classification (profanity vs none)
    df['multiclass_f1'] = np.nan     # Not available in comprehensive results
    
    return df

def create_vad_vs_comprehensive_comparison():
    """Create comprehensive comparison between VAD and original results"""
    print("📊 Creating VAD vs Comprehensive Results Comparison...")
    
    try:
        # Load both datasets
        vad_df = load_and_process_vad_data()
        comp_df = load_and_process_comprehensive_data()
        
        print(f"✅ Loaded {len(vad_df)} VAD results")
        print(f"✅ Loaded {len(comp_df)} comprehensive results")
        
        # Create output directory
        output_dir = Path("vad_vs_comprehensive_comparison")
        output_dir.mkdir(exist_ok=True)
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('VAD Refined vs Comprehensive Evaluation Results Comparison', 
                    fontsize=16, fontweight='bold')
        
        # 1. Binary F1 Comparison by Window Size
        create_f1_comparison_plot(vad_df, comp_df, ax1)
        
        # 2. IoU Comparison by Window Size
        create_iou_comparison_plot(vad_df, comp_df, ax2)
        
        # 3. Performance Distribution Comparison
        create_distribution_comparison(vad_df, comp_df, ax3)
        
        # 4. Summary Statistics Table
        create_summary_table(vad_df, comp_df, ax4)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'vad_vs_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'vad_vs_comprehensive_comparison.pdf', bbox_inches='tight')
        plt.show()
        
        # Create detailed statistics report
        create_detailed_statistics_report(vad_df, comp_df, output_dir)
        
        print(f"✅ Comparison graphs saved to {output_dir}")
        
    except Exception as e:
        print(f"❌ Error creating comparison: {e}")
        import traceback
        traceback.print_exc()

def create_f1_comparison_plot(vad_df, comp_df, ax):
    """Create F1 comparison plot by window size"""
    # Calculate mean F1 by window size for both datasets
    vad_f1_by_window = vad_df.groupby('window_size')['binary_f1'].mean().reset_index()
    comp_f1_by_window = comp_df.groupby('window_size')['binary_f1'].mean().reset_index()
    
    # Plot both lines
    ax.plot(vad_f1_by_window['window_size'], vad_f1_by_window['binary_f1'], 
           marker='o', linewidth=2.5, markersize=7, label='VAD Refined', 
           color='#2E86AB', alpha=0.8)
    
    ax.plot(comp_f1_by_window['window_size'], comp_f1_by_window['binary_f1'], 
           marker='s', linewidth=2.5, markersize=7, label='Comprehensive (Original)', 
           color='#E63946', alpha=0.8)
    
    # Add scatter points for individual measurements
    ax.scatter(vad_df['window_size'], vad_df['binary_f1'], alpha=0.2, s=15, color='#2E86AB')
    ax.scatter(comp_df['window_size'], comp_df['binary_f1'], alpha=0.2, s=15, color='#E63946')
    
    ax.set_xlabel('Window Size (seconds)', fontweight='bold')
    ax.set_ylabel('Binary F1 Score', fontweight='bold')
    ax.set_title('Binary F1 Performance Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1)

def create_iou_comparison_plot(vad_df, comp_df, ax):
    """Create IoU comparison plot by window size"""
    # Calculate mean IoU by window size for both datasets
    vad_iou_by_window = vad_df.groupby('window_size')['combined_iou'].mean().reset_index()
    comp_iou_by_window = comp_df.groupby('window_size')['combined_iou'].mean().reset_index()
    
    # Plot both lines
    ax.plot(vad_iou_by_window['window_size'], vad_iou_by_window['combined_iou'], 
           marker='o', linewidth=2.5, markersize=7, label='VAD Refined', 
           color='#F18F01', alpha=0.8)
    
    ax.plot(comp_iou_by_window['window_size'], comp_iou_by_window['combined_iou'], 
           marker='s', linewidth=2.5, markersize=7, label='Comprehensive (Original)', 
           color='#F77F00', alpha=0.8)
    
    # Add scatter points for individual measurements
    ax.scatter(vad_df['window_size'], vad_df['combined_iou'], alpha=0.2, s=15, color='#F18F01')
    ax.scatter(comp_df['window_size'], comp_df['combined_iou'], alpha=0.2, s=15, color='#F77F00')
    
    ax.set_xlabel('Window Size (seconds)', fontweight='bold')
    ax.set_ylabel('Combined IoU', fontweight='bold')
    ax.set_title('IoU Performance Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

def create_distribution_comparison(vad_df, comp_df, ax):
    """Create distribution comparison of F1 scores"""
    # Create box plots for F1 score distributions
    data_to_plot = [vad_df['binary_f1'].values, comp_df['binary_f1'].values]
    labels = ['VAD Refined', 'Comprehensive']
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = ['#2E86AB', '#E63946']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Binary F1 Score', fontweight='bold')
    ax.set_title('F1 Score Distribution Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

def create_summary_table(vad_df, comp_df, ax):
    """Create summary statistics table"""
    ax.axis('off')
    
    # Calculate statistics
    vad_f1_mean = vad_df['binary_f1'].mean()
    vad_f1_std = vad_df['binary_f1'].std()
    vad_f1_best = vad_df['binary_f1'].max()
    vad_iou_mean = vad_df['combined_iou'].mean()
    vad_iou_std = vad_df['combined_iou'].std()
    vad_iou_best = vad_df['combined_iou'].max()
    
    comp_f1_mean = comp_df['binary_f1'].mean()
    comp_f1_std = comp_df['binary_f1'].std()
    comp_f1_best = comp_df['binary_f1'].max()
    comp_iou_mean = comp_df['combined_iou'].mean()
    comp_iou_std = comp_df['combined_iou'].std()
    comp_iou_best = comp_df['combined_iou'].max()
    
    # Calculate improvements
    f1_mean_improvement = ((vad_f1_mean - comp_f1_mean) / comp_f1_mean * 100)
    f1_best_improvement = ((vad_f1_best - comp_f1_best) / comp_f1_best * 100)
    iou_mean_improvement = ((vad_iou_mean - comp_iou_mean) / comp_iou_mean * 100)
    iou_best_improvement = ((vad_iou_best - comp_iou_best) / comp_iou_best * 100)
    
    table_data = [
        ['Metric', 'VAD Refined', 'Comprehensive', 'Improvement'],
        ['', '', '', ''],
        ['Binary F1 Mean', f'{vad_f1_mean:.3f} ± {vad_f1_std:.3f}', 
         f'{comp_f1_mean:.3f} ± {comp_f1_std:.3f}', f'{f1_mean_improvement:+.1f}%'],
        ['Binary F1 Best', f'{vad_f1_best:.3f}', f'{comp_f1_best:.3f}', f'{f1_best_improvement:+.1f}%'],
        ['IoU Mean', f'{vad_iou_mean:.3f} ± {vad_iou_std:.3f}', 
         f'{comp_iou_mean:.3f} ± {comp_iou_std:.3f}', f'{iou_mean_improvement:+.1f}%'],
        ['IoU Best', f'{vad_iou_best:.3f}', f'{comp_iou_best:.3f}', f'{iou_best_improvement:+.1f}%'],
        ['', '', '', ''],
        ['Configurations', f'{len(vad_df)}', f'{len(comp_df)}', ''],
        ['Processing Method', 'Window → VAD Refined', 'Window → Direct Eval', '']
    ]
    
    table = ax.table(cellText=table_data,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    
    # Style the header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(1, i)].set_facecolor('#D9E2F3')  # Empty row for spacing
        table[(6, i)].set_facecolor('#D9E2F3')  # Empty row for spacing
    
    # Color improvement cells
    for row in [2, 3, 4, 5]:
        cell_value = table_data[row][3]
        if '+' in cell_value:
            table[(row, 3)].set_facecolor('#C6EFCE')  # Light green for positive
        elif cell_value and cell_value != '':
            table[(row, 3)].set_facecolor('#FFC7CE')  # Light red for negative
    
    ax.set_title('Performance Comparison Summary', fontweight='bold', pad=20, fontsize=14)

def create_detailed_statistics_report(vad_df, comp_df, output_dir):
    """Create detailed statistics report"""
    print("\n📊 DETAILED PERFORMANCE COMPARISON:")
    print("=" * 50)
    
    # Overall statistics
    print(f"VAD REFINED RESULTS:")
    print(f"   Binary F1: {vad_df['binary_f1'].mean():.3f} ± {vad_df['binary_f1'].std():.3f} (best: {vad_df['binary_f1'].max():.3f})")
    print(f"   Multiclass F1: {vad_df['multiclass_f1'].mean():.3f} ± {vad_df['multiclass_f1'].std():.3f} (best: {vad_df['multiclass_f1'].max():.3f})")
    print(f"   Combined IoU: {vad_df['combined_iou'].mean():.3f} ± {vad_df['combined_iou'].std():.3f} (best: {vad_df['combined_iou'].max():.3f})")
    
    print(f"\nCOMPREHENSIVE RESULTS:")
    print(f"   Binary F1: {comp_df['binary_f1'].mean():.3f} ± {comp_df['binary_f1'].std():.3f} (best: {comp_df['binary_f1'].max():.3f})")
    print(f"   Combined IoU: {comp_df['combined_iou'].mean():.3f} ± {comp_df['combined_iou'].std():.3f} (best: {comp_df['combined_iou'].max():.3f})")
    
    # Calculate improvements
    f1_mean_improvement = ((vad_df['binary_f1'].mean() - comp_df['binary_f1'].mean()) / comp_df['binary_f1'].mean() * 100)
    f1_best_improvement = ((vad_df['binary_f1'].max() - comp_df['binary_f1'].max()) / comp_df['binary_f1'].max() * 100)
    iou_mean_improvement = ((vad_df['combined_iou'].mean() - comp_df['combined_iou'].mean()) / comp_df['combined_iou'].mean() * 100)
    iou_best_improvement = ((vad_df['combined_iou'].max() - comp_df['combined_iou'].max()) / comp_df['combined_iou'].max() * 100)
    
    print(f"\n🚀 VAD REFINEMENT IMPROVEMENTS:")
    print(f"   Binary F1 Mean: {f1_mean_improvement:+.1f}%")
    print(f"   Binary F1 Best: {f1_best_improvement:+.1f}%")
    print(f"   IoU Mean: {iou_mean_improvement:+.1f}%")
    print(f"   IoU Best: {iou_best_improvement:+.1f}%")
    
    # Find best configurations
    best_vad_f1 = vad_df.loc[vad_df['binary_f1'].idxmax()]
    best_vad_iou = vad_df.loc[vad_df['combined_iou'].idxmax()]
    
    print(f"\n🏆 BEST VAD CONFIGURATIONS:")
    print(f"   Best F1: {best_vad_f1['binary_f1']:.3f} ({best_vad_f1['eval_type']}, {best_vad_f1['window_name']}, {best_vad_f1['stride_name']})")
    print(f"   Best IoU: {best_vad_iou['combined_iou']:.3f} ({best_vad_iou['eval_type']}, {best_vad_iou['window_name']}, {best_vad_iou['stride_name']})")
    
    # Save detailed report
    report_path = output_dir / "detailed_comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write("VAD REFINED vs COMPREHENSIVE EVALUATION DETAILED COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET SUMMARY:\n")
        f.write(f"VAD Refined Results: {len(vad_df)} configurations\n")
        f.write(f"Comprehensive Results: {len(comp_df)} configurations\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"VAD Binary F1: {vad_df['binary_f1'].mean():.3f} ± {vad_df['binary_f1'].std():.3f}\n")
        f.write(f"Comprehensive Binary F1: {comp_df['binary_f1'].mean():.3f} ± {comp_df['binary_f1'].std():.3f}\n")
        f.write(f"Binary F1 Improvement: {f1_mean_improvement:+.1f}%\n\n")
        
        f.write(f"VAD IoU: {vad_df['combined_iou'].mean():.3f} ± {vad_df['combined_iou'].std():.3f}\n")
        f.write(f"Comprehensive IoU: {comp_df['combined_iou'].mean():.3f} ± {comp_df['combined_iou'].std():.3f}\n")
        f.write(f"IoU Improvement: {iou_mean_improvement:+.1f}%\n\n")
        
        f.write("BEST CONFIGURATIONS:\n")
        f.write(f"Best VAD F1: {best_vad_f1['binary_f1']:.3f} ({best_vad_f1['eval_type']}, {best_vad_f1['window_name']}, {best_vad_f1['stride_name']})\n")
        f.write(f"Best VAD IoU: {best_vad_iou['combined_iou']:.3f} ({best_vad_iou['eval_type']}, {best_vad_iou['window_name']}, {best_vad_iou['stride_name']})\n")
    
    print(f"📄 Detailed report saved to {report_path}")

def main():
    """Main function"""
    try:
        create_vad_vs_comprehensive_comparison()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
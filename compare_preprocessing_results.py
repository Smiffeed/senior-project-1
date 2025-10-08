#!/usr/bin/env python3
"""
Comparative Analysis: Comprehensive (Advanced) vs Frame-Level (Simple) Results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_compare_results():
    """Load and compare comprehensive vs frame-level results"""
    
    print("🔍 COMPREHENSIVE vs FRAME-LEVEL COMPARISON")
    print("=" * 50)
    
    # Load comprehensive results (advanced preprocessing)
    comp_path = Path('fixed_smart_parallel_results/detailed_analysis/complete_analysis_data.csv')
    frame_path = Path('frame_level_evaluation_results')
    
    if not comp_path.exists():
        print("❌ Comprehensive results not found")
        return
    
    comp_df = pd.read_csv(comp_path)
    print(f"📊 Comprehensive results: {len(comp_df)} configurations")
    
    # Load frame-level results summary if available
    frame_summary_files = [
        'frame_level_evaluation_results/frame_level_summary.csv',
        'frame_level_evaluation_results/evaluation_summary.csv'
    ]
    
    frame_df = None
    for frame_file in frame_summary_files:
        if Path(frame_file).exists():
            frame_df = pd.read_csv(frame_file)
            print(f"📊 Frame-level results: {len(frame_df)} configurations")
            break
    
    if frame_df is None:
        print("⚠️  No frame-level summary found - creating comparison with available data")
        create_preprocessing_impact_analysis(comp_df)
    else:
        create_full_comparison(comp_df, frame_df)

def create_preprocessing_impact_analysis(comp_df):
    """Analyze preprocessing impact using comprehensive results only"""
    
    print("\n📈 PREPROCESSING IMPACT ANALYSIS")
    print("-" * 35)
    
    # Group by window size for analysis
    window_analysis = comp_df.groupby('window_numeric').agg({
        'window_f1': ['mean', 'std', 'min', 'max'],
        'word_f1': ['mean', 'std', 'min', 'max'],
        'combined_iou': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print("📊 Performance by Window Size (Advanced Preprocessing):")
    print(window_analysis)
    
    # Best configurations
    print(f"\n🏆 TOP PERFORMING CONFIGURATIONS:")
    print(f"Best Window F1: {comp_df.loc[comp_df['window_f1'].idxmax()][['window', 'stride', 'window_f1']].values}")
    print(f"Best Word F1: {comp_df.loc[comp_df['word_f1'].idxmax()][['window', 'stride', 'word_f1']].values}")
    print(f"Best IoU: {comp_df.loc[comp_df['combined_iou'].idxmax()][['window', 'stride', 'combined_iou']].values}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Evaluation Results (Advanced Preprocessing)', fontsize=16, fontweight='bold')
    
    # Window F1 by window size
    ax1 = axes[0, 0]
    window_f1_stats = comp_df.groupby('window_numeric')['window_f1'].agg(['mean', 'std']).reset_index()
    ax1.errorbar(window_f1_stats['window_numeric'], window_f1_stats['mean'], 
                yerr=window_f1_stats['std'], marker='o', capsize=5)
    ax1.set_title('Window F1 Score by Window Size')
    ax1.set_xlabel('Window Size (seconds)')
    ax1.set_ylabel('Window F1 Score')
    ax1.grid(True, alpha=0.3)
    
    # Word F1 by window size
    ax2 = axes[0, 1]
    word_f1_stats = comp_df.groupby('window_numeric')['word_f1'].agg(['mean', 'std']).reset_index()
    ax2.errorbar(word_f1_stats['window_numeric'], word_f1_stats['mean'], 
                yerr=word_f1_stats['std'], marker='s', color='red', capsize=5)
    ax2.set_title('Word F1 Score by Window Size')
    ax2.set_xlabel('Window Size (seconds)')
    ax2.set_ylabel('Word F1 Score')
    ax2.grid(True, alpha=0.3)
    
    # IoU by window size
    ax3 = axes[1, 0]
    iou_stats = comp_df.groupby('window_numeric')['combined_iou'].agg(['mean', 'std']).reset_index()
    ax3.errorbar(iou_stats['window_numeric'], iou_stats['mean'], 
                yerr=iou_stats['std'], marker='^', color='green', capsize=5)
    ax3.set_title('Combined IoU by Window Size')
    ax3.set_xlabel('Window Size (seconds)')
    ax3.set_ylabel('Combined IoU')
    ax3.grid(True, alpha=0.3)
    
    # Correlation heatmap
    ax4 = axes[1, 1]
    corr_data = comp_df[['window_numeric', 'stride_numeric', 'window_f1', 'word_f1', 'combined_iou']].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax4)
    ax4.set_title('Metric Correlations')
    
    plt.tight_layout()
    plt.savefig('comprehensive_preprocessing_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💡 INSIGHTS FROM ADVANCED PREPROCESSING:")
    print(f"   • Window F1 range: {comp_df['window_f1'].min():.4f} - {comp_df['window_f1'].max():.4f}")
    print(f"   • Word F1 range: {comp_df['word_f1'].min():.4f} - {comp_df['word_f1'].max():.4f}")
    print(f"   • IoU range: {comp_df['combined_iou'].min():.4f} - {comp_df['combined_iou'].max():.4f}")
    print(f"   • Best window sizes: {comp_df.groupby('window_numeric')['word_f1'].mean().nlargest(3).index.tolist()}")

def create_full_comparison(comp_df, frame_df):
    """Create full comparison between comprehensive and frame-level results"""
    print("\n🔄 FULL COMPARISON ANALYSIS")
    print("-" * 30)
    # Implementation for full comparison when frame-level data is available
    pass

if __name__ == "__main__":
    load_and_compare_results()
#!/usr/bin/env python3
"""
Create line graphs showing Window Size vs F1 Score trends
Generates separate graphs for Binary (Window F1) and Multiclass/Word F1 classification
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def load_and_prepare_data():
    """Load and prepare the analysis data"""
    data_path = Path("fixed_smart_parallel_results/correlation_analysis/complete_analysis_data.csv")
    df = pd.read_csv(data_path)
    
    # Filter successful evaluations only
    df_success = df[df['success'] == True].copy()
    
    return df_success

def create_window_f1_trend_graphs(df):
    """Create line graphs for Window Size vs F1 Score trends"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    output_dir = Path("fixed_smart_parallel_results/f1_trend_graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique window sizes
    window_sizes = sorted(df['window_numeric'].unique())
    
    # Calculate statistics for each window size
    window_stats = []
    for window in window_sizes:
        window_data = df[df['window_numeric'] == window]
        
        stats = {
            'window_size': window,
            'window_f1_mean': window_data['window_f1'].mean(),
            'window_f1_max': window_data['window_f1'].max(),
            'window_f1_min': window_data['window_f1'].min(),
            'window_f1_std': window_data['window_f1'].std(),
            'word_f1_mean': window_data['word_f1'].mean(),
            'word_f1_max': window_data['word_f1'].max(),
            'word_f1_min': window_data['word_f1'].min(),
            'word_f1_std': window_data['word_f1'].std(),
            'count': len(window_data)
        }
        window_stats.append(stats)
    
    stats_df = pd.DataFrame(window_stats)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # ============ GRAPH 1: BINARY CLASSIFICATION (Window F1) ============
    ax1.plot(stats_df['window_size'], stats_df['window_f1_mean'], 
             marker='o', linewidth=3, markersize=8, color='#2E86AB', label='Mean Window F1')
    ax1.plot(stats_df['window_size'], stats_df['window_f1_max'], 
             marker='^', linewidth=2, markersize=6, color='#A23B72', linestyle='--', label='Max Window F1')
    ax1.plot(stats_df['window_size'], stats_df['window_f1_min'], 
             marker='v', linewidth=2, markersize=6, color='#F18F01', linestyle='--', label='Min Window F1')
    
    # Add error bars for standard deviation
    ax1.fill_between(stats_df['window_size'], 
                     stats_df['window_f1_mean'] - stats_df['window_f1_std'],
                     stats_df['window_f1_mean'] + stats_df['window_f1_std'],
                     alpha=0.2, color='#2E86AB')
    
    # Find the best window size for Window F1
    best_window_f1_idx = stats_df['window_f1_max'].idxmax()
    best_window_f1_size = stats_df.loc[best_window_f1_idx, 'window_size']
    best_window_f1_score = stats_df.loc[best_window_f1_idx, 'window_f1_max']
    
    # Highlight the best point
    ax1.plot(best_window_f1_size, best_window_f1_score, 
             marker='*', markersize=15, color='red', markeredgecolor='black', markeredgewidth=1)
    ax1.annotate(f'Best: {best_window_f1_size}s\n({best_window_f1_score:.4f})', 
                xy=(best_window_f1_size, best_window_f1_score),
                xytext=(best_window_f1_size + 0.2, best_window_f1_score + 0.02),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, fontweight='bold', color='red')
    
    ax1.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Binary Classification F1 Score', fontsize=12, fontweight='bold')
    ax1.set_title('Window Size vs Binary Classification (Window F1) Performance Trend', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    
    # Set x-axis to show all window sizes
    ax1.set_xticks(window_sizes)
    ax1.set_xlim(window_sizes[0] - 0.1, window_sizes[-1] + 0.1)
    
    # Add trend information
    trend_text = f"Trend: Strong positive correlation (+0.61)\nLarger windows → Better Binary F1"
    ax1.text(0.02, 0.98, trend_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
             verticalalignment='top', fontsize=9)
    
    # ============ GRAPH 2: MULTICLASS/WORD CLASSIFICATION (Word F1) ============
    ax2.plot(stats_df['window_size'], stats_df['word_f1_mean'], 
             marker='o', linewidth=3, markersize=8, color='#F18F01', label='Mean Word F1')
    ax2.plot(stats_df['window_size'], stats_df['word_f1_max'], 
             marker='^', linewidth=2, markersize=6, color='#C73E1D', linestyle='--', label='Max Word F1')
    ax2.plot(stats_df['window_size'], stats_df['word_f1_min'], 
             marker='v', linewidth=2, markersize=6, color='#2E86AB', linestyle='--', label='Min Word F1')
    
    # Add error bars for standard deviation
    ax2.fill_between(stats_df['window_size'], 
                     stats_df['word_f1_mean'] - stats_df['word_f1_std'],
                     stats_df['word_f1_mean'] + stats_df['word_f1_std'],
                     alpha=0.2, color='#F18F01')
    
    # Find the best window size for Word F1
    best_word_f1_idx = stats_df['word_f1_max'].idxmax()
    best_word_f1_size = stats_df.loc[best_word_f1_idx, 'window_size']
    best_word_f1_score = stats_df.loc[best_word_f1_idx, 'word_f1_max']
    
    # Highlight the best point
    ax2.plot(best_word_f1_size, best_word_f1_score, 
             marker='*', markersize=15, color='red', markeredgecolor='black', markeredgewidth=1)
    ax2.annotate(f'Best: {best_word_f1_size}s\n({best_word_f1_score:.4f})', 
                xy=(best_word_f1_size, best_word_f1_score),
                xytext=(best_word_f1_size + 0.2, best_word_f1_score - 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, fontweight='bold', color='red')
    
    ax2.set_xlabel('Window Size (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Multiclass/Word Classification F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Window Size vs Multiclass/Word Classification (Word F1) Performance Trend', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Set x-axis to show all window sizes
    ax2.set_xticks(window_sizes)
    ax2.set_xlim(window_sizes[0] - 0.1, window_sizes[-1] + 0.1)
    
    # Add trend information
    trend_text = f"Trend: Strong negative correlation (-0.90)\nSmaller windows → Better Word F1"
    ax2.text(0.02, 0.02, trend_text, transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
             verticalalignment='bottom', fontsize=9)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / "window_size_f1_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============ SEPARATE INDIVIDUAL GRAPHS ============
    
    # Individual Graph 1: Binary Classification (Window F1)
    plt.figure(figsize=(10, 6))
    plt.plot(stats_df['window_size'], stats_df['window_f1_mean'], 
             marker='o', linewidth=3, markersize=10, color='#2E86AB', label='Mean Window F1')
    plt.plot(stats_df['window_size'], stats_df['window_f1_max'], 
             marker='^', linewidth=2, markersize=8, color='#A23B72', linestyle='--', label='Max Window F1')
    
    # Fill area between min and max
    plt.fill_between(stats_df['window_size'], stats_df['window_f1_min'], stats_df['window_f1_max'],
                     alpha=0.2, color='#2E86AB', label='Min-Max Range')
    
    # Highlight best point
    plt.plot(best_window_f1_size, best_window_f1_score, 
             marker='*', markersize=20, color='red', markeredgecolor='black', markeredgewidth=2)
    plt.annotate(f'Optimal Window Size: {best_window_f1_size}s\nBest F1 Score: {best_window_f1_score:.4f}', 
                xy=(best_window_f1_size, best_window_f1_score),
                xytext=(best_window_f1_size + 0.3, best_window_f1_score + 0.03),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8))
    
    plt.xlabel('Window Size (seconds)', fontsize=14, fontweight='bold')
    plt.ylabel('Binary Classification F1 Score', fontsize=14, fontweight='bold')
    plt.title('Binary Classification Performance: Window Size Impact', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=12)
    plt.xticks(window_sizes)
    
    # Add statistics box
    stats_text = f"Correlation: +0.61 (Strong Positive)\nTrend: Larger windows perform better\nRange: {stats_df['window_f1_min'].min():.3f} - {stats_df['window_f1_max'].max():.3f}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
             verticalalignment='top', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "binary_classification_window_trend.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual Graph 2: Multiclass/Word Classification (Word F1)
    plt.figure(figsize=(10, 6))
    plt.plot(stats_df['window_size'], stats_df['word_f1_mean'], 
             marker='o', linewidth=3, markersize=10, color='#F18F01', label='Mean Word F1')
    plt.plot(stats_df['window_size'], stats_df['word_f1_max'], 
             marker='^', linewidth=2, markersize=8, color='#C73E1D', linestyle='--', label='Max Word F1')
    
    # Fill area between min and max
    plt.fill_between(stats_df['window_size'], stats_df['word_f1_min'], stats_df['word_f1_max'],
                     alpha=0.2, color='#F18F01', label='Min-Max Range')
    
    # Highlight best point
    plt.plot(best_word_f1_size, best_word_f1_score, 
             marker='*', markersize=20, color='red', markeredgecolor='black', markeredgewidth=2)
    plt.annotate(f'Optimal Window Size: {best_word_f1_size}s\nBest F1 Score: {best_word_f1_score:.4f}', 
                xy=(best_word_f1_size, best_word_f1_score),
                xytext=(best_word_f1_size + 0.3, best_word_f1_score - 0.15),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8))
    
    plt.xlabel('Window Size (seconds)', fontsize=14, fontweight='bold')
    plt.ylabel('Multiclass/Word Classification F1 Score', fontsize=14, fontweight='bold')
    plt.title('Multiclass/Word Classification Performance: Window Size Impact', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=12)
    plt.xticks(window_sizes)
    
    # Add statistics box
    stats_text = f"Correlation: -0.90 (Strong Negative)\nTrend: Smaller windows perform better\nRange: {stats_df['word_f1_min'].min():.3f} - {stats_df['word_f1_max'].max():.3f}"
    plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
             verticalalignment='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "multiclass_word_classification_window_trend.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats_df, output_dir

def create_summary_table(stats_df, output_dir):
    """Create a summary table of the results"""
    
    # Create summary table
    summary_table = stats_df[['window_size', 'window_f1_mean', 'window_f1_max', 
                             'word_f1_mean', 'word_f1_max', 'count']].copy()
    summary_table = summary_table.round(4)
    
    # Save to CSV
    summary_table.to_csv(output_dir / "window_size_f1_summary.csv", index=False)
    
    # Create a formatted text summary
    with open(output_dir / "f1_trend_analysis_summary.txt", 'w') as f:
        f.write("=== WINDOW SIZE vs F1 SCORE TREND ANALYSIS ===\n\n")
        
        f.write("BINARY CLASSIFICATION (Window F1) RESULTS:\n")
        f.write("-" * 50 + "\n")
        best_binary_idx = stats_df['window_f1_max'].idxmax()
        f.write(f"Best Window Size: {stats_df.loc[best_binary_idx, 'window_size']}s\n")
        f.write(f"Best F1 Score: {stats_df.loc[best_binary_idx, 'window_f1_max']:.4f}\n")
        f.write(f"Mean F1 at Best Window: {stats_df.loc[best_binary_idx, 'window_f1_mean']:.4f}\n")
        f.write(f"Correlation with Window Size: +0.61 (Strong Positive)\n")
        f.write(f"Trend: Larger windows perform better for binary classification\n\n")
        
        f.write("MULTICLASS/WORD CLASSIFICATION (Word F1) RESULTS:\n")
        f.write("-" * 50 + "\n")
        best_word_idx = stats_df['word_f1_max'].idxmax()
        f.write(f"Best Window Size: {stats_df.loc[best_word_idx, 'window_size']}s\n")
        f.write(f"Best F1 Score: {stats_df.loc[best_word_idx, 'word_f1_max']:.4f}\n")
        f.write(f"Mean F1 at Best Window: {stats_df.loc[best_word_idx, 'word_f1_mean']:.4f}\n")
        f.write(f"Correlation with Window Size: -0.90 (Strong Negative)\n")
        f.write(f"Trend: Smaller windows perform better for word classification\n\n")
        
        f.write("DETAILED WINDOW SIZE ANALYSIS:\n")
        f.write("-" * 50 + "\n")
        f.write("Window | Binary F1 (Mean/Max) | Word F1 (Mean/Max) | Count\n")
        f.write("-" * 60 + "\n")
        
        for _, row in stats_df.iterrows():
            f.write(f"{row['window_size']:4.1f}s  | {row['window_f1_mean']:.3f}/{row['window_f1_max']:.3f}     | "
                   f"{row['word_f1_mean']:.3f}/{row['word_f1_max']:.3f}     | {row['count']:3.0f}\n")
    
    print(f"📊 Summary table saved to: {output_dir / 'window_size_f1_summary.csv'}")
    print(f"📄 Detailed analysis saved to: {output_dir / 'f1_trend_analysis_summary.txt'}")

def main():
    print("=== CREATING WINDOW SIZE vs F1 SCORE TREND GRAPHS ===\n")
    
    # Load data
    print("📊 Loading analysis data...")
    df = load_and_prepare_data()
    print(f"✅ Loaded {len(df)} successful evaluations")
    
    # Create trend graphs
    print("📈 Creating F1 score trend graphs...")
    stats_df, output_dir = create_window_f1_trend_graphs(df)
    
    # Create summary table
    print("📋 Creating summary table...")
    create_summary_table(stats_df, output_dir)
    
    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📁 All graphs and analysis saved to: {output_dir}")
    print(f"📊 Generated files:")
    print(f"   • window_size_f1_trends.png (Combined graph)")
    print(f"   • binary_classification_window_trend.png (Binary F1 only)")
    print(f"   • multiclass_word_classification_window_trend.png (Word F1 only)")
    print(f"   • window_size_f1_summary.csv (Data table)")
    print(f"   • f1_trend_analysis_summary.txt (Detailed summary)")
    
    # Print key findings
    print(f"\n💡 KEY FINDINGS:")
    best_binary_idx = stats_df['window_f1_max'].idxmax()
    best_word_idx = stats_df['word_f1_max'].idxmax()
    
    print(f"🏆 Best Binary Classification (Window F1):")
    print(f"   Window Size: {stats_df.loc[best_binary_idx, 'window_size']}s")
    print(f"   F1 Score: {stats_df.loc[best_binary_idx, 'window_f1_max']:.4f}")
    
    print(f"🏆 Best Word Classification (Word F1):")
    print(f"   Window Size: {stats_df.loc[best_word_idx, 'window_size']}s")
    print(f"   F1 Score: {stats_df.loc[best_word_idx, 'word_f1_max']:.4f}")
    
    print(f"\n📈 TRENDS:")
    print(f"   Binary F1: Increases with larger windows (+0.61 correlation)")
    print(f"   Word F1: Decreases with larger windows (-0.90 correlation)")
    print(f"   ⚖️ Trade-off: Choose window size based on your priority!")

if __name__ == "__main__":
    main()

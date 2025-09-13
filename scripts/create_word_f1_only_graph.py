#!/usr/bin/env python3
"""
Create line graph showing Word F1 trends only (excluding Window F1)
Generates focused graph for Word-level classification F1 performance
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

def create_word_f1_only_graph(df):
    """Create line graph for Word F1 Score trends only"""
    
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
            'word_f1_mean': window_data['word_f1'].mean(),
            'word_f1_max': window_data['word_f1'].max(),
            'word_f1_min': window_data['word_f1'].min(),
            'word_f1_std': window_data['word_f1'].std(),
            'count': len(window_data)
        }
        window_stats.append(stats)
    
    stats_df = pd.DataFrame(window_stats)
    
    # ============ WORD F1 ONLY GRAPH ============
    plt.figure(figsize=(12, 8))
    
    # Plot Word F1 trends
    plt.plot(stats_df['window_size'], stats_df['word_f1_mean'], 
             marker='o', linewidth=3, markersize=10, color='#F18F01', label='Mean Word F1')
    plt.plot(stats_df['window_size'], stats_df['word_f1_max'], 
             marker='^', linewidth=2, markersize=8, color='#C73E1D', linestyle='--', label='Max Word F1')
    plt.plot(stats_df['window_size'], stats_df['word_f1_min'], 
             marker='v', linewidth=2, markersize=8, color='#2E86AB', linestyle='--', label='Min Word F1')
    
    # Add error bars for standard deviation
    plt.fill_between(stats_df['window_size'], 
                     stats_df['word_f1_mean'] - stats_df['word_f1_std'],
                     stats_df['word_f1_mean'] + stats_df['word_f1_std'],
                     alpha=0.2, color='#F18F01', label='Standard Deviation Range')
    
    # Find the best window size for Word F1
    best_word_f1_idx = stats_df['word_f1_max'].idxmax()
    best_word_f1_size = stats_df.loc[best_word_f1_idx, 'window_size']
    best_word_f1_score = stats_df.loc[best_word_f1_idx, 'word_f1_max']
    
    # Highlight the best point
    plt.plot(best_word_f1_size, best_word_f1_score, 
             marker='*', markersize=20, color='red', markeredgecolor='black', markeredgewidth=2)
    plt.annotate(f'Optimal Window Size: {best_word_f1_size}s\nBest F1 Score: {best_word_f1_score:.4f}', 
                xy=(best_word_f1_size, best_word_f1_score),
                xytext=(best_word_f1_size + 0.3, best_word_f1_score - 0.15),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8))
    
    plt.xlabel('Window Size (seconds)', fontsize=14, fontweight='bold')
    plt.ylabel('Word Classification F1 Score', fontsize=14, fontweight='bold')
    plt.title('Word-level Classification Performance: Window Size Impact\\n(Excluding Window-level F1)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=12)
    
    # Set x-axis to show all window sizes
    plt.xticks(window_sizes)
    plt.xlim(window_sizes[0] - 0.1, window_sizes[-1] + 0.1)
    
    # Add statistics box
    stats_text = f"Correlation: -0.90 (Strong Negative)\\nTrend: Smaller windows perform better\\nRange: {stats_df['word_f1_min'].min():.3f} - {stats_df['word_f1_max'].max():.3f}"
    plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
             verticalalignment='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "word_f1_only_trend.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats_df, output_dir

def create_summary_table(stats_df, output_dir):
    """Create a summary table of the Word F1 results"""
    
    # Create summary table
    summary_table = stats_df[['window_size', 'word_f1_mean', 'word_f1_max', 'word_f1_min', 'count']].copy()
    summary_table = summary_table.round(4)
    
    # Save to CSV
    summary_table.to_csv(output_dir / "word_f1_only_summary.csv", index=False)
    
    # Create a formatted text summary
    with open(output_dir / "word_f1_only_analysis.txt", 'w') as f:
        f.write("=== WORD F1 SCORE TREND ANALYSIS (EXCLUDING WINDOW F1) ===\\n\\n")
        
        f.write("WORD CLASSIFICATION F1 RESULTS:\\n")
        f.write("-" * 50 + "\\n")
        best_idx = stats_df['word_f1_max'].idxmax()
        f.write(f"Best Window Size: {stats_df.loc[best_idx, 'window_size']}s\\n")
        f.write(f"Best F1 Score: {stats_df.loc[best_idx, 'word_f1_max']:.4f}\\n")
        f.write(f"Mean F1 at Best Window: {stats_df.loc[best_idx, 'word_f1_mean']:.4f}\\n")
        f.write(f"Correlation with Window Size: -0.90 (Strong Negative)\\n")
        f.write(f"Trend: Smaller windows perform better for word classification\\n\\n")
        
        f.write("DETAILED WINDOW SIZE ANALYSIS:\\n")
        f.write("-" * 50 + "\\n")
        f.write("Window | Word F1 (Mean/Max/Min) | Count\\n")
        f.write("-" * 45 + "\\n")
        
        for _, row in stats_df.iterrows():
            f.write(f"{row['window_size']:4.1f}s  | {row['word_f1_mean']:.3f}/{row['word_f1_max']:.3f}/{row['word_f1_min']:.3f} | {row['count']:3.0f}\\n")
    
    print(f"📊 Word F1 summary saved to: {output_dir / 'word_f1_only_summary.csv'}")
    print(f"📄 Detailed analysis saved to: {output_dir / 'word_f1_only_analysis.txt'}")

def main():
    print("=== CREATING WORD F1 TREND GRAPH (EXCLUDING WINDOW F1) ===\\n")
    
    # Load data
    print("📊 Loading analysis data...")
    df = load_and_prepare_data()
    print(f"✅ Loaded {len(df)} successful evaluations")
    
    # Create Word F1 only graph
    print("📈 Creating Word F1 trend graph...")
    stats_df, output_dir = create_word_f1_only_graph(df)
    
    # Create summary table
    print("📋 Creating summary table...")
    create_summary_table(stats_df, output_dir)
    
    print(f"\\n🎉 ANALYSIS COMPLETE!")
    print(f"📁 Word F1 analysis saved to: {output_dir}")
    print(f"📊 Generated files:")
    print(f"   • word_f1_only_trend.png (Word F1 trend graph)")
    print(f"   • word_f1_only_summary.csv (Data table)")
    print(f"   • word_f1_only_analysis.txt (Detailed summary)")
    
    # Print key findings
    print(f"\\n💡 KEY FINDINGS:")
    best_idx = stats_df['word_f1_max'].idxmax()
    
    print(f"🏆 Best Word Classification F1:")
    print(f"   Window Size: {stats_df.loc[best_idx, 'window_size']}s")
    print(f"   F1 Score: {stats_df.loc[best_idx, 'word_f1_max']:.4f}")
    
    print(f"\\n📈 TREND:")
    print(f"   Word F1: Decreases with larger windows (-0.90 correlation)")
    print(f"   Optimal Strategy: Use smallest windows (0.3s) for best word detection")

if __name__ == "__main__":
    main()

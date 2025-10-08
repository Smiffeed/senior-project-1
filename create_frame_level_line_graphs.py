#!/usr/bin/env python3
"""
📊 FRAME-LEVEL EVALUATION LINE GRAPHS GENERATOR
Generate F1 score and window size line graphs for frame-level evaluation results

Features:
- Binary and Multiclass F1 score line graphs by window size
- IoU performance analysis by window size
- Comparison between eval_percent and eval_by_0.05 methods
- Professional publication-ready visualizations
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

class FrameLevelGraphGenerator:
    """Generate line graphs for frame-level evaluation results"""
    
    def __init__(self, results_dir: str):
        """Initialize with results directory"""
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "frame_level_line_graphs"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load evaluation summary
        self.summary_df = self._load_evaluation_summary()
        print(f"✅ Loaded {len(self.summary_df)} evaluation results")
    
    def _load_evaluation_summary(self):
        """Load the evaluation summary CSV"""
        summary_path = self.results_dir / "evaluation_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Evaluation summary not found: {summary_path}")
        
        df = pd.read_csv(summary_path)
        
        # Clean and process data
        df = df[df['success'] == True].copy()  # Only successful evaluations
        df['window_size'] = pd.to_numeric(df['window_size'], errors='coerce')
        df['binary_f1'] = pd.to_numeric(df['binary_f1'], errors='coerce')
        df['multiclass_f1'] = pd.to_numeric(df['multiclass_f1'], errors='coerce')
        df['combined_mean_iou'] = pd.to_numeric(df['combined_mean_iou'], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna(subset=['window_size', 'binary_f1', 'multiclass_f1', 'combined_mean_iou'])
        
        return df
    
    def create_f1_by_window_size_graphs(self):
        """Create F1 score line graphs by window size - Combined Binary and Multiclass"""
        print("📊 Creating combined F1 score by window size graphs...")
        
        # Use only eval_percent data (percentage stride)
        eval_percent_df = self.summary_df[self.summary_df['eval_type'] == 'eval_percent'].copy()
        
        if eval_percent_df.empty:
            print("⚠️ No percentage stride data found!")
            return
        
        # Create single figure for combined F1 scores
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('Frame-Level Evaluation: F1 Scores by Window Size (Percentage Stride)', fontsize=16, fontweight='bold')
        
        # Plot both binary and multiclass F1 on the same graph
        self._plot_combined_f1_by_window(eval_percent_df, ax)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.output_dir / 'frame_level_f1_by_window_size.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'frame_level_f1_by_window_size.pdf', bbox_inches='tight')
        plt.show()
        
        print(f"✅ Combined F1 by window size graph saved to {self.output_dir}")
    
    def _plot_combined_f1_by_window(self, df, ax):
        """Plot both binary and multiclass F1 scores by window size on the same graph"""
        # Group by window size and calculate statistics for both metrics
        binary_stats = df.groupby('window_size')['binary_f1'].agg(['mean', 'std', 'count']).reset_index()
        binary_stats['sem'] = binary_stats['std'] / np.sqrt(binary_stats['count'])
        
        multiclass_stats = df.groupby('window_size')['multiclass_f1'].agg(['mean', 'std', 'count']).reset_index()
        multiclass_stats['sem'] = multiclass_stats['std'] / np.sqrt(multiclass_stats['count'])
        
        # Plot binary F1
        ax.errorbar(binary_stats['window_size'], binary_stats['mean'], 
                   yerr=binary_stats['sem'], 
                   marker='o', linewidth=2.5, markersize=7, capsize=5, 
                   label='Binary F1', color='#2E86AB', alpha=0.8)
        
        # Plot multiclass F1
        ax.errorbar(multiclass_stats['window_size'], multiclass_stats['mean'], 
                   yerr=multiclass_stats['sem'], 
                   marker='s', linewidth=2.5, markersize=7, capsize=5, 
                   label='Multiclass F1', color='#A23B72', alpha=0.8)
        
        # Add scatter points for individual measurements (with transparency)
        ax.scatter(df['window_size'], df['binary_f1'], alpha=0.2, s=15, color='#2E86AB')
        ax.scatter(df['window_size'], df['multiclass_f1'], alpha=0.2, s=15, color='#A23B72')
        
        # Add trend lines
        if len(binary_stats) > 1:
            z_binary = np.polyfit(binary_stats['window_size'], binary_stats['mean'], 1)
            p_binary = np.poly1d(z_binary)
            ax.plot(binary_stats['window_size'], p_binary(binary_stats['window_size']), 
                   "--", alpha=0.6, linewidth=1.5, color='#2E86AB')
        
        if len(multiclass_stats) > 1:
            z_multi = np.polyfit(multiclass_stats['window_size'], multiclass_stats['mean'], 1)
            p_multi = np.poly1d(z_multi)
            ax.plot(multiclass_stats['window_size'], p_multi(multiclass_stats['window_size']), 
                   "--", alpha=0.6, linewidth=1.5, color='#A23B72')
        
        # Formatting
        ax.set_xlabel('Window Size (seconds)', fontsize=14, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
        ax.set_title('Binary vs Multiclass F1 Performance by Window Size', fontsize=15, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        
        # Add performance annotations
        best_binary_idx = binary_stats['mean'].idxmax()
        best_multiclass_idx = multiclass_stats['mean'].idxmax()
        
        best_binary_window = binary_stats.iloc[best_binary_idx]['window_size']
        best_binary_score = binary_stats.iloc[best_binary_idx]['mean']
        
        best_multiclass_window = multiclass_stats.iloc[best_multiclass_idx]['window_size']
        best_multiclass_score = multiclass_stats.iloc[best_multiclass_idx]['mean']
        
        ax.annotate(f'Best Binary F1: {best_binary_score:.3f}\n@ {best_binary_window}s window',
                   xy=(best_binary_window, best_binary_score), 
                   xytext=(best_binary_window + 0.2, best_binary_score + 0.05),
                   arrowprops=dict(arrowstyle='->', color='#2E86AB', alpha=0.7),
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.annotate(f'Best Multiclass F1: {best_multiclass_score:.3f}\n@ {best_multiclass_window}s window',
                   xy=(best_multiclass_window, best_multiclass_score), 
                   xytext=(best_multiclass_window + 0.5, best_multiclass_score - 0.08),
                   arrowprops=dict(arrowstyle='->', color='#A23B72', alpha=0.7, connectionstyle='arc3,rad=-0.2'),
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    def _plot_f1_by_window(self, df, f1_column, title, ax):
        """Plot F1 scores by window size with error bars"""
        # Group by window size and calculate statistics
        window_stats = df.groupby('window_size')[f1_column].agg(['mean', 'std', 'count']).reset_index()
        window_stats['sem'] = window_stats['std'] / np.sqrt(window_stats['count'])  # Standard error
        
        # Create line plot with error bars
        ax.errorbar(window_stats['window_size'], window_stats['mean'], 
                   yerr=window_stats['sem'], 
                   marker='o', linewidth=2, markersize=6, capsize=5)
        
        # Scatter plot for individual points (with some transparency)
        ax.scatter(df['window_size'], df[f1_column], alpha=0.3, s=20, color='gray')
        
        ax.set_xlabel('Window Size (seconds)', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add trend line
        if len(window_stats) > 1:
            z = np.polyfit(window_stats['window_size'], window_stats['mean'], 1)
            p = np.poly1d(z)
            ax.plot(window_stats['window_size'], p(window_stats['window_size']), 
                   "--", alpha=0.7, linewidth=1.5, color='red')
    
    def create_iou_by_window_size_graphs(self):
        """Create IoU line graphs by window size - Percentage Stride Only"""
        print("📊 Creating IoU by window size graphs...")
        
        # Use only eval_percent data (percentage stride)
        eval_percent_df = self.summary_df[self.summary_df['eval_type'] == 'eval_percent'].copy()
        
        if eval_percent_df.empty:
            print("⚠️ No percentage stride data found!")
            return
        
        # Create single figure for IoU
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('Frame-Level Evaluation: IoU Performance by Window Size (Percentage Stride)', fontsize=16, fontweight='bold')
        
        # Plot IoU - Percentage Stride only
        self._plot_iou_by_window(eval_percent_df, 'Combined Mean IoU Performance', ax)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'frame_level_iou_by_window_size.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'frame_level_iou_by_window_size.pdf', bbox_inches='tight')
        plt.show()
        
        print(f"✅ IoU by window size graph saved to {self.output_dir}")
    
    def _plot_iou_by_window(self, df, title, ax):
        """Plot IoU by window size with error bars"""
        # Group by window size and calculate statistics
        window_stats = df.groupby('window_size')['combined_mean_iou'].agg(['mean', 'std', 'count']).reset_index()
        window_stats['sem'] = window_stats['std'] / np.sqrt(window_stats['count'])
        
        # Create line plot with error bars
        ax.errorbar(window_stats['window_size'], window_stats['mean'], 
                   yerr=window_stats['sem'], 
                   marker='o', linewidth=2, markersize=6, capsize=5, color='green')
        
        # Scatter plot for individual points
        ax.scatter(df['window_size'], df['combined_mean_iou'], alpha=0.3, s=20, color='gray')
        
        ax.set_xlabel('Window Size (seconds)', fontsize=12)
        ax.set_ylabel('Combined Mean IoU (%)', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(window_stats) > 1:
            z = np.polyfit(window_stats['window_size'], window_stats['mean'], 1)
            p = np.poly1d(z)
            ax.plot(window_stats['window_size'], p(window_stats['window_size']), 
                   "--", alpha=0.7, linewidth=1.5, color='red')
    
    def create_comparison_graphs(self):
        """Create comprehensive performance analysis graphs"""
        print("📊 Creating performance analysis graphs...")
        
        # Use only eval_percent data (percentage stride)
        eval_percent_df = self.summary_df[self.summary_df['eval_type'] == 'eval_percent'].copy()
        
        if eval_percent_df.empty:
            print("⚠️ No percentage stride data found!")
            return
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Frame-Level Evaluation: Performance Analysis by Window Size (Percentage Stride)', fontsize=16, fontweight='bold')
        
        # Calculate mean values by window size
        percent_stats = eval_percent_df.groupby('window_size')[['binary_f1', 'multiclass_f1', 'combined_mean_iou']].mean().reset_index()
        
        # 1. Combined F1 Scores
        ax1.plot(percent_stats['window_size'], percent_stats['binary_f1'], 
                marker='o', linewidth=2.5, markersize=7, label='Binary F1', color='#2E86AB')
        ax1.plot(percent_stats['window_size'], percent_stats['multiclass_f1'], 
                marker='s', linewidth=2.5, markersize=7, label='Multiclass F1', color='#A23B72')
        ax1.set_title('F1 Score Performance', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Window Size (seconds)', fontweight='bold')
        ax1.set_ylabel('F1 Score', fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. IoU Performance
        ax2.plot(percent_stats['window_size'], percent_stats['combined_mean_iou'], 
                marker='D', linewidth=2.5, markersize=7, label='Combined Mean IoU', color='#F18F01')
        ax2.set_title('IoU Performance', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Window Size (seconds)', fontweight='bold')
        ax2.set_ylabel('Combined Mean IoU (%)', fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance Summary Statistics
        ax3.axis('off')
        
        # Create detailed summary statistics
        summary_data = [
            ['Metric', 'Best Score', 'Best Window Size', 'Mean ± Std'],
            ['', '', '', ''],
            ['Binary F1', f"{eval_percent_df['binary_f1'].max():.3f}", 
             f"{eval_percent_df.loc[eval_percent_df['binary_f1'].idxmax(), 'window_size']:.1f}s",
             f"{eval_percent_df['binary_f1'].mean():.3f} ± {eval_percent_df['binary_f1'].std():.3f}"],
            ['Multiclass F1', f"{eval_percent_df['multiclass_f1'].max():.3f}", 
             f"{eval_percent_df.loc[eval_percent_df['multiclass_f1'].idxmax(), 'window_size']:.1f}s",
             f"{eval_percent_df['multiclass_f1'].mean():.3f} ± {eval_percent_df['multiclass_f1'].std():.3f}"],
            ['Combined IoU', f"{eval_percent_df['combined_mean_iou'].max():.2f}%", 
             f"{eval_percent_df.loc[eval_percent_df['combined_mean_iou'].idxmax(), 'window_size']:.1f}s",
             f"{eval_percent_df['combined_mean_iou'].mean():.2f} ± {eval_percent_df['combined_mean_iou'].std():.2f}%"]
        ]
        
        table = ax3.table(cellText=summary_data,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.2, 0.25, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)
        
        # Style the header row
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(1, i)].set_facecolor('#D9E2F3')  # Empty row for spacing
        
        ax3.set_title('Performance Summary Statistics', fontweight='bold', pad=20, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'frame_level_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'frame_level_performance_analysis.pdf', bbox_inches='tight')
        plt.show()
        
        print(f"✅ Performance analysis graphs saved to {self.output_dir}")
    
    def create_detailed_window_analysis(self):
        """Create detailed analysis for optimal window sizes - Percentage Stride Only"""
        print("📊 Creating detailed window size analysis...")
        
        # Use only eval_percent data (percentage stride)
        eval_percent_df = self.summary_df[self.summary_df['eval_type'] == 'eval_percent'].copy()
        
        if eval_percent_df.empty:
            print("⚠️ No percentage stride data found!")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Frame-Level Evaluation: Detailed Window Size Analysis (Percentage Stride)', fontsize=16, fontweight='bold')
        
        window_sizes = sorted(eval_percent_df['window_size'].unique())
        
        # 1. Binary F1 distribution by window size
        binary_f1_by_window = [eval_percent_df[eval_percent_df['window_size'] == ws]['binary_f1'].values 
                              for ws in window_sizes]
        
        bp1 = ax1.boxplot(binary_f1_by_window, positions=window_sizes, widths=0.05, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('#2E86AB')
            patch.set_alpha(0.7)
        ax1.set_title('Binary F1 Score Distribution by Window Size', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Window Size (seconds)', fontweight='bold')
        ax1.set_ylabel('Binary F1 Score', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Multiclass F1 distribution by window size
        multiclass_f1_by_window = [eval_percent_df[eval_percent_df['window_size'] == ws]['multiclass_f1'].values 
                                  for ws in window_sizes]
        
        bp2 = ax2.boxplot(multiclass_f1_by_window, positions=window_sizes, widths=0.05, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('#A23B72')
            patch.set_alpha(0.7)
        ax2.set_title('Multiclass F1 Score Distribution by Window Size', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Window Size (seconds)', fontweight='bold')
        ax2.set_ylabel('Multiclass F1 Score', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. IoU distribution by window size
        iou_by_window = [eval_percent_df[eval_percent_df['window_size'] == ws]['combined_mean_iou'].values 
                        for ws in window_sizes]
        
        bp3 = ax3.boxplot(iou_by_window, positions=window_sizes, widths=0.05, patch_artist=True)
        for patch in bp3['boxes']:
            patch.set_facecolor('#F18F01')
            patch.set_alpha(0.7)
        ax3.set_title('Combined Mean IoU Distribution by Window Size', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Window Size (seconds)', fontweight='bold')
        ax3.set_ylabel('Combined Mean IoU (%)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Best configurations and statistical summary
        ax4.axis('off')
        
        # Find best configurations
        best_binary_f1 = eval_percent_df.loc[eval_percent_df['binary_f1'].idxmax()]
        best_multiclass_f1 = eval_percent_df.loc[eval_percent_df['multiclass_f1'].idxmax()]
        best_iou = eval_percent_df.loc[eval_percent_df['combined_mean_iou'].idxmax()]
        
        best_configs = [
            ['Metric', 'Best Score', 'Window Size', 'Stride', 'Config ID'],
            ['', '', '', '', ''],
            ['Binary F1', f"{best_binary_f1['binary_f1']:.3f}", f"{best_binary_f1['window_size']}s", 
             f"{best_binary_f1['stride_label']}", f"W{best_binary_f1['window_size']}_S{best_binary_f1['stride_value']}"],
            ['Multiclass F1', f"{best_multiclass_f1['multiclass_f1']:.3f}", f"{best_multiclass_f1['window_size']}s", 
             f"{best_multiclass_f1['stride_label']}", f"W{best_multiclass_f1['window_size']}_S{best_multiclass_f1['stride_value']}"],
            ['Combined IoU', f"{best_iou['combined_mean_iou']:.2f}%", f"{best_iou['window_size']}s", 
             f"{best_iou['stride_label']}", f"W{best_iou['window_size']}_S{best_iou['stride_value']}"],
            ['', '', '', '', ''],
            ['Overall Stats', 'Mean ± Std', 'Range', 'Count', 'Best Overall'],
            ['Binary F1', f"{eval_percent_df['binary_f1'].mean():.3f} ± {eval_percent_df['binary_f1'].std():.3f}", 
             f"{eval_percent_df['binary_f1'].min():.3f}-{eval_percent_df['binary_f1'].max():.3f}", 
             f"{len(eval_percent_df)}", f"{best_binary_f1['window_size']}s window"],
            ['Multiclass F1', f"{eval_percent_df['multiclass_f1'].mean():.3f} ± {eval_percent_df['multiclass_f1'].std():.3f}", 
             f"{eval_percent_df['multiclass_f1'].min():.3f}-{eval_percent_df['multiclass_f1'].max():.3f}", 
             f"{len(eval_percent_df)}", f"{best_multiclass_f1['window_size']}s window"]
        ]
        
        table = ax4.table(cellText=best_configs,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.22, 0.22, 0.18, 0.18, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)
        
        # Style the header rows
        for i in range(len(best_configs[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(6, i)].set_facecolor('#70AD47')
            table[(6, i)].set_text_props(weight='bold', color='white')
            # Empty spacing rows
            table[(1, i)].set_facecolor('#F2F2F2')
            table[(5, i)].set_facecolor('#F2F2F2')
        
        ax4.set_title('Optimal Configurations and Statistical Summary', fontweight='bold', pad=20, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'frame_level_detailed_window_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'frame_level_detailed_window_analysis.pdf', bbox_inches='tight')
        plt.show()
        
        print(f"✅ Detailed window analysis saved to {self.output_dir}")
    
    def generate_all_graphs(self):
        """Generate all frame-level evaluation graphs"""
        print("🚀 Generating Frame-Level Evaluation Line Graphs")
        print("=" * 60)
        
        # Print data summary
        print(f"📊 Data Summary:")
        print(f"   Total evaluations: {len(self.summary_df)}")
        print(f"   Window sizes: {sorted(self.summary_df['window_size'].unique())}")
        print(f"   Evaluation types: {list(self.summary_df['eval_type'].unique())}")
        print(f"   F1 score range: {self.summary_df['binary_f1'].min():.3f} - {self.summary_df['binary_f1'].max():.3f}")
        print()
        
        # Generate all graphs
        self.create_f1_by_window_size_graphs()
        self.create_iou_by_window_size_graphs()
        self.create_comparison_graphs()
        self.create_detailed_window_analysis()
        
        print("\n✅ All frame-level evaluation graphs generated successfully!")
        print(f"📁 Output directory: {self.output_dir}")

def main():
    """Main function"""
    results_dir = "frame_level_evaluation_results"
    
    # Check if results directory exists
    if not Path(results_dir).exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("Please ensure the frame_level_evaluation_results directory exists with evaluation_summary.csv")
        return
    
    try:
        # Create graph generator
        generator = FrameLevelGraphGenerator(results_dir)
        
        # Generate all graphs
        generator.generate_all_graphs()
        
    except Exception as e:
        print(f"❌ Error generating graphs: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
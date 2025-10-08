#!/usr/bin/env python3
"""
📊 VAD REFINED EVALUATION LINE GRAPHS GENERATOR
Generate F1 score and window size line graphs for VAD refined evaluation results

Features:
- Binary and Multiclass F1 score line graphs by window size
- IoU performance analysis by window size
- Comparison between eval_percent and eval_by_0.05 methods
- Professional publication-ready visualizations
- Comparison with original comprehensive evaluation results
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

class VADRefinedGraphGenerator:
    """Generate line graphs for VAD refined evaluation results"""
    
    def __init__(self, vad_results_dir: str, comprehensive_results_dir: str = None):
        """Initialize with VAD results directory and optional comprehensive results for comparison"""
        self.vad_results_dir = Path(vad_results_dir)
        self.comprehensive_results_dir = Path(comprehensive_results_dir) if comprehensive_results_dir else None
        self.output_dir = self.vad_results_dir / "vad_refined_line_graphs"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load VAD evaluation summary
        self.vad_summary_df = self._load_vad_evaluation_summary()
        print(f"✅ Loaded {len(self.vad_summary_df)} VAD refined evaluation results")
        
        # Load comprehensive evaluation summary for comparison if available
        self.comprehensive_summary_df = None
        if self.comprehensive_results_dir:
            try:
                self.comprehensive_summary_df = self._load_comprehensive_evaluation_summary()
                print(f"✅ Loaded {len(self.comprehensive_summary_df)} comprehensive evaluation results for comparison")
            except Exception as e:
                print(f"⚠️ Could not load comprehensive results for comparison: {e}")
    
    def _load_vad_evaluation_summary(self):
        """Load the VAD refinement summary CSV"""
        summary_path = self.vad_results_dir / "vad_refinement_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"VAD refinement summary not found: {summary_path}")
        
        df = pd.read_csv(summary_path)
        
        # Clean and process data
        df = df[df['success'] == True].copy()  # Only successful evaluations
        df['window_size'] = pd.to_numeric(df['window_size'], errors='coerce')
        df['binary_f1'] = pd.to_numeric(df['binary_f1'], errors='coerce')
        df['multiclass_f1'] = pd.to_numeric(df['multiclass_f1'], errors='coerce')
        df['combined_iou'] = pd.to_numeric(df['combined_iou'], errors='coerce')
        
        # Parse stride information for better analysis
        df['stride_value'] = df['stride_info'].str.extract(r'(\d+\.?\d*)').astype(float)
        df['stride_unit'] = df['stride_info'].str.extract(r'(\w+)$')
        
        # Create stride labels for better visualization
        df['stride_label'] = df['stride_info']
        
        # Remove any rows with NaN values
        df = df.dropna(subset=['window_size', 'binary_f1', 'multiclass_f1', 'combined_iou'])
        
        return df
    
    def _load_comprehensive_evaluation_summary(self):
        """Load comprehensive evaluation summary for comparison"""
        summary_path = self.comprehensive_results_dir / "comprehensive_evaluation_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Comprehensive evaluation summary not found: {summary_path}")
        
        df = pd.read_csv(summary_path)
        
        # Clean and process data similar to VAD results
        df = df[df['success'] == True].copy()
        df['window_size'] = pd.to_numeric(df['window_size'], errors='coerce')
        df['binary_f1'] = pd.to_numeric(df['binary_f1'], errors='coerce')
        df['multiclass_f1'] = pd.to_numeric(df['multiclass_f1'], errors='coerce')
        df['combined_mean_iou'] = pd.to_numeric(df['combined_mean_iou'], errors='coerce')
        
        # Rename IoU column to match VAD results
        if 'combined_mean_iou' in df.columns:
            df['combined_iou'] = df['combined_mean_iou']
        
        df = df.dropna(subset=['window_size', 'binary_f1', 'multiclass_f1', 'combined_iou'])
        
        return df
    
    def create_vad_f1_by_window_size_graphs(self):
        """Create F1 score line graphs by window size - Combined Binary and Multiclass"""
        print("📊 Creating VAD refined F1 score by window size graphs...")
        
        # Create separate graphs for each evaluation type
        eval_types = self.vad_summary_df['eval_type'].unique()
        
        for eval_type in eval_types:
            eval_df = self.vad_summary_df[self.vad_summary_df['eval_type'] == eval_type].copy()
            
            if eval_df.empty:
                print(f"⚠️ No data found for {eval_type}!")
                continue
            
            # Determine stride type for title
            stride_type = "Fixed Time Stride" if eval_type == 'eval_by_0.05' else "Percentage Stride"
            
            # Create single figure for combined F1 scores
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            fig.suptitle(f'VAD Refined Evaluation: F1 Scores by Window Size ({stride_type})', 
                        fontsize=16, fontweight='bold')
            
            # Plot both binary and multiclass F1 on the same graph
            self._plot_vad_combined_f1_by_window(eval_df, ax, eval_type)
            
            # Adjust layout and save
            plt.tight_layout()
            output_name = f'vad_refined_f1_by_window_size_{eval_type}'
            plt.savefig(self.output_dir / f'{output_name}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / f'{output_name}.pdf', bbox_inches='tight')
            plt.show()
            
            print(f"✅ VAD refined F1 by window size graph saved for {eval_type}")
    
    def _plot_vad_combined_f1_by_window(self, df, ax, eval_type):
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
                   label='VAD Binary F1', color='#2E86AB', alpha=0.8)
        
        # Plot multiclass F1
        ax.errorbar(multiclass_stats['window_size'], multiclass_stats['mean'], 
                   yerr=multiclass_stats['sem'], 
                   marker='s', linewidth=2.5, markersize=7, capsize=5, 
                   label='VAD Multiclass F1', color='#A23B72', alpha=0.8)
        
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
        ax.set_title('VAD Refined: Binary vs Multiclass F1 Performance by Window Size', 
                    fontsize=15, fontweight='bold', pad=20)
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
        
        ax.annotate(f'Best VAD Binary F1: {best_binary_score:.3f}\n@ {best_binary_window}s window',
                   xy=(best_binary_window, best_binary_score), 
                   xytext=(best_binary_window + 0.2, best_binary_score + 0.05),
                   arrowprops=dict(arrowstyle='->', color='#2E86AB', alpha=0.7),
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.annotate(f'Best VAD Multiclass F1: {best_multiclass_score:.3f}\n@ {best_multiclass_window}s window',
                   xy=(best_multiclass_window, best_multiclass_score), 
                   xytext=(best_multiclass_window + 0.5, best_multiclass_score - 0.08),
                   arrowprops=dict(arrowstyle='->', color='#A23B72', alpha=0.7, connectionstyle='arc3,rad=-0.2'),
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    def create_vad_iou_by_window_size_graphs(self):
        """Create IoU line graphs by window size"""
        print("📊 Creating VAD refined IoU by window size graphs...")
        
        # Create separate graphs for each evaluation type
        eval_types = self.vad_summary_df['eval_type'].unique()
        
        for eval_type in eval_types:
            eval_df = self.vad_summary_df[self.vad_summary_df['eval_type'] == eval_type].copy()
            
            if eval_df.empty:
                print(f"⚠️ No data found for {eval_type}!")
                continue
            
            # Determine stride type for title
            stride_type = "Fixed Time Stride" if eval_type == 'eval_by_0.05' else "Percentage Stride"
            
            # Create single figure for IoU
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            fig.suptitle(f'VAD Refined Evaluation: IoU Performance by Window Size ({stride_type})', 
                        fontsize=16, fontweight='bold')
            
            # Plot IoU
            self._plot_vad_iou_by_window(eval_df, f'VAD Refined IoU Performance ({stride_type})', ax)
            
            plt.tight_layout()
            output_name = f'vad_refined_iou_by_window_size_{eval_type}'
            plt.savefig(self.output_dir / f'{output_name}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / f'{output_name}.pdf', bbox_inches='tight')
            plt.show()
            
            print(f"✅ VAD refined IoU by window size graph saved for {eval_type}")
    
    def _plot_vad_iou_by_window(self, df, title, ax):
        """Plot IoU by window size with error bars"""
        # Group by window size and calculate statistics
        window_stats = df.groupby('window_size')['combined_iou'].agg(['mean', 'std', 'count']).reset_index()
        window_stats['sem'] = window_stats['std'] / np.sqrt(window_stats['count'])
        
        # Create line plot with error bars
        ax.errorbar(window_stats['window_size'], window_stats['mean'], 
                   yerr=window_stats['sem'], 
                   marker='o', linewidth=2.5, markersize=7, capsize=5, color='#F18F01', alpha=0.8)
        
        # Scatter plot for individual points
        ax.scatter(df['window_size'], df['combined_iou'], alpha=0.3, s=20, color='gray')
        
        ax.set_xlabel('Window Size (seconds)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Combined IoU', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(window_stats) > 1:
            z = np.polyfit(window_stats['window_size'], window_stats['mean'], 1)
            p = np.poly1d(z)
            ax.plot(window_stats['window_size'], p(window_stats['window_size']), 
                   "--", alpha=0.7, linewidth=1.5, color='red')
        
        # Add best performance annotation
        best_idx = window_stats['mean'].idxmax()
        best_window = window_stats.iloc[best_idx]['window_size']
        best_score = window_stats.iloc[best_idx]['mean']
        
        ax.annotate(f'Best VAD IoU: {best_score:.3f}\n@ {best_window}s window',
                   xy=(best_window, best_score), 
                   xytext=(best_window + 0.3, best_score + 0.02),
                   arrowprops=dict(arrowstyle='->', color='#F18F01', alpha=0.7),
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    def create_vad_vs_comprehensive_comparison(self):
        """Create comparison graphs between VAD refined and comprehensive evaluation results"""
        if self.comprehensive_summary_df is None:
            print("⚠️ No comprehensive evaluation data available for comparison")
            return
        
        print("📊 Creating VAD vs Comprehensive evaluation comparison graphs...")
        
        # Find common configurations for fair comparison
        eval_types = ['eval_by_0.05', 'eval_percent']
        
        for eval_type in eval_types:
            if eval_type not in self.vad_summary_df['eval_type'].values:
                continue
                
            vad_df = self.vad_summary_df[self.vad_summary_df['eval_type'] == eval_type].copy()
            
            if eval_type in self.comprehensive_summary_df['eval_type'].values:
                comp_df = self.comprehensive_summary_df[self.comprehensive_summary_df['eval_type'] == eval_type].copy()
            else:
                print(f"⚠️ No comprehensive data for {eval_type}")
                continue
            
            # Create comparison figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            stride_type = "Fixed Time Stride" if eval_type == 'eval_by_0.05' else "Percentage Stride"
            fig.suptitle(f'VAD Refined vs Comprehensive Evaluation Comparison ({stride_type})', 
                        fontsize=16, fontweight='bold')
            
            # 1. Binary F1 Comparison
            self._plot_comparison_metric(vad_df, comp_df, 'binary_f1', 'Binary F1 Score', ax1)
            
            # 2. Multiclass F1 Comparison  
            self._plot_comparison_metric(vad_df, comp_df, 'multiclass_f1', 'Multiclass F1 Score', ax2)
            
            # 3. IoU Comparison
            self._plot_comparison_metric(vad_df, comp_df, 'combined_iou', 'Combined IoU', ax3)
            
            # 4. Performance Summary Table
            self._create_comparison_summary_table(vad_df, comp_df, ax4, eval_type)
            
            plt.tight_layout()
            output_name = f'vad_vs_comprehensive_comparison_{eval_type}'
            plt.savefig(self.output_dir / f'{output_name}.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / f'{output_name}.pdf', bbox_inches='tight')
            plt.show()
            
            print(f"✅ VAD vs Comprehensive comparison saved for {eval_type}")

    def _plot_comparison_metric(self, vad_df, comp_df, metric_col, metric_name, ax):
        """Plot comparison between VAD and comprehensive results for a specific metric"""
        # Calculate mean by window size for both datasets
        vad_stats = vad_df.groupby('window_size')[metric_col].mean().reset_index()
        comp_stats = comp_df.groupby('window_size')[metric_col].mean().reset_index()
        
        # Plot VAD results
        ax.plot(vad_stats['window_size'], vad_stats[metric_col], 
               marker='o', linewidth=2.5, markersize=7, label='VAD Refined', 
               color='#2E86AB', alpha=0.8)
        
        # Plot comprehensive results
        ax.plot(comp_stats['window_size'], comp_stats[metric_col], 
               marker='s', linewidth=2.5, markersize=7, label='Comprehensive (Original)', 
               color='#E63946', alpha=0.8)
        
        ax.set_xlabel('Window Size (seconds)', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if 'f1' in metric_col.lower():
            ax.set_ylim(0, 1)

    def _create_comparison_summary_table(self, vad_df, comp_df, ax, eval_type):
        """Create summary table comparing VAD and comprehensive results"""
        ax.axis('off')
        
        # Calculate summary statistics
        vad_binary_mean = vad_df['binary_f1'].mean()
        vad_binary_best = vad_df['binary_f1'].max()
        comp_binary_mean = comp_df['binary_f1'].mean()
        comp_binary_best = comp_df['binary_f1'].max()
        
        vad_multi_mean = vad_df['multiclass_f1'].mean()
        vad_multi_best = vad_df['multiclass_f1'].max()
        comp_multi_mean = comp_df['multiclass_f1'].mean()
        comp_multi_best = comp_df['multiclass_f1'].max()
        
        vad_iou_mean = vad_df['combined_iou'].mean()
        vad_iou_best = vad_df['combined_iou'].max()
        comp_iou_mean = comp_df['combined_iou'].mean()
        comp_iou_best = comp_df['combined_iou'].max()
        
        # Calculate improvements
        binary_mean_improvement = ((vad_binary_mean - comp_binary_mean) / comp_binary_mean * 100)
        binary_best_improvement = ((vad_binary_best - comp_binary_best) / comp_binary_best * 100)
        multi_mean_improvement = ((vad_multi_mean - comp_multi_mean) / comp_multi_mean * 100)
        multi_best_improvement = ((vad_multi_best - comp_multi_best) / comp_multi_best * 100)
        iou_mean_improvement = ((vad_iou_mean - comp_iou_mean) / comp_iou_mean * 100)
        iou_best_improvement = ((vad_iou_best - comp_iou_best) / comp_iou_best * 100)
        
        summary_data = [
            ['Metric', 'VAD Mean', 'Comp Mean', 'Mean Δ%', 'VAD Best', 'Comp Best', 'Best Δ%'],
            ['', '', '', '', '', '', ''],
            ['Binary F1', f'{vad_binary_mean:.3f}', f'{comp_binary_mean:.3f}', 
             f'{binary_mean_improvement:+.1f}%', f'{vad_binary_best:.3f}', f'{comp_binary_best:.3f}', 
             f'{binary_best_improvement:+.1f}%'],
            ['Multiclass F1', f'{vad_multi_mean:.3f}', f'{comp_multi_mean:.3f}', 
             f'{multi_mean_improvement:+.1f}%', f'{vad_multi_best:.3f}', f'{comp_multi_best:.3f}', 
             f'{multi_best_improvement:+.1f}%'],
            ['Combined IoU', f'{vad_iou_mean:.3f}', f'{comp_iou_mean:.3f}', 
             f'{iou_mean_improvement:+.1f}%', f'{vad_iou_best:.3f}', f'{comp_iou_best:.3f}', 
             f'{iou_best_improvement:+.1f}%']
        ]
        
        table = ax.table(cellText=summary_data,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.18, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style the header row
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(1, i)].set_facecolor('#D9E2F3')  # Empty row for spacing
        
        # Color improvement cells
        for row in range(2, len(summary_data)):
            # Mean improvement column (index 3)
            cell_value = summary_data[row][3]
            if '+' in cell_value:
                table[(row, 3)].set_facecolor('#C6EFCE')  # Light green for positive
            else:
                table[(row, 3)].set_facecolor('#FFC7CE')  # Light red for negative
            
            # Best improvement column (index 6)
            cell_value = summary_data[row][6]
            if '+' in cell_value:
                table[(row, 6)].set_facecolor('#C6EFCE')  # Light green for positive
            else:
                table[(row, 6)].set_facecolor('#FFC7CE')  # Light red for negative
        
        ax.set_title(f'Performance Comparison Summary ({eval_type})', fontweight='bold', pad=20)

    def create_vad_detailed_analysis(self):
        """Create detailed analysis of VAD refined results"""
        print("📊 Creating detailed VAD refined analysis...")
        
        # Create comprehensive analysis figure
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('VAD Refined Evaluation: Comprehensive Performance Analysis', 
                    fontsize=18, fontweight='bold')
        
        # Create subplots with custom layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall F1 performance by evaluation type
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_overall_f1_performance(ax1)
        
        # 2. Overall IoU performance by evaluation type
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_overall_iou_performance(ax2)
        
        # 3. Window size distribution analysis for eval_by_0.05
        ax3 = fig.add_subplot(gs[1, 0])
        eval_by_df = self.vad_summary_df[self.vad_summary_df['eval_type'] == 'eval_by_0.05']
        if not eval_by_df.empty:
            self._plot_window_distribution(eval_by_df, ax3, 'eval_by_0.05 (Fixed Time)')
        
        # 4. Window size distribution analysis for eval_percent
        ax4 = fig.add_subplot(gs[1, 1])
        eval_percent_df = self.vad_summary_df[self.vad_summary_df['eval_type'] == 'eval_percent']
        if not eval_percent_df.empty:
            self._plot_window_distribution(eval_percent_df, ax4, 'eval_percent (Percentage)')
        
        # 5. Processing time analysis
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_processing_time_analysis(ax5)
        
        # 6. Best configurations summary
        ax6 = fig.add_subplot(gs[2, :])
        self._create_best_configurations_table(ax6)
        
        plt.savefig(self.output_dir / 'vad_refined_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'vad_refined_comprehensive_analysis.pdf', bbox_inches='tight')
        plt.show()
        
        print("✅ VAD refined comprehensive analysis saved")

    def _plot_overall_f1_performance(self, ax):
        """Plot overall F1 performance comparison between evaluation types"""
        eval_types = self.vad_summary_df['eval_type'].unique()
        
        binary_means = []
        multiclass_means = []
        type_labels = []
        
        for eval_type in eval_types:
            eval_df = self.vad_summary_df[self.vad_summary_df['eval_type'] == eval_type]
            binary_means.append(eval_df['binary_f1'].mean())
            multiclass_means.append(eval_df['multiclass_f1'].mean())
            type_labels.append(eval_type.replace('_', ' ').title())
        
        x = np.arange(len(type_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, binary_means, width, label='Binary F1', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, multiclass_means, width, label='Multiclass F1', color='#A23B72', alpha=0.8)
        
        ax.set_xlabel('Evaluation Type', fontweight='bold')
        ax.set_ylabel('Mean F1 Score', fontweight='bold')
        ax.set_title('Overall F1 Performance by Evaluation Type', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(type_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)

    def _plot_overall_iou_performance(self, ax):
        """Plot overall IoU performance by evaluation type"""
        eval_types = self.vad_summary_df['eval_type'].unique()
        
        iou_means = []
        type_labels = []
        
        for eval_type in eval_types:
            eval_df = self.vad_summary_df[self.vad_summary_df['eval_type'] == eval_type]
            iou_means.append(eval_df['combined_iou'].mean())
            type_labels.append(eval_type.replace('_', ' ').title())
        
        bars = ax.bar(type_labels, iou_means, color='#F18F01', alpha=0.8)
        
        ax.set_xlabel('Evaluation Type', fontweight='bold')
        ax.set_ylabel('Mean Combined IoU', fontweight='bold')
        ax.set_title('Overall IoU Performance', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    def _plot_window_distribution(self, df, ax, title):
        """Plot distribution of performance by window size"""
        window_sizes = sorted(df['window_size'].unique())
        binary_f1_by_window = [df[df['window_size'] == ws]['binary_f1'].values for ws in window_sizes]
        
        bp = ax.boxplot(binary_f1_by_window, positions=window_sizes, widths=0.05, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#2E86AB')
            patch.set_alpha(0.7)
        
        ax.set_title(f'Binary F1 Distribution\n{title}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Window Size (s)', fontweight='bold')
        ax.set_ylabel('Binary F1', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def _plot_processing_time_analysis(self, ax):
        """Plot processing time analysis"""
        # Group by window size and calculate mean processing time
        time_stats = self.vad_summary_df.groupby('window_size')['duration_seconds'].mean().reset_index()
        
        ax.plot(time_stats['window_size'], time_stats['duration_seconds'], 
               marker='o', linewidth=2, markersize=6, color='#FF6B6B')
        
        ax.set_xlabel('Window Size (s)', fontweight='bold')
        ax.set_ylabel('Mean Processing Time (s)', fontweight='bold')
        ax.set_title('Processing Time by Window Size', fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _create_best_configurations_table(self, ax):
        """Create table showing best configurations for each metric"""
        ax.axis('off')
        
        # Find best configurations for each metric
        best_binary = self.vad_summary_df.loc[self.vad_summary_df['binary_f1'].idxmax()]
        best_multiclass = self.vad_summary_df.loc[self.vad_summary_df['multiclass_f1'].idxmax()]
        best_iou = self.vad_summary_df.loc[self.vad_summary_df['combined_iou'].idxmax()]
        
        # Find best overall configuration (weighted average)
        self.vad_summary_df['weighted_score'] = (
            0.4 * self.vad_summary_df['binary_f1'] + 
            0.3 * self.vad_summary_df['multiclass_f1'] + 
            0.3 * self.vad_summary_df['combined_iou']
        )
        best_overall = self.vad_summary_df.loc[self.vad_summary_df['weighted_score'].idxmax()]
        
        table_data = [
            ['Metric', 'Best Score', 'Eval Type', 'Window Size', 'Stride', 'Duration (s)', 'Configuration'],
            ['', '', '', '', '', '', ''],
            ['Binary F1', f"{best_binary['binary_f1']:.3f}", best_binary['eval_type'], 
             f"{best_binary['window_size']}s", best_binary['stride_info'], 
             f"{best_binary['duration_seconds']:.1f}", f"{best_binary['window_name']}_{best_binary['stride_name']}"],
            ['Multiclass F1', f"{best_multiclass['multiclass_f1']:.3f}", best_multiclass['eval_type'], 
             f"{best_multiclass['window_size']}s", best_multiclass['stride_info'], 
             f"{best_multiclass['duration_seconds']:.1f}", f"{best_multiclass['window_name']}_{best_multiclass['stride_name']}"],
            ['Combined IoU', f"{best_iou['combined_iou']:.3f}", best_iou['eval_type'], 
             f"{best_iou['window_size']}s", best_iou['stride_info'], 
             f"{best_iou['duration_seconds']:.1f}", f"{best_iou['window_name']}_{best_iou['stride_name']}"],
            ['Weighted Overall', f"{best_overall['weighted_score']:.3f}", best_overall['eval_type'], 
             f"{best_overall['window_size']}s", best_overall['stride_info'], 
             f"{best_overall['duration_seconds']:.1f}", f"{best_overall['window_name']}_{best_overall['stride_name']}"]
        ]
        
        table = ax.table(cellText=table_data,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15, 0.12, 0.15, 0.12, 0.12, 0.12, 0.22])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # Style the header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(1, i)].set_facecolor('#D9E2F3')  # Empty row for spacing
        
        # Highlight best overall row
        for i in range(len(table_data[0])):
            table[(5, i)].set_facecolor('#70AD47')
            table[(5, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Best VAD Refined Configurations for Each Metric', fontweight='bold', pad=20, fontsize=14)

    def generate_all_vad_graphs(self):
        """Generate all VAD refined evaluation graphs"""
        print("🚀 Generating VAD Refined Evaluation Line Graphs")
        print("=" * 60)
        
        # Print data summary
        print(f"📊 VAD Data Summary:")
        print(f"   Total evaluations: {len(self.vad_summary_df)}")
        print(f"   Successful evaluations: {len(self.vad_summary_df[self.vad_summary_df['success']])}")
        print(f"   Window sizes: {sorted(self.vad_summary_df['window_size'].unique())}")
        print(f"   Evaluation types: {list(self.vad_summary_df['eval_type'].unique())}")
        print(f"   Binary F1 range: {self.vad_summary_df['binary_f1'].min():.3f} - {self.vad_summary_df['binary_f1'].max():.3f}")
        print(f"   Multiclass F1 range: {self.vad_summary_df['multiclass_f1'].min():.3f} - {self.vad_summary_df['multiclass_f1'].max():.3f}")
        print(f"   Combined IoU range: {self.vad_summary_df['combined_iou'].min():.3f} - {self.vad_summary_df['combined_iou'].max():.3f}")
        print()
        
        # Generate all graphs
        self.create_vad_f1_by_window_size_graphs()
        self.create_vad_iou_by_window_size_graphs()
        self.create_vad_detailed_analysis()
        
        # Generate comparison if comprehensive data is available
        if self.comprehensive_summary_df is not None:
            self.create_vad_vs_comprehensive_comparison()
        
        print("\n✅ All VAD refined evaluation graphs generated successfully!")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Print best configurations summary
        print(f"\n🏆 BEST VAD REFINED CONFIGURATIONS:")
        best_binary = self.vad_summary_df.loc[self.vad_summary_df['binary_f1'].idxmax()]
        best_multiclass = self.vad_summary_df.loc[self.vad_summary_df['multiclass_f1'].idxmax()]
        best_iou = self.vad_summary_df.loc[self.vad_summary_df['combined_iou'].idxmax()]
        
        print(f"   Best Binary F1: {best_binary['binary_f1']:.3f} ({best_binary['eval_type']}, {best_binary['window_name']}, {best_binary['stride_name']})")
        print(f"   Best Multiclass F1: {best_multiclass['multiclass_f1']:.3f} ({best_multiclass['eval_type']}, {best_multiclass['window_name']}, {best_multiclass['stride_name']})")
        print(f"   Best Combined IoU: {best_iou['combined_iou']:.3f} ({best_iou['eval_type']}, {best_iou['window_name']}, {best_iou['stride_name']})")

def main():
    """Main function"""
    vad_results_dir = "vad_refined_results"
    comprehensive_results_dir = "fixed_smart_parallel_results"  # Optional for comparison
    
    # Check if VAD results directory exists
    if not Path(vad_results_dir).exists():
        print(f"❌ VAD results directory not found: {vad_results_dir}")
        print("Please ensure the vad_refined_results directory exists with vad_refinement_summary.csv")
        return
    
    try:
        # Create graph generator
        generator = VADRefinedGraphGenerator(vad_results_dir, comprehensive_results_dir)
        
        # Generate all graphs
        generator.generate_all_vad_graphs()
        
    except Exception as e:
        print(f"❌ Error generating VAD refined graphs: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
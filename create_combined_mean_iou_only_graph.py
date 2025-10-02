#!/usr/bin/env python3
"""
Combined Mean IoU Only Graph
Create a line graph showing only Combined Mean IoU across different window sizes
Using CSV files for accurate IoU data extraction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def extract_iou_data():
    """Extract Combined Mean IoU data from CSV files in IoU evaluation results"""
    
    base_dir = Path("fixed_smart_parallel_results")
    all_results = []
    
    # Process both eval_by_0.05 and eval_percent directories
    for eval_type in ["eval_by_0.05", "eval_percent"]:
        eval_dir = base_dir / eval_type / eval_type
        
        if not eval_dir.exists():
            print(f"Directory not found: {eval_dir}")
            continue
        
        print(f"Processing {eval_type}...")
        
        # IoU Evaluation method only
        iou_eval_dir = eval_dir / "iou_eval"
        if iou_eval_dir.exists():
            for window_dir in iou_eval_dir.iterdir():
                if window_dir.is_dir() and window_dir.name.startswith("window_"):
                    for stride_dir in window_dir.iterdir():
                        if stride_dir.is_dir() and stride_dir.name.startswith("stride_"):
                            csv_file = stride_dir / "window_stride_iou_summary.csv"
                            if csv_file.exists():
                                iou_data = extract_iou_from_csv(csv_file)
                                if iou_data:
                                    window_match = re.search(r'window_(\d+\.?\d*)s', window_dir.name)
                                    
                                    # Handle both time-based and percentage-based stride formats
                                    stride_match_time = re.search(r'stride_(\d+\.?\d*)s', stride_dir.name)
                                    stride_match_percent = re.search(r'stride_(\d+\.?\d*)%', stride_dir.name)
                                    
                                    if window_match and (stride_match_time or stride_match_percent):
                                        if stride_match_time:
                                            stride_numeric = float(stride_match_time.group(1))
                                            stride_type = 'seconds'
                                        else:
                                            stride_numeric = float(stride_match_percent.group(1))
                                            stride_type = 'percent'
                                        
                                        config_data = {
                                            'eval_type': eval_type,
                                            'method': 'iou_eval',
                                            'window': window_dir.name,
                                            'stride': stride_dir.name,
                                            'window_numeric': float(window_match.group(1)),
                                            'stride_numeric': stride_numeric,
                                            'stride_type': stride_type,
                                        }
                                        config_data.update(iou_data)
                                        all_results.append(config_data)
    
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\n✅ Total configurations extracted: {len(df)}")
        print(f"Evaluation types: {list(df['eval_type'].unique())}")
        print(f"Window sizes: {sorted(df['window_numeric'].unique())}")
        return df
    else:
        return pd.DataFrame()

def extract_iou_from_csv(csv_file):
    """Extract IoU values from CSV file and calculate simple mean"""
    try:
        df = pd.read_csv(csv_file)
        
        # Column names for the 4 profanity classes
        iou_columns = ['mean_iou_เหี้ย', 'mean_iou_มึง', 'mean_iou_กู', 'mean_iou_เย็ด']
        
        # Check if all required columns exist
        if all(col in df.columns for col in iou_columns):
            # Get the first (and should be only) row
            row = df.iloc[0]
            
            # Extract IoU values for each class
            iou_values = [row[col] for col in iou_columns]
            
            # Calculate simple mean (unweighted average)
            simple_mean_iou = sum(iou_values) / len(iou_values)
            
            # Convert to percentage
            return {
                'combined_mean_iou_percent': simple_mean_iou * 100,
                'individual_iou_values': iou_values
            }
        else:
            print(f"Missing IoU columns in {csv_file}")
            return {}
            
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return {}

def create_combined_mean_iou_graph(df, output_dir):
    """Create line graph showing only Combined Mean IoU"""
    
    # Filter for IoU evaluation method
    iou_data = df[df['method'] == 'iou_eval'].copy()
    
    if iou_data.empty or 'combined_mean_iou_percent' not in iou_data.columns:
        print("❌ No Combined Mean IoU data found")
        return
    
    # Create the graph
    plt.figure(figsize=(10, 6))
    
    # Set style
    plt.style.use('seaborn-v0_8')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    eval_types = sorted(iou_data['eval_type'].unique())
    
    for i, eval_type in enumerate(eval_types):
        eval_subset = iou_data[iou_data['eval_type'] == eval_type]
        
        # Group by window size and calculate mean/std
        window_stats = eval_subset.groupby('window_numeric')['combined_mean_iou_percent'].agg(['mean', 'std']).reset_index()
        
        color = colors[i % len(colors)]
        
        # Plot main line
        plt.plot(
            window_stats['window_numeric'], 
            window_stats['mean'], 
            label=eval_type, 
            marker='o', 
            linewidth=2.5, 
            markersize=6,
            alpha=0.9,
            color=color
        )
        
        # Plot filled area for confidence interval
        plt.fill_between(
            window_stats['window_numeric'],
            window_stats['mean'] - window_stats['std'],
            window_stats['mean'] + window_stats['std'],
            alpha=0.2,  # Semi-transparent fill
            color=color
        )
    
    # Customize the plot
    plt.xlabel('Window Size (seconds)', fontsize=14, fontweight='bold')
    plt.ylabel('Mean IoU (%)', fontsize=14, fontweight='bold')
    plt.title('IoU Evaluation - Combined Mean IoU (Simple Average)', fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Customize legend
    plt.legend(fontsize=12, framealpha=0.9, fancybox=True, shadow=True)
    
    # Set axis limits for better visualization
    plt.xlim(0.25, 2.0)
    plt.ylim(0, max(iou_data['combined_mean_iou_percent']) * 1.1)
    
    # Customize ticks
    plt.xticks(np.arange(0.3, 2.1, 0.2), fontsize=11)
    plt.yticks(fontsize=11)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / "combined_mean_iou_only_graph.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"💾 Graph saved: {output_file}")
    
    # Print summary statistics
    print(f"\n📊 COMBINED MEAN IoU SUMMARY (Simple Average - Unweighted):")
    for eval_type in eval_types:
        eval_subset = iou_data[iou_data['eval_type'] == eval_type]
        best_config = eval_subset.loc[eval_subset['combined_mean_iou_percent'].idxmax()]
        worst_config = eval_subset.loc[eval_subset['combined_mean_iou_percent'].idxmin()]
        
        print(f"\n{eval_type.upper()}:")
        print(f"   Best:  {best_config['combined_mean_iou_percent']:.2f}% ({best_config['window']}, {best_config['stride']})")
        print(f"   Worst: {worst_config['combined_mean_iou_percent']:.2f}% ({worst_config['window']}, {worst_config['stride']})")
        print(f"   Mean:  {eval_subset['combined_mean_iou_percent'].mean():.2f}% ± {eval_subset['combined_mean_iou_percent'].std():.2f}%")

def main():
    """Main execution function"""
    print("🎯 Creating Combined Mean IoU Only Graph...")
    
    # Extract data
    df = extract_iou_data()
    if df.empty:
        print("❌ No data extracted. Check directory structure.")
        return
    
    # Create output directory
    output_dir = Path("combined_mean_iou_graph")
    output_dir.mkdir(exist_ok=True)
    
    # Create the graph
    create_combined_mean_iou_graph(df, output_dir)
    
    print(f"\n🎉 Combined Mean IoU graph created successfully!")
    print(f"📁 Output directory: {output_dir}")

if __name__ == "__main__":
    main()
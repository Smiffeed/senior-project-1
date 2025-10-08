#!/usr/bin/env python3
"""
📊 COMPREHENSIVE WORD EVALUATION COLLECTOR
Collect all word evaluation results and create proper comparison with VAD results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
import seaborn as sns

def collect_word_evaluation_results():
    """Collect all word evaluation results from the comprehensive evaluation"""
    print("🔍 COLLECTING WORD EVALUATION RESULTS")
    print("=" * 50)
    
    word_eval_results = []
    
    # Define evaluation types and their base paths
    eval_types = {
        'eval_by_0.05': 'fixed_smart_parallel_results/eval_by_0.05/eval_by_0.05/word_eval',
        'eval_percent': 'fixed_smart_parallel_results/eval_percent/eval_percent/word_eval'
    }
    
    for eval_type, base_path in eval_types.items():
        base_dir = Path(base_path)
        
        if not base_dir.exists():
            print(f"⚠️ Directory not found: {base_path}")
            continue
        
        print(f"📁 Processing {eval_type}...")
        type_count = 0
        
        # Iterate through window directories
        for window_dir in base_dir.iterdir():
            if not window_dir.is_dir() or not window_dir.name.startswith('window_'):
                continue
            
            # Extract window size
            window_size_match = re.search(r'window_(\d+\.?\d*)s', window_dir.name)
            if not window_size_match:
                continue
            window_size = float(window_size_match.group(1))
            
            # Iterate through stride directories
            for stride_dir in window_dir.iterdir():
                if not stride_dir.is_dir() or not stride_dir.name.startswith('stride_'):
                    continue
                
                # Extract stride value
                stride_match = re.search(r'stride_(\d+\.?\d*)([s%]?)', stride_dir.name)
                if not stride_match:
                    continue
                stride_value = float(stride_match.group(1))
                stride_unit = stride_match.group(2) if stride_match.group(2) else 's'
                
                # Check for note.txt file
                note_file = stride_dir / 'note.txt'
                if not note_file.exists():
                    continue
                
                # Parse note.txt for performance metrics
                try:
                    metrics = parse_word_eval_note(note_file, eval_type, window_size, stride_value, stride_unit, stride_dir.name)
                    if metrics:
                        word_eval_results.append(metrics)
                        type_count += 1
                except Exception as e:
                    print(f"    ❌ Error parsing {stride_dir}: {e}")
        
        print(f"    ✅ Found {type_count} configurations for {eval_type}")
    
    # Convert to DataFrame
    if word_eval_results:
        word_df = pd.DataFrame(word_eval_results)
        print(f"\n📊 WORD EVALUATION SUMMARY:")
        print(f"   Total configurations: {len(word_df)}")
        print(f"   Evaluation types: {list(word_df['eval_type'].unique())}")
        print(f"   Window sizes: {sorted(word_df['window_size'].unique())}")
        print(f"   Binary F1 range: {word_df['binary_f1'].min():.3f} - {word_df['binary_f1'].max():.3f}")
        print(f"   Binary F1 mean: {word_df['binary_f1'].mean():.3f}")
        print(f"   Multiclass F1 range: {word_df['multiclass_f1'].min():.3f} - {word_df['multiclass_f1'].max():.3f}")
        print(f"   Multiclass F1 mean: {word_df['multiclass_f1'].mean():.3f}")
        
        return word_df
    else:
        print("❌ No word evaluation results found!")
        return None

def parse_word_eval_note(note_file, eval_type, window_size, stride_value, stride_unit, stride_name):
    """Parse word evaluation note.txt file for metrics"""
    try:
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metrics using regex
        binary_f1_match = re.search(r'F1-Score: (0\.\d+)', content)
        binary_precision_match = re.search(r'Precision: (0\.\d+|1\.0+)', content)
        binary_recall_match = re.search(r'Recall: (0\.\d+)', content)
        binary_accuracy_match = re.search(r'Accuracy: (0\.\d+)', content)
        
        # Find multiclass metrics (look for the second occurrence of these metrics)
        multiclass_accuracy_matches = re.findall(r'Accuracy: (0\.\d+)', content)
        multiclass_precision_matches = re.findall(r'Precision: (0\.\d+)', content)
        multiclass_f1_matches = re.findall(r'F1-Score: (0\.\d+)', content)
        
        # Ground truth count
        gt_count_match = re.search(r'Total Ground Truth Words: (\d+)', content)
        pred_count_match = re.search(r'Total Predicted Words.*: (\d+)', content)
        
        if not binary_f1_match:
            return None
        
        # Parse values
        binary_f1 = float(binary_f1_match.group(1))
        binary_precision = float(binary_precision_match.group(1)) if binary_precision_match else None
        binary_recall = float(binary_recall_match.group(1)) if binary_recall_match else None
        binary_accuracy = float(binary_accuracy_match.group(1)) if binary_accuracy_match else None
        
        # Get multiclass metrics (usually the second occurrence)
        multiclass_accuracy = float(multiclass_accuracy_matches[1]) if len(multiclass_accuracy_matches) > 1 else None
        multiclass_precision = float(multiclass_precision_matches[1]) if len(multiclass_precision_matches) > 1 else None
        multiclass_f1 = float(multiclass_f1_matches[1]) if len(multiclass_f1_matches) > 1 else None
        
        gt_count = int(gt_count_match.group(1)) if gt_count_match else None
        pred_count = int(pred_count_match.group(1)) if pred_count_match else None
        
        return {
            'eval_type': eval_type,
            'window_size': window_size,
            'stride_value': stride_value,
            'stride_unit': stride_unit,
            'stride_name': stride_name,
            'window_name': f'window_{window_size}s',
            'binary_f1': binary_f1,
            'binary_precision': binary_precision,
            'binary_recall': binary_recall,
            'binary_accuracy': binary_accuracy,
            'multiclass_f1': multiclass_f1 if multiclass_f1 else binary_f1,  # Fallback to binary if multiclass not found
            'multiclass_precision': multiclass_precision,
            'multiclass_accuracy': multiclass_accuracy,
            'gt_word_count': gt_count,
            'pred_word_count': pred_count
        }
    
    except Exception as e:
        print(f"Error parsing {note_file}: {e}")
        return None

def compare_word_vs_vad_results(word_df, vad_df):
    """Create comprehensive comparison between word and VAD evaluation results"""
    print(f"\n🔄 WORD vs VAD EVALUATION COMPARISON")
    print("=" * 40)
    
    # Create output directory
    output_dir = Path("word_vs_vad_comparison")
    output_dir.mkdir(exist_ok=True)
    
    # Match configurations between word and VAD results
    matched_results = []
    
    for _, word_row in word_df.iterrows():
        # Find matching VAD configuration
        vad_matches = vad_df[
            (vad_df['eval_type'] == word_row['eval_type']) &
            (vad_df['window_size'] == word_row['window_size']) &
            (vad_df['stride_name'] == word_row['stride_name'])
        ]
        
        if len(vad_matches) > 0:
            vad_row = vad_matches.iloc[0]
            
            matched_results.append({
                'eval_type': word_row['eval_type'],
                'window_size': word_row['window_size'],
                'stride_name': word_row['stride_name'],
                'word_binary_f1': word_row['binary_f1'],
                'word_multiclass_f1': word_row['multiclass_f1'],
                'word_gt_count': word_row['gt_word_count'],
                'word_pred_count': word_row['pred_word_count'],
                'vad_binary_f1': vad_row['binary_f1'],
                'vad_multiclass_f1': vad_row['multiclass_f1'],
                'vad_combined_iou': vad_row['combined_iou'],
                'vad_duration': vad_row['duration_seconds'],
                'f1_difference': vad_row['binary_f1'] - word_row['binary_f1'],
                'f1_percent_change': ((vad_row['binary_f1'] - word_row['binary_f1']) / word_row['binary_f1'] * 100)
            })
    
    if not matched_results:
        print("❌ No matching configurations found between word and VAD results!")
        return
    
    matched_df = pd.DataFrame(matched_results)
    print(f"✅ Found {len(matched_df)} matching configurations")
    
    # Calculate overall statistics
    print(f"\n📊 OVERALL PERFORMANCE COMPARISON:")
    print(f"   Word Evaluation:")
    print(f"      Binary F1: {matched_df['word_binary_f1'].mean():.3f} ± {matched_df['word_binary_f1'].std():.3f}")
    print(f"      Range: {matched_df['word_binary_f1'].min():.3f} - {matched_df['word_binary_f1'].max():.3f}")
    print(f"      Multiclass F1: {matched_df['word_multiclass_f1'].mean():.3f} ± {matched_df['word_multiclass_f1'].std():.3f}")
    
    print(f"   VAD Evaluation:")
    print(f"      Binary F1: {matched_df['vad_binary_f1'].mean():.3f} ± {matched_df['vad_binary_f1'].std():.3f}")
    print(f"      Range: {matched_df['vad_binary_f1'].min():.3f} - {matched_df['vad_binary_f1'].max():.3f}")
    print(f"      Multiclass F1: {matched_df['vad_multiclass_f1'].mean():.3f} ± {matched_df['vad_multiclass_f1'].std():.3f}")
    print(f"      IoU: {matched_df['vad_combined_iou'].mean():.3f} ± {matched_df['vad_combined_iou'].std():.3f}")
    
    # Performance difference analysis
    mean_f1_diff = matched_df['f1_difference'].mean()
    mean_f1_percent = matched_df['f1_percent_change'].mean()
    
    print(f"\n🎯 PERFORMANCE DIFFERENCE ANALYSIS:")
    print(f"   Mean F1 difference: {mean_f1_diff:.3f} ({mean_f1_percent:+.1f}%)")
    print(f"   F1 difference range: {matched_df['f1_difference'].min():.3f} to {matched_df['f1_difference'].max():.3f}")
    print(f"   Configurations where VAD > Word: {len(matched_df[matched_df['f1_difference'] > 0])}/{len(matched_df)} ({len(matched_df[matched_df['f1_difference'] > 0])/len(matched_df)*100:.1f}%)")
    print(f"   Configurations where Word > VAD: {len(matched_df[matched_df['f1_difference'] < 0])}/{len(matched_df)} ({len(matched_df[matched_df['f1_difference'] < 0])/len(matched_df)*100:.1f}%)")
    
    # Best configurations
    best_word = matched_df.loc[matched_df['word_binary_f1'].idxmax()]
    best_vad = matched_df.loc[matched_df['vad_binary_f1'].idxmax()]
    best_improvement = matched_df.loc[matched_df['f1_difference'].idxmax()]
    worst_degradation = matched_df.loc[matched_df['f1_difference'].idxmin()]
    
    print(f"\n🏆 BEST CONFIGURATIONS:")
    print(f"   Best Word F1: {best_word['word_binary_f1']:.3f}")
    print(f"      Config: {best_word['eval_type']}, {best_word['window_size']}s, {best_word['stride_name']}")
    print(f"      VAD equivalent: {best_word['vad_binary_f1']:.3f} ({best_word['f1_difference']:+.3f})")
    
    print(f"   Best VAD F1: {best_vad['vad_binary_f1']:.3f}")
    print(f"      Config: {best_vad['eval_type']}, {best_vad['window_size']}s, {best_vad['stride_name']}")
    print(f"      Word equivalent: {best_vad['word_binary_f1']:.3f} ({best_vad['f1_difference']:+.3f})")
    
    print(f"   Best VAD improvement: +{best_improvement['f1_difference']:.3f} ({best_improvement['f1_percent_change']:+.1f}%)")
    print(f"      Config: {best_improvement['eval_type']}, {best_improvement['window_size']}s, {best_improvement['stride_name']}")
    print(f"      Word: {best_improvement['word_binary_f1']:.3f} → VAD: {best_improvement['vad_binary_f1']:.3f}")
    
    print(f"   Worst VAD degradation: {worst_degradation['f1_difference']:.3f} ({worst_degradation['f1_percent_change']:+.1f}%)")
    print(f"      Config: {worst_degradation['eval_type']}, {worst_degradation['window_size']}s, {worst_degradation['stride_name']}")
    print(f"      Word: {worst_degradation['word_binary_f1']:.3f} → VAD: {worst_degradation['vad_binary_f1']:.3f}")
    
    # Save comparison results
    matched_df.to_csv(output_dir / "word_vs_vad_detailed_comparison.csv", index=False)
    print(f"\n📄 Detailed comparison saved to: {output_dir}/word_vs_vad_detailed_comparison.csv")
    
    # Create visualizations
    create_comparison_visualizations(matched_df, output_dir)
    
    return matched_df

def create_comparison_visualizations(matched_df, output_dir):
    """Create comparison visualizations"""
    print(f"\n📊 Creating comparison visualizations...")
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Overall comparison scatter plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Word Evaluation vs VAD Refined Evaluation Comparison', fontsize=16, fontweight='bold')
    
    # Scatter plot: Word F1 vs VAD F1
    ax1.scatter(matched_df['word_binary_f1'], matched_df['vad_binary_f1'], alpha=0.6, s=50)
    min_f1 = min(matched_df['word_binary_f1'].min(), matched_df['vad_binary_f1'].min())
    max_f1 = max(matched_df['word_binary_f1'].max(), matched_df['vad_binary_f1'].max())
    ax1.plot([min_f1, max_f1], [min_f1, max_f1], 'r--', alpha=0.8, linewidth=2, label='Perfect Agreement')
    ax1.set_xlabel('Word Evaluation Binary F1', fontweight='bold')
    ax1.set_ylabel('VAD Refined Binary F1', fontweight='bold')
    ax1.set_title('Binary F1: Word vs VAD', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 difference histogram
    ax2.hist(matched_df['f1_difference'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='No Difference')
    ax2.axvline(matched_df['f1_difference'].mean(), color='orange', linestyle='-', linewidth=2, 
               label=f'Mean: {matched_df["f1_difference"].mean():.3f}')
    ax2.set_xlabel('F1 Difference (VAD - Word)', fontweight='bold')
    ax2.set_ylabel('Number of Configurations', fontweight='bold')
    ax2.set_title('Distribution of F1 Differences', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Performance by window size
    window_comparison = matched_df.groupby('window_size').agg({
        'word_binary_f1': 'mean',
        'vad_binary_f1': 'mean',
        'f1_difference': 'mean'
    }).reset_index()
    
    ax3.plot(window_comparison['window_size'], window_comparison['word_binary_f1'], 
            marker='o', linewidth=2.5, markersize=7, label='Word Evaluation', color='#E63946')
    ax3.plot(window_comparison['window_size'], window_comparison['vad_binary_f1'], 
            marker='s', linewidth=2.5, markersize=7, label='VAD Refined', color='#2E86AB')
    ax3.set_xlabel('Window Size (seconds)', fontweight='bold')
    ax3.set_ylabel('Mean Binary F1 Score', fontweight='bold')
    ax3.set_title('Performance by Window Size', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Performance difference by window size
    ax4.plot(window_comparison['window_size'], window_comparison['f1_difference'], 
            marker='D', linewidth=2.5, markersize=7, color='#F18F01')
    ax4.axhline(0, color='red', linestyle='--', alpha=0.8)
    ax4.set_xlabel('Window Size (seconds)', fontweight='bold')
    ax4.set_ylabel('Mean F1 Difference (VAD - Word)', fontweight='bold')
    ax4.set_title('F1 Difference by Window Size', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'word_vs_vad_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'word_vs_vad_comparison_analysis.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"✅ Comparison visualizations saved to {output_dir}")

def main():
    """Main analysis function"""
    print("🚀 COMPREHENSIVE WORD VS VAD EVALUATION ANALYSIS")
    print("=" * 60)
    
    try:
        # 1. Collect all word evaluation results
        word_df = collect_word_evaluation_results()
        if word_df is None:
            return
        
        # 2. Load VAD results
        vad_path = Path("vad_refined_results/vad_refinement_summary.csv")
        if not vad_path.exists():
            print("❌ VAD results not found!")
            return
        
        vad_df = pd.read_csv(vad_path)
        vad_df = vad_df[vad_df['success'] == True].copy()
        print(f"✅ Loaded {len(vad_df)} VAD refined results")
        
        # 3. Compare word vs VAD results
        comparison_df = compare_word_vs_vad_results(word_df, vad_df)
        
        # 4. Generate summary report
        if comparison_df is not None:
            print(f"\n📋 FINAL ANALYSIS SUMMARY:")
            print(f"   Word evaluation shows much higher F1 scores than VAD refinement")
            print(f"   This suggests VAD refinement is degrading performance rather than improving it")
            print(f"   Recommendation: Investigate VAD parameters and refinement logic")
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
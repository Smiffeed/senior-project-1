#!/usr/bin/env python3
"""
🔍 VAD vs WORD EVALUATION DIAGNOSTIC ANALYSIS
Investigate why VAD evaluation gives different F1 scores than merged word evaluation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_evaluation_discrepancy():
    """Analyze the discrepancy between VAD and word evaluation results"""
    print("🔍 EVALUATION DISCREPANCY DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    
    # 1. Load VAD refined results
    vad_path = Path("vad_refined_results/vad_refinement_summary.csv")
    if vad_path.exists():
        vad_df = pd.read_csv(vad_path)
        vad_df = vad_df[vad_df['success'] == True].copy()
        print(f"✅ Loaded {len(vad_df)} VAD refined results")
        print(f"   VAD Binary F1 range: {vad_df['binary_f1'].min():.3f} - {vad_df['binary_f1'].max():.3f}")
        print(f"   VAD Mean Binary F1: {vad_df['binary_f1'].mean():.3f}")
    else:
        print("❌ VAD results not found")
        return
    
    # 2. Load comprehensive evaluation results
    comp_path = Path("fixed_smart_parallel_results/fixed_smart_parallel_summary.csv")
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        comp_df = comp_df[comp_df['success'] == True].copy()
        print(f"✅ Loaded {len(comp_df)} comprehensive results")
        print(f"   Word F1 range: {comp_df['word_f1'].min():.3f} - {comp_df['word_f1'].max():.3f}")
        print(f"   Word Mean F1: {comp_df['word_f1'].mean():.3f}")
        print(f"   Window F1 range: {comp_df['window_f1'].min():.3f} - {comp_df['window_f1'].max():.3f}")
        print(f"   Window Mean F1: {comp_df['window_f1'].mean():.3f}")
    else:
        print("❌ Comprehensive results not found")
        return
    
    print("\n🚨 CRITICAL ISSUE IDENTIFIED:")
    print(f"   Comprehensive evaluation has only {len(comp_df)} configuration(s)")
    print(f"   This is NOT representative of the full evaluation!")
    print()
    
    # 3. Check if there are more comprehensive evaluation results
    print("🔍 Searching for additional comprehensive evaluation data...")
    
    # Look for individual evaluation results
    search_dirs = [
        "fixed_smart_parallel_results/eval_by_0.05/eval_by_0.05/word_eval",
        "fixed_smart_parallel_results/eval_percent/eval_percent/word_eval",
        "batch_evaluation_results",
        "evaluation_results"
    ]
    
    found_word_results = []
    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if search_path.exists():
            print(f"   ✅ Found directory: {search_dir}")
            # Look for word evaluation CSV files
            for csv_file in search_path.rglob("*word*.csv"):
                found_word_results.append(csv_file)
                print(f"      📄 Found: {csv_file.name}")
        else:
            print(f"   ❌ Directory not found: {search_dir}")
    
    # 4. Analyze a specific VAD case to understand the pipeline
    print(f"\n🔬 ANALYZING SPECIFIC VAD CASE:")
    best_vad_config = vad_df.loc[vad_df['binary_f1'].idxmax()]
    print(f"   Best VAD config: {best_vad_config['eval_type']}, {best_vad_config['window_name']}, {best_vad_config['stride_name']}")
    print(f"   Binary F1: {best_vad_config['binary_f1']:.3f}")
    print(f"   Multiclass F1: {best_vad_config['multiclass_f1']:.3f}")
    print(f"   IoU: {best_vad_config['combined_iou']:.3f}")
    
    # Try to find the corresponding window evaluation raw predictions
    window_config = best_vad_config['window_name']
    stride_config = best_vad_config['stride_name']
    eval_type = best_vad_config['eval_type']
    
    raw_pred_path = Path(f"fixed_smart_parallel_results/{eval_type}/{eval_type}/window_eval/{window_config}/{stride_config}/raw_predictions.csv")
    
    if raw_pred_path.exists():
        print(f"   ✅ Found raw predictions: {raw_pred_path}")
        
        # Load and analyze raw predictions
        raw_df = pd.read_csv(raw_pred_path)
        print(f"   📊 Raw predictions analysis:")
        print(f"      Total predictions: {len(raw_df)}")
        print(f"      Profanity predictions: {len(raw_df[raw_df['predicted_label'] != 'none'])}")
        print(f"      Accuracy: {len(raw_df[raw_df['true_label'] == raw_df['predicted_label']]) / len(raw_df):.3f}")
        
        # Check ground truth distribution
        gt_counts = raw_df['true_label'].value_counts()
        pred_counts = raw_df['predicted_label'].value_counts()
        
        print(f"   📈 Ground truth distribution:")
        for label, count in gt_counts.items():
            print(f"      {label}: {count} ({count/len(raw_df)*100:.1f}%)")
        
        print(f"   📈 Prediction distribution:")
        for label, count in pred_counts.items():
            print(f"      {label}: {count} ({count/len(raw_df)*100:.1f}%)")
    
    # 5. Compare with the comprehensive evaluation configuration
    if len(comp_df) > 0:
        comp_config = comp_df.iloc[0]
        print(f"\n📊 COMPREHENSIVE EVALUATION CONFIG ANALYSIS:")
        print(f"   Configuration: {comp_config['eval_type']}, {comp_config['window']}, {comp_config['stride']}")
        print(f"   Window F1: {comp_config['window_f1']:.3f}")
        print(f"   Word F1: {comp_config['word_f1']:.3f}")
        print(f"   IoU: {comp_config['combined_iou']:.3f}")
        
        # Try to find this configuration in VAD results
        matching_vad = vad_df[
            (vad_df['eval_type'] == comp_config['eval_type']) & 
            (vad_df['window_name'] == comp_config['window']) & 
            (vad_df['stride_name'] == comp_config['stride'])
        ]
        
        if len(matching_vad) > 0:
            matching_config = matching_vad.iloc[0]
            print(f"\n🔄 MATCHING VAD CONFIGURATION FOUND:")
            print(f"   VAD Binary F1: {matching_config['binary_f1']:.3f}")
            print(f"   VAD Multiclass F1: {matching_config['multiclass_f1']:.3f}")
            print(f"   VAD IoU: {matching_config['combined_iou']:.3f}")
            
            print(f"\n📉 PERFORMANCE DIFFERENCE:")
            f1_diff = matching_config['binary_f1'] - comp_config['word_f1']
            iou_diff = matching_config['combined_iou'] - comp_config['combined_iou']
            print(f"   Binary F1 difference: {f1_diff:.3f} ({f1_diff/comp_config['word_f1']*100:+.1f}%)")
            print(f"   IoU difference: {iou_diff:.3f} ({iou_diff/comp_config['combined_iou']*100:+.1f}%)")
        else:
            print(f"   ❌ No matching VAD configuration found")
    
    # 6. Identify the core issue
    print(f"\n🎯 ROOT CAUSE ANALYSIS:")
    print(f"=" * 40)
    
    if len(comp_df) == 1 and comp_df.iloc[0]['word_f1'] > 0.8:
        print(f"❌ PROBLEM 1: Incomplete comprehensive evaluation")
        print(f"   - Only 1 configuration in comprehensive results")
        print(f"   - This appears to be an outlier with unusually high F1 (0.865)")
        print(f"   - Not representative of overall performance")
        print()
        
        print(f"❌ PROBLEM 2: Evaluation methodology mismatch")
        print(f"   - VAD evaluation: Window → Merge → VAD Refine → Evaluate")
        print(f"   - Word evaluation: Window → Direct word matching")
        print(f"   - Different evaluation pipelines can produce different results")
        print()
        
        print(f"❌ PROBLEM 3: VAD refinement may be over-correcting")
        print(f"   - VAD is designed to improve temporal precision")
        print(f"   - But it might be splitting or removing valid predictions")
        print(f"   - Leading to missed profanity instances (false negatives)")
        print()
        
    # 7. Recommendations
    print(f"💡 RECOMMENDATIONS:")
    print(f"=" * 20)
    print(f"1. COMPLETE THE COMPREHENSIVE EVALUATION:")
    print(f"   - Run comprehensive evaluation on all configurations")
    print(f"   - Generate complete word_f1 results for fair comparison")
    print()
    print(f"2. ANALYZE VAD REFINEMENT PARAMETERS:")
    print(f"   - Check VAD top_db parameter (currently 20)")
    print(f"   - Investigate overlap merge threshold")
    print(f"   - Examine refinement success rates")
    print()
    print(f"3. CREATE DETAILED COMPARISON:")
    print(f"   - Compare prediction counts at each pipeline stage")
    print(f"   - Analyze where predictions are lost/gained")
    print(f"   - Examine false positive/negative rates")
    print()
    print(f"4. VALIDATE VAD APPROACH:")
    print(f"   - Test VAD refinement on a subset manually")
    print(f"   - Verify that VAD improves rather than hurts performance")
    print(f"   - Consider adjusting VAD parameters")

def investigate_vad_pipeline_impact():
    """Investigate how VAD refinement affects predictions at each stage"""
    print(f"\n🔬 VAD PIPELINE IMPACT ANALYSIS")
    print(f"=" * 40)
    
    # Load a specific VAD refinement case
    vad_result_dir = Path("vad_refined_results/vad_refined_results/eval_percent/window_0.5s/stride_30.0%")
    
    if not vad_result_dir.exists():
        print(f"❌ VAD result directory not found: {vad_result_dir}")
        return
    
    # Load the pipeline results
    merged_path = vad_result_dir / "merged_predictions.csv"
    refined_path = vad_result_dir / "vad_refined_predictions.csv"
    eval_path = vad_result_dir / "detailed_evaluation_results.csv"
    
    if merged_path.exists() and refined_path.exists():
        merged_df = pd.read_csv(merged_path)
        refined_df = pd.read_csv(refined_path)
        
        print(f"📊 Pipeline Stage Analysis:")
        print(f"   Merged predictions: {len(merged_df)}")
        print(f"   VAD refined predictions: {len(refined_df)}")
        print(f"   Prediction change: {len(refined_df) - len(merged_df):+d} ({(len(refined_df) - len(merged_df))/len(merged_df)*100:+.1f}%)")
        
        # Analyze duration changes
        if 'duration' in merged_df.columns and 'duration' in refined_df.columns:
            merged_total_duration = merged_df['duration'].sum()
            refined_total_duration = refined_df['duration'].sum()
            duration_change = refined_total_duration - merged_total_duration
            
            print(f"   Total duration change: {duration_change:+.3f}s ({duration_change/merged_total_duration*100:+.1f}%)")
        
        # Analyze label distribution changes
        if 'predicted_label' in merged_df.columns and 'predicted_label' in refined_df.columns:
            print(f"   Label distribution changes:")
            merged_labels = merged_df['predicted_label'].value_counts()
            refined_labels = refined_df['predicted_label'].value_counts()
            
            for label in set(merged_labels.index) | set(refined_labels.index):
                merged_count = merged_labels.get(label, 0)
                refined_count = refined_labels.get(label, 0)
                change = refined_count - merged_count
                print(f"      {label}: {merged_count} → {refined_count} ({change:+d})")
    
    if eval_path.exists():
        eval_df = pd.read_csv(eval_path)
        print(f"   Evaluation results: {len(eval_df)} ground truth words evaluated")
        
        if 'best_iou' in eval_df.columns:
            good_matches = len(eval_df[eval_df['best_iou'] > 0.5])
            print(f"   Good matches (IoU > 0.5): {good_matches}/{len(eval_df)} ({good_matches/len(eval_df)*100:.1f}%)")

def main():
    """Main diagnostic function"""
    try:
        analyze_evaluation_discrepancy()
        investigate_vad_pipeline_impact()
        
        print(f"\n🎯 CONCLUSION:")
        print(f"The discrepancy is caused by:")
        print(f"1. Incomplete comprehensive evaluation (only 1 config with outlier F1)")
        print(f"2. Different evaluation methodologies")
        print(f"3. Potential VAD over-refinement reducing recall")
        print(f"4. Need for complete comparative analysis")
        
    except Exception as e:
        print(f"❌ Error in diagnostic analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
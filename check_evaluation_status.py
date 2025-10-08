#!/usr/bin/env python3
"""
Check comprehensive evaluation results status
"""
import pandas as pd
from pathlib import Path

# Check results summary
results_path = Path('fixed_smart_parallel_results/fixed_smart_parallel_summary.csv')
if results_path.exists():
    df = pd.read_csv(results_path)
    print(f"📊 COMPREHENSIVE EVALUATION RESULTS STATUS")
    print(f"=" * 50)
    print(f"Total configurations processed: {len(df)}")
    print(f"Success rate: {df['success'].sum()}/{len(df)} ({df['success'].mean()*100:.1f}%)")
    print(f"Total duration: {df['duration_seconds'].sum()/3600:.1f} hours")
    
    print(f"\n📈 F1 SCORE STATISTICS:")
    print(f"Window F1 range: {df['window_f1'].min():.4f} - {df['window_f1'].max():.4f}")
    print(f"Word F1 range: {df['word_f1'].min():.4f} - {df['word_f1'].max():.4f}")
    print(f"Combined IoU range: {df['combined_iou'].min():.4f} - {df['combined_iou'].max():.4f}")
    
    print(f"\n🎯 EVALUATION TYPES:")
    print(df['eval_type'].value_counts())
    
    print(f"\n📋 SAMPLE CONFIGURATIONS:")
    print(df[['eval_type', 'window', 'stride', 'success', 'window_f1', 'word_f1', 'combined_iou']].head(10))
    
    # Check if we have frame-level equivalent data
    print(f"\n🔍 FRAME-LEVEL COMPARISON POTENTIAL:")
    frame_level_path = Path('frame_level_evaluation_results')
    if frame_level_path.exists():
        print("✅ Frame-level results exist - comparison possible")
    else:
        print("❌ No frame-level results found")
        
    print(f"\n💡 RECOMMENDATION:")
    print("Based on your comprehensive evaluation results, you have two options:")
    print("1. 🚀 REUSE: Use existing results if they match your VAD requirements")
    print("2. 🔄 RE-RUN: Run new evaluation with VAD system using advanced preprocessing")
else:
    print("❌ No comprehensive evaluation results found")
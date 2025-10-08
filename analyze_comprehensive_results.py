#!/usr/bin/env python3
"""
Analysis of Comprehensive Evaluation Results for VAD Compatibility
"""

import pandas as pd
from pathlib import Path

print("🔍 COMPREHENSIVE EVALUATION RESULTS ANALYSIS")
print("=" * 55)

# Check main summary
summary_path = Path('fixed_smart_parallel_results/fixed_smart_parallel_summary.csv')
detailed_path = Path('fixed_smart_parallel_results/detailed_analysis/complete_analysis_data.csv')

if summary_path.exists():
    summary_df = pd.read_csv(summary_path)
    print(f"📊 MAIN SUMMARY:")
    print(f"   Configurations in summary: {len(summary_df)}")
    print(f"   Success rate: {summary_df['success'].sum()}/{len(summary_df)}")

if detailed_path.exists():
    detailed_df = pd.read_csv(detailed_path)
    print(f"📊 DETAILED ANALYSIS:")
    print(f"   Total configurations processed: {len(detailed_df)}")
    print(f"   Success rate: {detailed_df['success'].sum()}/{len(detailed_df)}")
    print(f"   Evaluation types: {detailed_df['eval_type'].unique()}")
    print(f"   Window sizes: {sorted(detailed_df['window_numeric'].unique())}")
    print(f"   F1 Score ranges:")
    print(f"     Window F1: {detailed_df['window_f1'].min():.4f} - {detailed_df['window_f1'].max():.4f}")
    print(f"     Word F1: {detailed_df['word_f1'].min():.4f} - {detailed_df['word_f1'].max():.4f}")
    print(f"     Combined IoU: {detailed_df['combined_iou'].min():.4f} - {detailed_df['combined_iou'].max():.4f}")

print(f"\n❓ QUESTION 1: Does it use advanced preprocessing?")
print(f"✅ YES - The fixed_smart_parallel_processor.py calls:")
print(f"   scripts/comprehensive_evaluation_processor.py")
print(f"   ├── Uses advanced_preprocess_audio() function")
print(f"   ├── Pre-emphasis filtering (librosa.effects.preemphasis)")
print(f"   ├── Epsilon-protected normalization")
print(f"   └── Minimum length padding")
print(f"\n   This is the SAME advanced preprocessing we identified earlier!")

print(f"\n❓ QUESTION 2: Can you use this data for VAD comparison?")
print(f"🎯 ANSWER: DEPENDS on what you want to compare:")

print(f"\n🔄 OPTION A: DIRECT COMPARISON (Limited)")
print(f"   ✅ Pros:")
print(f"     • Already uses advanced preprocessing")
print(f"     • {len(detailed_df) if detailed_path.exists() else 'Multiple'} configurations available")
print(f"     • Multiple evaluation methods (window, word, IoU)")
print(f"   ❌ Cons:")
print(f"     • Uses comprehensive_evaluation_processor.py approach")
print(f"     • Different merging/evaluation logic than VAD system")
print(f"     • Cannot directly validate VAD vs Frame-level hypothesis")

print(f"\n🧪 OPTION B: VAD EVALUATION (Recommended)")
print(f"   ✅ Pros:")
print(f"     • Uses same VAD evaluation logic as frame-level")
print(f"     • Direct comparison with frame-level results")
print(f"     • Can validate preprocessing hypothesis")
print(f"     • Fair apples-to-apples comparison")
print(f"   ⚠️  Cons:")
print(f"     • Requires running new evaluation")
print(f"     • Takes computation time")

print(f"\n💡 RECOMMENDATIONS:")
print(f"=" * 20)

if detailed_path.exists() and len(detailed_df) > 20:
    print(f"Since you have {len(detailed_df)} configurations with advanced preprocessing:")
    print(f"")
    print(f"🎯 HYBRID APPROACH:")
    print(f"1. Use existing comprehensive results as REFERENCE")
    print(f"2. Run VAD evaluation with advanced preprocessing")
    print(f"3. Compare both against frame-level results")
    print(f"4. Validate that advanced preprocessing eliminates differences")
    print(f"")
    print(f"Expected outcome:")
    print(f"• VAD+Advanced ≈ Comprehensive ≈ each other")
    print(f"• Both should differ from Frame-level (simple preprocessing)")
    print(f"• Proves preprocessing is the key factor")
else:
    print(f"Since you have limited comprehensive data:")
    print(f"")
    print(f"🔄 RECOMMENDED ACTION:")
    print(f"Run VAD evaluation with advanced preprocessing on all")
    print(f"frame-level configurations for complete comparison")

print(f"\n🛠️ IMPLEMENTATION CHOICE:")
print(f"Do you want to:")
print(f"A) Use existing comprehensive results for comparison")
print(f"B) Run new VAD evaluation with advanced preprocessing")
print(f"C) Both (most comprehensive analysis)")

answer = input(f"\nYour choice (A/B/C): ").upper().strip()

if answer == "A":
    print(f"\n✅ Using existing comprehensive results")
    print(f"Note: This compares different evaluation methodologies")
elif answer == "B":
    print(f"\n🔄 Running new VAD evaluation recommended")
    print(f"This will provide the fairest comparison")
elif answer == "C":
    print(f"\n🎯 Hybrid approach - BEST choice!")
    print(f"This gives you the most complete analysis")
else:
    print(f"\n💡 I recommend Option C for the most thorough analysis")
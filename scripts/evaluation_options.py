#!/usr/bin/env python3
"""
Practical evaluation approaches for both eval_by_0.05 and eval_percent datasets
"""

print("=== EVALUATION STRATEGIES FOR BOTH DATASETS ===")
print()

print("📊 DATASET OVERVIEW:")
print("• eval_by_0.05: 227 CSV files (absolute stride values)")
print("• eval_percent: 179 CSV files (percentage-based stride values)")
print("• Total: 406 CSV files")
print("• Estimated full evaluation time: ~13.5 hours")
print()

print("🎯 RECOMMENDED APPROACHES:")
print()

print("1. FOCUSED EVALUATION (Recommended)")
print("   Target specific promising configurations based on research:")
print("   • Small windows: 0.3s, 0.4s, 0.5s")
print("   • Medium windows: 1.0s, 1.2s")
print("   • Large windows: 1.8s, 2.0s")
print("   • Representative strides for each")
print()
print("   Command:")
print("   python scripts/comprehensive_evaluation_processor.py --csv_file csv/eval_by_0.05/window_0.5s/stride_0.25s.csv --output_dir focused_eval_results")
print()

print("2. SAMPLING EVALUATION")
print("   Evaluate a representative sample from each dataset:")
print("   • Every 3rd window size")
print("   • 2-3 stride configurations per window")
print("   • ~50 files total instead of 406")
print()

print("3. SPECIFIC COMPARISON")
print("   Compare specific configurations between datasets:")
print("   • Same window/stride concept in both eval_by_0.05 and eval_percent")
print("   • Example: window_1.0s with 50% stride vs stride_0.5s")
print()

print("4. PROGRESSIVE EVALUATION")
print("   Start small and expand based on results:")
print("   • Phase 1: Test 10 configurations")
print("   • Phase 2: Expand to promising ranges")
print("   • Phase 3: Full evaluation if needed")
print()

print("🚀 IMMEDIATE OPTIONS:")
print()

print("Option A: Quick Test (5 minutes)")
print("python scripts/batch_evaluation_processor.py --max_configs 3 --datasets csv/eval_by_0.05")
print()

print("Option B: Single Dataset Focus (3-4 hours)")  
print("python scripts/batch_evaluation_processor.py --datasets csv/eval_by_0.05")
print("# OR")
print("python scripts/batch_evaluation_processor.py --datasets csv/eval_percent")
print()

print("Option C: Specific Configurations")
print("# Evaluate specific promising configurations")
print("python scripts/comprehensive_evaluation_processor.py --csv_file csv/eval_by_0.05/window_1.0s/stride_0.5s.csv --output_dir specific_eval")
print()

print("Option D: Resume/Parallel Evaluation")
print("python scripts/batch_evaluation_processor.py --skip_existing")
print("# Run multiple terminals with different window ranges")
print()

print("📈 PERFORMANCE INSIGHTS:")
print("• Based on previous results, window 2.0s / stride 2.0s showed:")
print("  - Combined Mean IoU: 0.0168")
print("  - Window-level Binary F1: 0.1701") 
print("  - Word-level Binary F1: 0.2700")
print("• Consider testing variations around this configuration")
print()

print("🔧 CUSTOM EVALUATION:")
print("You can also evaluate specific files directly:")
print("python scripts/comprehensive_evaluation_processor.py \\")
print("    --csv_file csv/eval_percent/window_1.0s/stride_50.0%.csv \\")
print("    --output_dir percent_eval_test")
print()

print("Which approach would you like to try?")

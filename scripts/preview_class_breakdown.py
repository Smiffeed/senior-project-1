#!/usr/bin/env python3
"""
Test what the new Ground Truth vs Prediction Breakdown section will look like
"""

def simulate_class_breakdown():
    """Simulate what the new section will show based on your data"""
    print("=== GROUND TRUTH vs PREDICTION BREAKDOWN BY CLASS ===")
    print("Detailed analysis of each class: ground truth count vs correct/incorrect predictions")
    print()
    
    # Simulate data based on your note.txt results
    class_data = {
        'เย็ด': {'gt_count': 26, 'correct': 0, 'incorrect': 26},    # Based on your results
        'กู': {'gt_count': 157, 'correct': 14, 'incorrect': 143},   # 9% accuracy from your data
        'มึง': {'gt_count': 117, 'correct': 9, 'incorrect': 108},   # 8% accuracy from your data  
        'เหี้ย': {'gt_count': 78, 'correct': 26, 'incorrect': 52},  # 33% accuracy from your data
        'หี': {'gt_count': 20, 'correct': 3, 'incorrect': 17},     # Estimated
        'ควย': {'gt_count': 15, 'correct': 2, 'incorrect': 13},    # Estimated
        'none': {'gt_count': 287, 'correct': 175, 'incorrect': 112} # 61% accuracy from your data
    }
    
    print("Class Analysis:")
    print("-" * 70)
    print(f"{'Class':<12} {'GT Count':<10} {'Correct':<10} {'Incorrect':<12} {'Accuracy':<10}")
    print("-" * 70)
    
    total_gt = 0
    total_correct = 0
    total_incorrect = 0
    
    # Sort: profanity first, then none
    profanity_classes = [cls for cls in class_data.keys() if cls != 'none']
    
    for class_name in sorted(profanity_classes) + ['none']:
        data = class_data[class_name]
        gt_count = data['gt_count']
        correct = data['correct']
        incorrect = data['incorrect']
        accuracy = (correct / gt_count) * 100 if gt_count > 0 else 0.0
        
        print(f"{class_name:<12} {gt_count:<10} {correct:<10} {incorrect:<12} {accuracy:<9.1f}%")
        
        total_gt += gt_count
        total_correct += correct
        total_incorrect += incorrect
    
    print("-" * 70)
    overall_accuracy = (total_correct / total_gt) * 100 if total_gt > 0 else 0.0
    print(f"{'TOTAL':<12} {total_gt:<10} {total_correct:<10} {total_incorrect:<12} {overall_accuracy:<9.1f}%")
    print("-" * 70)
    print()
    
    print("Detailed Explanation:")
    for class_name in sorted(profanity_classes) + ['none']:
        data = class_data[class_name]
        gt_count = data['gt_count']
        correct = data['correct']
        incorrect = data['incorrect']
        
        if class_name == 'none':
            print(f"• {class_name}: {gt_count} regions in ground truth")
            print(f"  - {correct} correctly identified (no profanity predictions overlap)")
            print(f"  - {incorrect} incorrectly identified (profanity predictions overlap)")
        else:
            print(f"• {class_name}: {gt_count} words in ground truth")
            print(f"  - {correct} correctly predicted (sufficient IoU + correct label)")
            print(f"  - {incorrect} incorrectly predicted (insufficient IoU or wrong label)")
    
    print()
    print("This breakdown helps you understand:")
    print("1. Which classes have the most ground truth instances")
    print("2. Which classes are hardest to predict correctly")
    print("3. Where your model needs improvement")

if __name__ == "__main__":
    print("Preview of New Ground Truth vs Prediction Breakdown Section")
    print("=" * 60)
    print()
    simulate_class_breakdown()

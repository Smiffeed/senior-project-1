#!/usr/bin/env python3
"""
Preview of the new binary classification analysis in IoU evaluation
"""

def preview_binary_analysis():
    print("=== NEW BINARY CLASSIFICATION ANALYSIS FOR IoU EVALUATION ===")
    print()
    
    print("For EACH IoU threshold (0.1, 0.2, 0.3, ..., 0.9), you will now see:")
    print()
    
    # Example for IoU threshold 0.3
    threshold = 0.3
    print(f"--- IoU Threshold {threshold} ---")
    print("Binary Classification (Profane vs Non-Profane):")
    print("  Accuracy: 0.5607")
    print("  Balanced Accuracy: 0.5793") 
    print("  Precision: 0.8294")
    print("  Recall: 0.2589")
    print("  F1-Score: 0.3946")
    print()
    
    print("Multiclass Classification (All Classes):")
    print("  Accuracy: 0.3299")
    print("  Balanced Accuracy: 0.2023")
    print("  [Classification Report...]")
    print()
    
    print("*** NEW: Binary Classification Analysis (IoU >= 0.3) ***")
    print("  IoU Threshold: 0.3")
    print("  Detections with IoU >= 0.3: 45/378 (11.9%)")
    print("  Binary Confusion Matrix:")
    print("                    Predicted")  
    print("                    No-Prof  Profanity")
    print("  Actual No-Prof        250         37")
    print("  Actual Profanity      333         45")
    print("  ")
    print("  Binary Metrics:")
    print("    Precision: 0.5488")
    print("    Recall: 0.1191") 
    print("    F1-Score: 0.1942")
    print("    Accuracy: 0.4436")
    print()
    
    print("=== NEW: ENHANCED IoU THRESHOLD ANALYSIS ===")
    print("Percentage of ground truth words that achieve different IoU thresholds:")
    print()
    
    print("Overall IoU Distribution:")
    print("  IoU >= 0.1: 28/665 (4.2%)")
    print("  IoU >= 0.2: 7/665 (1.1%)")
    print("  IoU >= 0.3: 1/665 (0.2%)")
    print("  IoU >= 0.4: 1/665 (0.2%)")
    print("  IoU >= 0.5: 1/665 (0.2%)")
    print()
    
    print("Profanity Words IoU Distribution:")
    print("  IoU >= 0.1: 28/378 (7.4%)")
    print("  IoU >= 0.2: 7/378 (1.9%)")
    print("  IoU >= 0.3: 1/378 (0.3%)")
    print("  IoU >= 0.4: 1/378 (0.3%)")
    print("  IoU >= 0.5: 1/378 (0.3%)")
    print()
    
    print("IoU Range Distribution (Profanity Words Only):")
    print("  IoU 0.0-0.1: 350/378 (92.6%)")
    print("  IoU 0.1-0.3: 21/378 (5.6%)")
    print("  IoU 0.3-0.5: 0/378 (0.0%)")
    print("  IoU 0.5-0.7: 0/378 (0.0%)")
    print("  IoU 0.7-0.9: 0/378 (0.0%)")
    print("  IoU 0.9-1.0: 1/378 (0.3%)")
    print()
    
    print("Binary Detection Summary (Profanity vs None):")
    print("  IoU >= 0.1: Detected 28/378 profanity, Correct 250/287 none (Accuracy: 41.8%)")
    print("  IoU >= 0.3: Detected 1/378 profanity, Correct 250/287 none (Accuracy: 37.7%)")
    print("  IoU >= 0.5: Detected 1/378 profanity, Correct 250/287 none (Accuracy: 37.7%)")
    print("  IoU >= 0.7: Detected 0/378 profanity, Correct 250/287 none (Accuracy: 37.6%)")
    print("  IoU >= 0.9: Detected 0/378 profanity, Correct 250/287 none (Accuracy: 37.6%)")
    print()
    
    print("=== BENEFITS OF NEW ANALYSIS ===")
    print("1. ✅ Binary confusion matrix for each IoU threshold")
    print("2. ✅ IoU percentage distribution analysis")
    print("3. ✅ Separate profanity vs overall IoU statistics") 
    print("4. ✅ Binary detection summary across key thresholds")
    print("5. ✅ Clear understanding of detection vs classification performance")
    print()
    
    print("This helps you understand:")
    print("- How many detections meet each IoU threshold")
    print("- Binary classification performance at each threshold")
    print("- IoU distribution patterns in your predictions")
    print("- Trade-offs between detection sensitivity and precision")

if __name__ == "__main__":
    preview_binary_analysis()

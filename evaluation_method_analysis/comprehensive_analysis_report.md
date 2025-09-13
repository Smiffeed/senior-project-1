# EVALUATION METHOD ANALYSIS REPORT
==================================================

## 1. BASIC PERFORMANCE COMPARISON
- Old Method F1 Score: 0.1649 ± 0.0446
- New Method Word F1: 0.5614 ± 0.1960
- Performance Improvement: 3.40x

## 2. WINDOW SIZE EFFECTS
- Old Method Window-F1 Correlation: 0.987
- New Method Window-F1 Correlation: -0.901
- Old Method: F1 INCREASES with larger windows
- New Method: F1 DECREASES with larger windows

## 3. OPTIMAL CONFIGURATIONS
- Old Method Best: window_2.0s + stride_1.7s (F1: 0.2333)
- New Method Best: window_0.3s + stride_10.0% (F1: 0.9960)

## 4. KEY DIFFERENCES IDENTIFIED

### OPPOSITE WINDOW SIZE TRENDS:
- Old method benefits from LARGER windows
- New method benefits from SMALLER windows
- This suggests different underlying evaluation mechanisms

### PERFORMANCE VARIABILITY:
- Old Method F1 Range: 0.1877
- New Method F1 Range: 0.9699
- New method shows HIGHER variability (more sensitive to configuration)

### OLD METHOD PRECISION-RECALL PATTERN:
- Average Precision: 0.0941
- Average Recall: 0.7476
- High recall, low precision → Many false positives
- The old method is overly sensitive (detects too much)

## 5. POTENTIAL REASONS FOR PERFORMANCE DIFFERENCE

### A. EVALUATION METHODOLOGY:
- Old method likely uses IoU-based word evaluation with strict overlap requirements
- New method may use more sophisticated word-level matching criteria
- Different ground truth alignment strategies

### B. WINDOW SIZE SENSITIVITY:
- Old method: Larger windows provide more context → better detection
- New method: Smaller windows provide better precision → less false positives
- Suggests new method has better localization accuracy

### C. ALGORITHM IMPROVEMENTS:
- New method likely incorporates advanced NLP techniques
- Better handling of word boundaries and segmentation
- Improved feature extraction and classification algorithms

## 6. RECOMMENDATIONS

1. **Use New Method**: Clearly superior performance across all metrics
2. **Optimal Configuration**: Small windows (0.3-0.5s) with moderate stride
3. **Further Investigation**: Analyze specific cases where old method fails
4. **Hybrid Approach**: Consider combining strengths of both methods
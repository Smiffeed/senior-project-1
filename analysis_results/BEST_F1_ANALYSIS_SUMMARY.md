# Best F1 Window and Stride Configuration Analysis Summary

**Analysis Date:** September 11, 2025  
**Total Configurations Tested:** 1,619 across 4 evaluation methods  
**Analysis Output:** Comprehensive CSV files and visualizations

## 📊 Evaluation Methods Analyzed

1. **IoU (Intersection over Union)** - 405 configurations
2. **IoU Word Separated** - 404 configurations  
3. **Window-based** - 405 configurations
4. **Word-level** - 405 configurations

## 🏆 Key Findings: Best F1 Configurations by Method

### 1. IoU Evaluation Method
- **Best Binary F1:** 0.2333
  - Window: 2.0s, Stride: 1.7
  - Accuracy: 37.56%, Precision: 13.82%, Recall: 74.93%
- **Best Word-level F1:** 0.1947
  - Window: 2.0s, Stride: 30.0% (0.3 numeric)
  - Precision: 10.74%, Recall: 104.12%
- **Best IoU F1:** 0.0884
  - Window: 1.5s, Stride: 1.35
  - Precision: 5.20%, Recall: 29.64%, Mean IoU: 28.0%

### 2. IoU Word Separated Method
- **Best Binary F1:** 0.2333 (same as IoU)
  - Window: 2.0s, Stride: 1.7
- **Best Word-level F1:** 0.1947 (same as IoU)
  - Window: 2.0s, Stride: 30.0%
- **Best IoU F1:** 0.0916 (slightly better than regular IoU)
  - Window: 1.5s, Stride: 1.35

### 3. Window-based Evaluation Method
- **Best Binary F1:** 0.2333 (consistent across methods)
  - Window: 2.0s, Stride: 1.7
- **Best Multiclass F1:** 0.4626 ⭐ **HIGHEST OVERALL**
  - Window: 0.7s, Stride: 0.7
  - Accuracy: 53.64%, Precision: 61.10%, Recall: 48.93%

### 4. Word-level Evaluation Method
- **Best Binary F1:** 0.2333 (consistent)
  - Window: 2.0s, Stride: 1.7
- **Best Word-level F1:** 0.1947
  - Window: 2.0s, Stride: 30.0%

## 🎯 Overall Best Configurations Summary

| Score Type | Best F1 Score | Method | Window | Stride | Key Metrics |
|------------|---------------|---------|---------|---------|-------------|
| **Binary F1** | 0.2333 | All methods | 2.0s | 1.7 | Acc: 37.56%, Rec: 74.93% |
| **Multiclass F1** | **0.4626** | Window | **0.7s** | **0.7** | Acc: 53.64%, Prec: 61.10% |
| **Word-level F1** | 0.1947 | IoU/Word | 2.0s | 30.0% | Rec: 104.12% |
| **IoU F1** | 0.0916 | IoU Word Sep | 1.5s | 1.35 | Mean IoU: 28.0% |

## 📈 Key Insights

1. **Consistent Binary Performance:** All methods achieve the same best binary F1 score (0.2333) with identical window/stride configuration (2.0s/1.7)

2. **Multiclass Excellence:** The Window-based method significantly outperforms others for multiclass classification with a much smaller window/stride (0.7s/0.7)

3. **Short vs Long Windows:** 
   - Short windows (0.7s) work best for multiclass classification
   - Longer windows (2.0s) work best for binary classification
   - Medium windows (1.5s) work best for IoU-based evaluation

4. **Stride Patterns:**
   - Large stride (1.7) for binary classification
   - Equal window/stride (0.7/0.7) for multiclass
   - Percentage-based stride (30%) for word-level evaluation

## 📁 Generated Files

- **Summary CSV:** `best_f1_configurations_summary_20250911_013718.csv`
- **Detailed CSV:** `best_f1_configurations_detailed_20250911_013718.csv`
- **Overview Visualization:** `best_f1_configurations_overview_20250911_013718.png`
- **Detailed Method Plots:** Individual plots for each evaluation method

## 🔍 Next Steps Recommendations

Based on your specific use case:

- **For Binary Classification:** Use Window=2.0s, Stride=1.7 (consistent across all methods)
- **For Multiclass Classification:** Use Window=0.7s, Stride=0.7 (Window method)
- **For Word-level Detection:** Use Window=2.0s, Stride=30% (IoU/Word methods)
- **For IoU-based Evaluation:** Use Window=1.5s, Stride=1.35 (IoU Word Sep method)

The analysis shows clear patterns that can guide parameter selection based on your evaluation objectives.

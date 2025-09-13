# Line Graph Analysis Summary for Evaluation Methods

**Created on:** September 11, 2025  
**Total Graphs Generated:** 13 comprehensive line graph visualizations

## 📊 Line Graphs Created

### 1. **Individual Method Analysis (by Window Size)**
- `line_graphs_iou_by_window_20250911_014600.png`
- `line_graphs_iou_word_sep_by_window_20250911_014600.png`
- `line_graphs_window_by_window_20250911_014600.png`
- `line_graphs_word_by_window_20250911_014600.png`

**What they show:** How F1 scores change with stride values for different fixed window sizes within each evaluation method.

### 2. **Individual Method Analysis (by Stride Value)**
- `line_graphs_iou_by_stride_20250911_014606.png`
- `line_graphs_iou_word_sep_by_stride_20250911_014606.png`
- `line_graphs_window_by_stride_20250911_014606.png`
- `line_graphs_word_by_stride_20250911_014606.png`

**What they show:** How F1 scores change with window sizes for different fixed stride values within each evaluation method.

### 3. **Cross-Method Comparison**
- `combined_evaluation_comparison_20250911_014611.png`

**What it shows:** Direct comparison of all evaluation methods on the same plots, showing how each method performs across different parameter ranges.

### 4. **Best Configuration Trends**
- `best_configurations_trends_20250911_014612.png`

**What it shows:** Performance trends around the best configurations identified in our analysis, highlighting the optimal parameter regions.

## 📈 Key Trends Discovered

### **Strong Positive Correlations (>0.9)**
1. **Binary F1 vs Window Size:** 0.988 correlation across all methods
   - Larger windows (1.9-2.0s) consistently produce better binary F1 scores
   - Very strong linear relationship

2. **Word-level F1 vs Window Size:** 0.932 correlation
   - Word-level evaluation also benefits significantly from larger windows
   - Requires more context for accurate word boundary detection

### **Negative Correlations (Multiclass)**
1. **Multiclass F1 vs Window Size:** -0.918 correlation
   - **Opposite trend:** Smaller windows (0.3-0.7s) work better for multiclass
   - Shorter analysis windows prevent class confusion

### **Moderate Correlations (Stride Effects)**
1. **Binary F1 vs Stride:** 0.532 correlation
2. **IoU F1 vs Stride:** 0.576-0.582 correlation
3. **Multiclass F1 vs Stride:** -0.522 correlation (negative)

## 🎯 Optimal Parameter Ranges by Score Type

| Score Type | Optimal Window Range | Optimal Stride Range | Most Common Config |
|------------|---------------------|---------------------|-------------------|
| **Binary F1** | 1.9-2.0s | 0.300-1.950 | Window: 2.0s, Stride: 1.250 |
| **Multiclass F1** | 0.3-1.1s | 0.100-1.050 | Window: 0.3s, Stride: 0.200 |
| **Word-level F1** | 1.8-2.0s | 0.100-1.900 | Window: 2.0s, Stride: 0.300 |
| **IoU F1** | 1.2-2.0s | 0.500-1.800 | Medium windows preferred |

## 📋 Line Graph Reading Guide

### **By Window Size Graphs:**
- **X-axis:** Stride values (0.1 to 2.0)
- **Y-axis:** F1 scores
- **Different lines:** Different window sizes (0.3s to 2.0s)
- **Key insight:** Shows how stride affects performance for each window size

### **By Stride Value Graphs:**
- **X-axis:** Window sizes (0.3s to 2.0s)
- **Y-axis:** F1 scores
- **Different lines:** Different stride values
- **Key insight:** Shows how window size affects performance for each stride

### **Combined Comparison Graph:**
- **Multiple subplots:** Binary, Multiclass, Word-level comparisons
- **Different colors:** Different evaluation methods
- **Key insight:** Direct method-to-method performance comparison

## 🔍 Key Insights from Line Graphs

### **1. Clear Method Distinctions:**
- **IoU methods:** Show gradual, smooth improvements
- **Window method:** Shows sharp peaks for multiclass performance
- **Word method:** More sensitive to stride variations
- **All methods:** Consistent binary performance patterns

### **2. Parameter Sensitivity:**
- **Window size:** High impact on all score types
- **Stride value:** Moderate impact, method-dependent
- **Interaction effects:** Window-stride combinations matter

### **3. Trade-off Patterns:**
- **Large windows:** Better for binary and word-level (more context)
- **Small windows:** Better for multiclass (less confusion)
- **Large strides:** Faster processing, potentially less precision
- **Small strides:** More detailed analysis, higher computation cost

### **4. Convergence Points:**
- **Binary F1:** All methods converge to ~0.233 at optimal settings
- **Multiclass F1:** Only window method achieves high performance (0.463)
- **Word/IoU scores:** Show method-specific optimal points

## 💡 Practical Recommendations

### **For Binary Classification:**
- Use **any evaluation method** (all perform similarly)
- Set **window = 2.0s, stride = 1.7**
- Expect F1 score around **0.233**

### **For Multiclass Classification:**
- Use **window evaluation method only**
- Set **window = 0.7s, stride = 0.7**
- Expect F1 score around **0.463**

### **For Word-level Detection:**
- Use **IoU or Word methods**
- Set **window = 2.0s, stride = 30%**
- Expect F1 score around **0.195**

### **For IoU-based Evaluation:**
- Use **IoU Word Separated method**
- Set **window = 1.5s, stride = 1.35**
- Expect F1 score around **0.092**

## 📁 Files Location
All line graphs are saved in: `analysis_results/` directory

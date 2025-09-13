# 📊 COMPREHENSIVE F1 SCORE CORRELATION ANALYSIS RESULTS

## 🎯 **KEY FINDINGS SUMMARY**

Based on the analysis of **406 successful evaluations** across both `eval_by_0.05` (227 configs) and `eval_percent` (179 configs) datasets, here are the critical insights about window size, stride, and F1 score correlations:

---

## 🏆 **TOP PERFORMING CONFIGURATIONS**

### **🥇 Best Window F1 Score: 0.3333**
- **Configuration**: `eval_by_0.05` | `2.0s window` | `100% stride` (stride_2.0s_sample)
- **Word F1**: 0.0261 | **IoU**: 0.0011
- **⚠️ Note**: This appears to be a special sample configuration with very high Window F1 but poor Word F1

### **🥈 Best Practical Window F1 Score: 0.2126**
- **Configuration**: `eval_by_0.05` | `1.2s window` | `70.8% stride` (stride_0.85s)
- **Word F1**: 0.5147 | **IoU**: 0.0402

### **🏅 Best Word F1 Score: 0.996**
- **Configuration**: `eval_percent` | `0.3s window` | `10% stride`
- **Window F1**: 0.0729 | **IoU**: 0.1328

---

## 📈 **CORRELATION ANALYSIS**

### **Window Size Effects:**
- **Window vs Window F1**: `+0.6086` ✅ **Strong positive correlation**
- **Window vs Word F1**: `-0.9011` ❌ **Very strong negative correlation**
- **Window vs IoU**: `-0.8874` ❌ **Very strong negative correlation**

### **Stride Percentage Effects:**
- **Stride% vs Window F1**: `+0.0699` ⚪ **Very weak correlation**
- **Stride% vs Word F1**: `-0.4759` ⚠️ **Moderate negative correlation**
- **Stride% vs IoU**: `-0.0993` ⚪ **Very weak correlation**

### **Stride Absolute Effects:**
- **Stride vs Window F1**: `+0.4418` ⚠️ **Moderate positive correlation**
- **Stride vs Word F1**: `-0.9173` ❌ **Very strong negative correlation**
- **Stride vs IoU**: `-0.6762` ❌ **Strong negative correlation**

---

## 🎯 **OPTIMAL CONFIGURATIONS BY WINDOW SIZE**

| Window | Best Window F1 | Word F1 | IoU | Optimal Stride% | Eval Type | Configuration |
|--------|----------------|---------|-----|-----------------|-----------|---------------|
| 0.3s   | **0.0744**     | 0.9169  | 0.2486 | 83.3%         | eval_by_0.05 | window_0.3s/stride_0.25s |
| 0.4s   | **0.1027**     | 0.8700  | 0.1941 | 87.5%         | eval_by_0.05 | window_0.4s/stride_0.35s |
| 0.5s   | **0.1288**     | 0.8800  | 0.1553 | 60.0%         | eval_by_0.05 | window_0.5s/stride_0.3s |
| 0.6s   | **0.1564**     | 0.8458  | 0.1370 | 66.7%         | eval_by_0.05 | window_0.6s/stride_0.4s |
| 0.7s   | **0.1732**     | 0.7121  | 0.1070 | 90.0%         | eval_percent | window_0.7s/stride_90.0% |
| 0.8s   | **0.1838**     | 0.7379  | 0.0960 | 62.5%         | eval_by_0.05 | window_0.8s/stride_0.5s |
| 0.9s   | **0.1955**     | 0.7208  | 0.0780 | 55.6%         | eval_by_0.05 | window_0.9s/stride_0.5s |
| 1.0s   | **0.2003**     | 0.6524  | 0.0630 | 55.0%         | eval_by_0.05 | window_1.0s/stride_0.55s |
| 1.1s   | **0.2073**     | 0.5517  | 0.0519 | 81.8%         | eval_by_0.05 | window_1.1s/stride_0.9s |
| 1.2s   | **0.2126**     | 0.5147  | 0.0402 | 70.8%         | eval_by_0.05 | window_1.2s/stride_0.85s |

---

## ⚖️ **EVALUATION TYPE COMPARISON**

### **eval_by_0.05 vs eval_percent Performance:**

| Metric | eval_by_0.05 | eval_percent | Winner |
|--------|--------------|--------------|---------|
| **Window F1** | Mean: 0.1540 | Mean: 0.1731 | **eval_percent** 📈 |
| **Word F1** | Mean: 0.6583 | Mean: 0.6584 | **Tie** ⚖️ |
| **Combined IoU** | Mean: 0.1015 | Mean: 0.0958 | **eval_by_0.05** 📈 |

---

## 💡 **CRITICAL INSIGHTS & TRADE-OFFS**

### **🔍 Window Size Trade-off:**
- **Larger windows (1.0s-2.0s)**: ✅ Better Window F1 scores | ❌ Worse Word F1 scores
- **Smaller windows (0.3s-0.6s)**: ❌ Lower Window F1 scores | ✅ Much better Word F1 scores
- **Sweet spot**: **0.5s-0.8s** windows provide good balance

### **🎯 Stride Impact:**
- **Stride percentage has minimal direct impact** on Window F1 (correlation: +0.07)
- **Smaller strides generally better for Word F1** (correlation: -0.48)
- **Optimal stride percentages vary by window size** (55%-90% range)

### **📊 Performance Patterns:**
1. **Best Window F1**: Achieved with larger windows (1.0s+) and high stride percentages
2. **Best Word F1**: Achieved with smaller windows (0.3s-0.4s) and very small strides (10-20%)
3. **Best IoU**: Generally correlates with Word F1 performance

---

## 🎯 **RECOMMENDATIONS**

### **For Maximum Window F1 Performance:**
- **Use**: 1.0s-1.2s windows with 70-85% stride
- **Best Config**: `window_1.2s/stride_0.85s` (eval_by_0.05)
- **Expected**: Window F1 ~0.21, Word F1 ~0.51

### **For Maximum Word F1 Performance:**
- **Use**: 0.3s-0.4s windows with 10-20% stride
- **Best Config**: `window_0.3s/stride_10.0%` (eval_percent)
- **Expected**: Word F1 ~0.996, Window F1 ~0.07

### **For Balanced Performance:**
- **Use**: 0.5s-0.8s windows with 55-65% stride
- **Recommended**: `window_0.6s/stride_0.4s` (eval_by_0.05)
- **Expected**: Window F1 ~0.16, Word F1 ~0.85, IoU ~0.14

### **Evaluation Method Choice:**
- **eval_percent** slightly better for Window F1 tasks
- **eval_by_0.05** slightly better for IoU tasks
- **No significant difference** for Word F1 tasks

---

## 📁 **Generated Analysis Files**

All detailed analysis results are saved in:
- `fixed_smart_parallel_results/correlation_analysis/complete_analysis_data.csv`
- `fixed_smart_parallel_results/correlation_analysis/correlation_matrix.csv`
- `fixed_smart_parallel_results/correlation_analysis/top_10_window_f1_configs.csv`
- `fixed_smart_parallel_results/correlation_analysis/top_10_word_f1_configs.csv`
- `fixed_smart_parallel_results/correlation_analysis/best_config_per_window.csv`

---

## 🚀 **Next Steps**

1. **Focus on your specific use case**: Choose Window F1 vs Word F1 optimization based on your task requirements
2. **Test the recommended configurations** in your specific application context
3. **Consider ensemble approaches** combining multiple window sizes for robust performance
4. **Monitor GPU memory usage** when using smaller strides (they require more processing)

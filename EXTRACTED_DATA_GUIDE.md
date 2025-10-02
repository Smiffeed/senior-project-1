# Extracted Evaluation Data Guide

## 📊 Overview

This document provides a comprehensive guide to the extracted evaluation data from 406 frame-level evaluation configurations. The data has been processed and organized for easy analysis and visualization.

## 📁 Data Files Description

### Core Data Files

#### `complete_evaluation_data.csv`
**Complete dataset with all configurations and metrics**
- **Total Records**: 406 configurations
- **Evaluation Types**: 
  - `eval_percent`: 180 configurations (percentage-based strides: 10%-100%)
  - `eval_by_0`: 226 configurations (absolute time-based strides: 0.125s-2.0s)
- **Window Sizes**: 0.3s - 2.0s (18 different sizes)
- **Key Columns**:
  - `window_size`: Window duration in seconds
  - `stride_value`: Stride value (percentage or seconds)
  - `stride_type`: Either "percentage" or absolute time
  - `eval_type`: Evaluation method used
  - `combined_mean_iou`: Overall IoU percentage (4.21% - 29.82%)
  - `binary_f1`: Binary classification F1 score (0.340 - 0.400)
  - `multiclass_f1`: 4-class classification F1 score (0.330 - 0.400)
  - `iou_by_class`: Dictionary with IoU for each Thai profanity class
  - `binary_accuracy`, `binary_precision`, `binary_recall`: Additional metrics

### Heatmap Data Files

#### `eval_percent_binary_f1_heatmap.csv`
**Binary F1 scores organized as heatmap data**
- **Rows**: Stride percentages (10% - 100%)
- **Columns**: Window sizes (0.3s - 2.0s)
- **Values**: Binary F1 scores for direct heatmap plotting

#### `eval_percent_combined_mean_iou_heatmap.csv`
**Combined Mean IoU percentages as heatmap data**
- **Rows**: Stride percentages (10% - 100%)
- **Columns**: Window sizes (0.3s - 2.0s)
- **Values**: IoU percentages for direct heatmap plotting

#### `eval_percent_iou_[class]_heatmap.csv`
**Individual class IoU heatmaps for each Thai profanity class**
- Files for: `เย็ด`, `กู`, `มึง`, `เหี้ย`
- Same structure as combined IoU heatmap
- Values represent class-specific IoU percentages

### Analysis Files

#### `optimal_configurations.csv`
**Best performing configurations for each metric**
- **Binary F1 Optimal**: Window=1.6s, Stride=100%, F1=0.400
- **Multiclass F1 Optimal**: Window=2.0s, Stride=100%, F1=0.400  
- **Combined IoU Optimal**: Window=0.3s, Stride=80%, IoU=29.82%

#### `summary_statistics.json`
**Statistical overview of the dataset**
- Performance ranges for each evaluation type
- Window size and stride distributions
- Overall dataset characteristics

## 🎯 Key Findings

### Performance Insights

1. **Best Overall Performance**:
   - **Highest Binary F1**: 0.400 (Window=1.6s, Stride=100%)
   - **Highest Multiclass F1**: 0.400 (Window=2.0s, Stride=100%)
   - **Highest IoU**: 29.82% (Window=0.3s, Stride=80%)

2. **Window Size Trends**:
   - **For F1 Scores**: Larger windows (1.6s-2.0s) perform better
   - **For IoU**: Smaller windows (0.3s-0.6s) achieve higher IoU
   - **Trade-off**: IoU vs F1 performance inversely correlated

3. **Stride Impact**:
   - **High Strides (80-100%)**: Better for individual metrics
   - **Medium Strides (40-60%)**: More balanced performance
   - **Low Strides (10-20%)**: Higher computational cost, mixed results

### Evaluation Method Comparison

#### `eval_percent` (180 configs)
- **Average Binary F1**: 0.367
- **Average Multiclass F1**: 0.356
- **Average IoU**: 13.18%
- **Stride Range**: 10%-100% (percentage-based)

#### `eval_by_0` (226 configs)  
- **Average Binary F1**: 0.372
- **Average Multiclass F1**: 0.365
- **Average IoU**: 11.86%
- **Stride Range**: 0.125s-2.0s (absolute time-based)

## 📈 How to Use This Data

### For Plotting Heatmaps

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load heatmap data
heatmap_data = pd.read_csv('eval_percent_binary_f1_heatmap.csv', index_col=0)

# Create heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r')
plt.title('Binary F1 Score Heatmap')
plt.xlabel('Window Size (seconds)')
plt.ylabel('Stride (%)')
plt.show()
```

### For Line Graph Analysis

```python
# Load complete dataset
df = pd.read_csv('complete_evaluation_data.csv')

# Group by window size for trend analysis
window_trends = df.groupby('window_size').agg({
    'binary_f1': ['mean', 'std'],
    'multiclass_f1': ['mean', 'std'],
    'combined_mean_iou': ['mean', 'std']
})

# Plot trends
plt.figure(figsize=(12, 6))
plt.plot(window_trends.index, window_trends[('binary_f1', 'mean')], 'o-', label='Binary F1')
plt.fill_between(window_trends.index, 
                 window_trends[('binary_f1', 'mean')] - window_trends[('binary_f1', 'std')],
                 window_trends[('binary_f1', 'mean')] + window_trends[('binary_f1', 'std')],
                 alpha=0.2)
plt.xlabel('Window Size (seconds)')
plt.ylabel('F1 Score')
plt.legend()
plt.show()
```

### For Custom Analysis

```python
# Find configurations meeting specific criteria
df = pd.read_csv('complete_evaluation_data.csv')

# High performance configurations
high_perf = df[(df['binary_f1'] > 0.38) & (df['combined_mean_iou'] > 20)]
print(f"High performance configs: {len(high_perf)}")

# Balanced performance
balanced = df[(df['binary_f1'] > 0.36) & (df['multiclass_f1'] > 0.35) & (df['combined_mean_iou'] > 15)]
print(f"Balanced configs: {len(balanced)}")

# Analyze by window size
window_analysis = df.groupby('window_size').agg({
    'binary_f1': 'max',
    'combined_mean_iou': 'max'
}).round(3)
print(window_analysis)
```

## 🔧 Recreation Scripts

### `extract_evaluation_data.py`
- **Purpose**: Extract data from note.txt files in evaluation results
- **Output**: All CSV and JSON files described above
- **Usage**: `python extract_evaluation_data.py`

### `analyze_extracted_data.py`  
- **Purpose**: Analyze extracted data and recreate visualizations
- **Output**: Statistical analysis and PNG graph files
- **Usage**: `python analyze_extracted_data.py`

## 📊 Visualization Outputs

The analysis generates these visualization files in `./visualization_outputs/`:

1. **`eval_percent_binary_f1_heatmap.png`**: Binary F1 score heatmap
2. **`eval_percent_iou_heatmap.png`**: Combined Mean IoU heatmap  
3. **`method_comparison_line_graphs.png`**: Comparison of evaluation methods
4. **`iou_evaluation_line_graph.png`**: IoU trends across window sizes

## 🎯 Recommended Configurations

Based on the analysis, here are the recommended configurations for different use cases:

### For Maximum Binary Classification Performance
- **Window**: 1.6s - 2.0s
- **Stride**: 80% - 100%
- **Expected Binary F1**: 0.38 - 0.40
- **Trade-off**: Lower IoU (6-8%)

### For Maximum IoU Performance
- **Window**: 0.3s - 0.5s  
- **Stride**: 60% - 80%
- **Expected IoU**: 25% - 30%
- **Trade-off**: Lower F1 scores (0.36-0.38)

### For Balanced Performance
- **Window**: 0.6s - 1.0s
- **Stride**: 50% - 70%
- **Expected Performance**: F1=0.36-0.37, IoU=15-20%
- **Advantage**: Good compromise between metrics

## 💡 Usage Tips

1. **For Publications**: Use `optimal_configurations.csv` to report best results
2. **For Comparisons**: Use `complete_evaluation_data.csv` for statistical analysis
3. **For Visualizations**: Use pre-formatted heatmap CSV files
4. **For Trends**: Group by window_size or stride_value in complete dataset
5. **For Class Analysis**: Use individual class IoU heatmap files

## 📝 Data Quality Notes

- All 406 configurations successfully processed
- No missing values in core metrics
- IoU values converted to percentages for consistency
- Thai characters properly encoded in class names
- Stride types clearly distinguished (percentage vs absolute)

This extracted data provides a comprehensive foundation for analysis, visualization, and decision-making regarding optimal frame-level evaluation configurations for the profanity detection system.
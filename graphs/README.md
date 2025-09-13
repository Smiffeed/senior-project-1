# Graph Organization Structure

This folder contains all generated graphs and visualizations organized by evaluation type.

## Folder Structure

```
graphs/
├── eval_window/     # Window-based evaluation graphs
├── eval_word/       # Word-level evaluation graphs  
├── eval_iou/        # IoU (Intersection over Union) evaluation graphs
└── eval_other/      # Other evaluation types and miscellaneous graphs
```

## Current Contents

### eval_window/
Contains graphs related to window stride and window size analysis:
- `main_stride_f1_correlation.png` - Main correlation plots between stride and F1 scores
- `heatmap_stride_f1_analysis.png` - Original heatmap analysis
- `optimal_stride_analysis.png` - Optimal stride analysis by window size
- `clean_heatmap_analysis.png` - Simplified, readable heatmaps
- `detailed_performance_grid.png` - Detailed grid with exact values
- `comprehensive_detailed_heatmap.png` - Complete grid with all values
- `split_detailed_heatmaps.png` - Split by stride ranges for clarity
- `stride_f1_correlation_analysis.png` - Original correlation analysis
- `individual_class_f1_stride_analysis.png` - Individual class analysis

### eval_word/
*Reserved for word-level evaluation graphs*

### eval_iou/
*Reserved for IoU evaluation graphs*

### eval_other/
*Reserved for other evaluation types*

## Usage

When creating new graphs:
1. Determine the evaluation type
2. Save graphs to the appropriate subfolder
3. Use descriptive filenames that indicate the analysis type
4. Update this README if adding new evaluation types

## Naming Convention

Recommended naming pattern: `[analysis_type]_[metric]_[details].png`

Examples:
- `stride_f1_correlation.png`
- `window_accuracy_heatmap.png`
- `optimal_parameters_analysis.png`

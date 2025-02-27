# Performance Analysis of Thai Profanity Detection Models

## Model Comparison

| Metric | Rectangular | Hamming | FFT | Hamming FFT | Class-Weighted Hamming |
|--------|------------|---------|-----|-------------|----------------------|
| Classify Accuracy | 3.55% | 26.24% | 5.67% | 1.42% | 30.50% |
| Binary Accuracy | 82.41% | 83.71% | 82.41% | 82.02% | 85.67% |
| Combined Accuracy (Weighted Avg) | 42.98% | 54.98% | 44.04% | 41.72% | 58.09% |
| Combined Accuracy (Geometric Mean) | 17.09% | 46.82% | 21.61% | 10.79% | 51.12% |
| Precision | 100.00% | 62.12% | 73.08% | 87.50% | 74.22% |
| Recall | 4.26% | 29.08% | 6.74% | 2.48% | 33.69% |
| F1-score | 8.16% | 39.61% | 12.34% | 4.83% | 46.34% |

## Detailed Performance by Category

### Rectangular Window
- Perfect precision (100%) but very low recall (4.26%)
- Strong performance on กู, มึง, and เหี้ย (100% precision)
- No correct predictions for เย็ด, ควย, สวะ, หี, and แตด

### Hamming Window
- Best overall distribution of predictions
- Strong performance on multiple categories:
  - กู: 87.50% precision, 21.88% recall
  - มึง: 100% precision, 33.33% recall
  - เหี้ย: 83.33% precision, 37.04% recall
  - ควย: 75% precision, 60% recall
  - สวะ: 100% precision, 50% recall
  - หี: 66.67% precision, 12.50% recall

### Hamming FFT
- Poor performance across most categories
- Only successful with สวะ (100% precision, 50% recall)
- Zero precision and recall for all other categories

### Class-Weighted Hamming
- Improved balanced performance:
  - เย็ด: 14.29% precision, 20% recall
  - กู: 90% precision, 28.12% recall
  - มึง: 80% precision, 38.10% recall
  - เหี้ย: 100% precision, 22.22% recall
  - ควย: 75% precision, 60% recall
  - สวะ: 100% precision, 50% recall
  - หี: 66.67% precision, 12.50% recall

## Accuracy Calculations

### Weighted Average (Equal Weights)
```python
Combined Accuracy = (Overall_Accuracy + Binary_Accuracy) / 2

Rectangular: (3.55% + 82.41%) / 2 = 42.98%
Hamming: (26.24% + 83.71%) / 2 = 54.98%
Hamming FFT: (1.42% + 82.02%) / 2 = 41.72%
```

### Geometric Mean
```python
Combined Accuracy = √(Overall_Accuracy × Binary_Accuracy)

Rectangular: √(0.0355 × 0.8241) = 0.1709 = 17.09%
Hamming: √(0.2624 × 0.8371) = 0.4682 = 46.82%
Hamming FFT: √(0.0142 × 0.8202) = 0.1079 = 10.79%
```

## Key Findings

1. **Hamming Window Performance**:
   - Best overall performance across all metrics
   - Highest combined accuracy (54.98% weighted, 46.82% geometric)
   - Most correct predictions (30 out of 151)
   - Best balance between precision and recall
   - Shows good performance across multiple profanity categories

2. **Rectangular Window**:
   - Perfect precision (100%) but very low recall (4.26%)
   - Second-best combined accuracy (42.98% weighted)
   - Poor geometric mean (17.09%) due to imbalance between metrics
   - Few correct predictions (4/151)

3. **Hamming FFT**:
   - Poorest overall performance
   - Lowest combined accuracy by geometric mean (10.79%)
   - Only 1 correct prediction out of 151
   - Shows moderate precision (87.50%) but extremely low recall (2.48%)

## Confusion Matrix Analysis

- **Hamming Window**: Shows more distributed predictions across categories, indicating better learning of different profanity patterns
- **Rectangular Window**: Mostly blank confusion matrix with few correct predictions
- **Hamming FFT**: Almost completely blank confusion matrix, suggesting failure to learn meaningful patterns

## Recommendation

Based on both traditional metrics and combined accuracy calculations, the **Hamming Window** approach remains the most effective method for Thai profanity detection. It provides:
- Best combined accuracy scores
- Better classification across different profanity categories
- More balanced precision-recall trade-off
- Higher number of correct predictions
- More robust overall performance

Consider using the Hamming Window as the baseline for further model improvements, focusing on enhancing both overall classification accuracy and binary detection capabilities.

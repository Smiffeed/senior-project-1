# Hyperparameter Optimization Tools Summary

## 📁 Files Created

### 🚀 Quick Start (Recommended for beginners)
1. **`quick_hyperparameter_finder.py`** - Fastest way to find good hyperparameters
   - Runtime: 30 minutes to 2 hours
   - Tests 13 most important combinations
   - Perfect for getting started

### 🔧 Simple Search Tools
2. **`simple_hyperparameter_search.py`** - Easy-to-use search with multiple modes
   - Quick mode: ~20 trials
   - Comprehensive mode: ~50 trials
   - Customizable search space

### 🧠 Advanced Optimization
3. **`hyperparameter_optimizer.py`** - Full-featured optimization suite
   - Grid Search
   - Random Search  
   - Bayesian Optimization (with Optuna)
   - Progressive Search (multi-stage)

### 📚 Documentation
4. **`HYPERPARAMETER_OPTIMIZATION_GUIDE.md`** - Complete guide with:
   - Hyperparameter explanations
   - Optimization strategies
   - Performance benchmarks
   - Troubleshooting tips

5. **`EVALUATION_GUIDE.md`** - Enhanced evaluation script usage guide

## 🎯 How to Find Optimal Hyperparameters

### Step 1: Quick Start (Do this first!)
```bash
python quick_hyperparameter_finder.py
```
- Takes 30 minutes to 2 hours
- Tests the most important hyperparameter combinations
- Gives you a good baseline to work from

### Step 2: Simple Search (For better results)
```bash
# Quick search (20 trials)
python simple_hyperparameter_search.py --mode quick

# Comprehensive search (50 trials)  
python simple_hyperparameter_search.py --mode comprehensive
```

### Step 3: Advanced Optimization (For best results)
```bash
# Bayesian optimization (smartest approach)
python hyperparameter_optimizer.py --method bayesian --trials 100

# Progressive search (multi-stage refinement)
python hyperparameter_optimizer.py --method progressive
```

## 🏆 Expected Results

### Quick Finder Results:
- **Good**: Accuracy > 85%
- **Excellent**: Accuracy > 90%
- **Time**: 30 minutes - 2 hours

### Comprehensive Search Results:
- **Good**: 5-10% improvement over default
- **Excellent**: 10-20% improvement over default
- **Time**: 4-12 hours

### Advanced Optimization Results:
- **Good**: Near-optimal hyperparameters
- **Excellent**: Maximum possible accuracy
- **Time**: 8-48 hours

## 📊 Key Hyperparameters (Ranked by Impact)

1. **Learning Rate** (HIGHEST IMPACT)
   - Range: 1e-6 to 1e-4
   - Common: 1e-5, 3e-5, 5e-5

2. **Batch Size** (HIGH IMPACT)
   - Range: 4 to 32
   - Depends on GPU memory

3. **Training Epochs** (HIGH IMPACT)
   - Range: 20 to 150
   - Use early stopping

4. **Pooling Mode** (MEDIUM-HIGH IMPACT)
   - Options: 'mean', 'max', 'min'
   - 'max' often best for detection

5. **Weight Decay** (MEDIUM IMPACT)
   - Range: 0.0 to 0.2
   - Controls overfitting

## 💡 Tips for Success

### For RTX 4090/3090 Users:
```python
RECOMMENDED_BATCH_SIZES = [16, 24, 32]
```

### For RTX 4070/3070 Users:
```python
RECOMMENDED_BATCH_SIZES = [8, 16, 24]
```

### For RTX 4060/3060 Users:
```python
RECOMMENDED_BATCH_SIZES = [4, 8, 16]
```

### Good Starting Point:
```python
GOOD_DEFAULTS = {
    'learning_rate': 3e-5,
    'batch_size': 16,  # Adjust for your GPU
    'epochs': 60,
    'pooling_mode': 'mean',  # Try 'max' if 'mean' doesn't work well
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
}
```

## 🚨 Common Issues and Solutions

### Issue: "Out of Memory"
**Solution**: Reduce batch_size from 16 to 8 or 4

### Issue: "Very Low Accuracy (<70%)"
**Solutions**:
- Increase learning_rate (try 5e-5 or 8e-5)
- Check data quality
- Try 'max' pooling instead of 'mean'

### Issue: "Training Takes Too Long"
**Solutions**:
- Increase learning_rate
- Reduce warmup_ratio
- Use fp16=True

### Issue: "Overfitting (Train >> Val Accuracy)"
**Solutions**:
- Increase weight_decay (0.01 → 0.05)
- Reduce learning_rate
- Reduce epochs

## 🎯 Recommended Workflow

### Day 1: Quick Discovery
1. Run `quick_hyperparameter_finder.py`
2. Identify 2-3 best combinations
3. Note which hyperparameters work best

### Day 2: Refinement
1. Run `simple_hyperparameter_search.py --mode comprehensive`
2. Focus on variations around Day 1 results
3. Test different pooling modes and batch sizes

### Day 3+: Final Optimization
1. Use Bayesian optimization for final tuning
2. Include all relevant hyperparameters
3. Run longer searches for production models

## 📈 Performance Expectations

### Profanity Detection Accuracy Targets:
- **Minimum Acceptable**: 80%
- **Good Performance**: 85%+
- **Excellent Performance**: 90%+
- **Outstanding Performance**: 95%+

### Class-Specific Expectations:
- **Long words** (เย็ด, เหี้ย, ควย): 85-95%
- **Short words** (กู, มึง): 70-85% (harder to detect)
- **Background/None**: 90%+ (should be easy)

Remember: **Good hyperparameters can improve your model accuracy by 10-20%**! 

It's worth spending time on optimization. Start with `quick_hyperparameter_finder.py` and work your way up to more advanced methods as needed.

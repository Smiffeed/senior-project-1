# Hyperparameter Optimization Guide

## Overview
This guide provides three different approaches to find the best hyperparameters for your Wav2Vec2 model:

1. **Manual Testing** - Test specific configurations manually
2. **Simple Grid Search** - Automated search over predefined parameter ranges  
3. **Advanced Optuna Search** - Intelligent Bayesian optimization

## Quick Start

### Option 1: Manual Testing (Recommended for beginners)
```bash
python manual_hyperparameter_test.py
```

This will test 4 predefined configurations and show you which works best. The configurations include:
- Conservative setup (good starting point)
- Higher learning rate with strong focal loss
- Small batch with linear scheduler  
- Baseline without focal loss

**Pros:** Easy to understand, modify configurations easily
**Cons:** Limited search space, manual process

### Option 2: Simple Grid Search
```bash
python simple_grid_search.py
```

This will automatically test 15 different combinations of hyperparameters.

**Pros:** Automated, covers more combinations
**Cons:** Can take longer, may miss optimal combinations

### Option 3: Advanced Optuna Search
```bash
pip install optuna
python hyperparameter_search.py
```

This uses intelligent Bayesian optimization to find the best hyperparameters.

**Pros:** Most thorough, intelligent search strategy
**Cons:** Requires additional library, takes longest time

## Key Hyperparameters to Optimize

### Most Important:
1. **learning_rate** (1e-6 to 1e-4)
   - Controls how fast the model learns
   - Too high = unstable training
   - Too low = slow convergence

2. **batch_size** (4, 6, 8, 12)
   - Limited by GPU memory
   - Smaller = more stable gradients
   - Larger = faster training

3. **focal_loss parameters**
   - **gamma** (1-5): Focus on hard examples
   - **alpha** (0.25-0.95): Balance classes

### Secondary:
4. **num_epochs** (50-200)
5. **weight_decay** (1e-5 to 1e-2)
6. **warmup_ratio** (0.0-0.3)
7. **gradient_accumulation_steps** (1-4)

## Understanding Results

### Key Metrics:
- **F1 Macro**: Equal weight to all classes (most important for imbalanced data)
- **Accuracy**: Overall correctness
- **Profanity F1**: Performance on profanity classes specifically

### What to Look For:
- High F1 Macro score (>0.7 is good, >0.8 is excellent)
- Stable training (no sudden drops in validation score)
- Good profanity detection (Profanity F1 > 0.6)

## Interpreting Your Current Cross-Validation

Looking at your current code, there are several issues:

### Problems:
1. **No separate test set** - You're using all data for cross-validation
2. **No final evaluation** - No independent test to measure true performance  
3. **Manual process** - You train folds manually without automated evaluation

### Fixed Implementation:
Your corrected cross-validation now:
1. ✅ Splits data: 60% train, 20% validation, 20% test
2. ✅ Uses train+val for k-fold cross-validation  
3. ✅ Keeps test set completely separate
4. ✅ Automatically finds best fold
5. ✅ Evaluates best model on test set

## Recommended Workflow

### Step 1: Quick Test (30 minutes)
```bash
python manual_hyperparameter_test.py
```
This gives you a baseline and shows which approach works best.

### Step 2: Fine-tune (2-3 hours)
Take the best configuration from Step 1 and modify it:
- Try learning rates around the best value (±50%)
- Test different focal loss parameters
- Experiment with batch sizes

### Step 3: Full Search (Optional, 4-8 hours)
```bash
python hyperparameter_search.py
```
For the most thorough optimization.

## Expected Performance

### Baseline (no optimization):
- F1 Macro: 0.2-0.4
- Accuracy: 0.5-0.7

### After hyperparameter optimization:
- F1 Macro: 0.6-0.8
- Accuracy: 0.7-0.9
- Profanity F1: 0.5-0.8

## GPU Memory Management

If you get CUDA out of memory errors:
1. Reduce `batch_size` to 4 or 6
2. Increase `gradient_accumulation_steps` to 4 or 6
3. Set `dataloader_num_workers=0`

## Next Steps After Finding Best Parameters

1. **Train final model** with best hyperparameters on full dataset
2. **Evaluate thoroughly** on completely separate test data
3. **Save the model** for deployment
4. **Document the configuration** for reproducibility

## Files Created

- `manual_hyperparameter_test.py` - Manual testing script
- `simple_grid_search.py` - Automated grid search  
- `hyperparameter_search.py` - Advanced Optuna optimization
- Results saved in `./models/[method_name]/` directories

## Getting Help

If you encounter issues:
1. Check GPU memory usage
2. Reduce batch size if getting CUDA errors
3. Check that all file paths are correct
4. Ensure you have the latest versions of transformers and torch

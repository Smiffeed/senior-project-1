# Hyperparameter Optimization Guide for Wav2Vec2 Profanity Detection

## Overview

Hyperparameter optimization is crucial for achieving the best performance from your Wav2Vec2 model. This guide covers the most important hyperparameters and provides practical strategies for finding optimal values.

## 🎯 Most Important Hyperparameters (Ranked by Impact)

### 1. Learning Rate (HIGHEST IMPACT)
- **Range**: 1e-6 to 1e-4
- **Common values**: 1e-5, 3e-5, 5e-5
- **Effect**: Controls how fast the model learns
- **Too high**: Model diverges or oscillates
- **Too low**: Very slow learning, may not converge

### 2. Batch Size (HIGH IMPACT)
- **Range**: 4 to 32 (limited by GPU memory)
- **Common values**: 8, 16, 24
- **Effect**: Affects gradient stability and training speed
- **Larger**: More stable gradients, but needs more memory
- **Smaller**: Less memory, but noisier gradients

### 3. Number of Training Epochs (HIGH IMPACT)
- **Range**: 20 to 150
- **Common values**: 50, 80, 100
- **Effect**: How long to train
- **Too few**: Underfitting
- **Too many**: Overfitting

### 4. Pooling Mode (MEDIUM-HIGH IMPACT)
- **Options**: 'mean', 'max', 'min'
- **Effect**: How to aggregate sequence features
- **Mean**: Average representation (most common)
- **Max**: Strongest features (good for detection tasks)
- **Min**: Weakest features (rarely used)

### 5. Weight Decay (MEDIUM IMPACT)
- **Range**: 0.0 to 0.2
- **Common values**: 0.01, 0.05, 0.1
- **Effect**: Regularization to prevent overfitting
- **Higher**: More regularization, may underfit
- **Lower**: Less regularization, may overfit

### 6. Warmup Ratio (MEDIUM IMPACT)
- **Range**: 0.0 to 0.3
- **Common values**: 0.1, 0.15, 0.2
- **Effect**: Gradual learning rate increase at start
- **Helps**: Stable training start
- **0.1 = 10%**: First 10% of training uses lower learning rate

### 7. Gradient Accumulation Steps (LOW-MEDIUM IMPACT)
- **Options**: 1, 2, 4, 8
- **Effect**: Simulates larger batch sizes
- **Use when**: GPU memory is limited
- **Example**: batch_size=8, accumulation=4 → effective_batch_size=32

## 🔍 Optimization Strategies

### Strategy 1: Quick Search (30 minutes - 2 hours)
```bash
python simple_hyperparameter_search.py --mode quick --max_trials 20
```
- Tests 3-4 most important hyperparameters
- ~20 trials total
- Good for initial exploration

### Strategy 2: Comprehensive Search (4-12 hours)
```bash
python simple_hyperparameter_search.py --mode comprehensive --max_trials 50
```
- Tests 6-7 hyperparameters
- ~50 trials total
- Better optimization

### Strategy 3: Advanced Bayesian Search (8-24 hours)
```bash
python hyperparameter_optimizer.py --method bayesian --trials 100
```
- Uses Optuna for smart search
- Learns from previous trials
- Most efficient for large search spaces

### Strategy 4: Progressive Search (12-48 hours)
```bash
python hyperparameter_optimizer.py --method progressive
```
- Multi-stage optimization
- Coarse → Fine → Ultra-fine
- Best for final optimization

## 📊 How to Interpret Results

### Good Hyperparameter Signs:
- **Accuracy > 0.85**: Generally good for profanity detection
- **Stable training**: Loss decreases smoothly
- **No overfitting**: Validation accuracy close to training accuracy
- **Fast convergence**: Reaches good accuracy quickly

### Bad Hyperparameter Signs:
- **Accuracy < 0.7**: Likely underfitting or poor hyperparameters
- **Unstable training**: Loss jumps around
- **Overfitting**: Training accuracy >> Validation accuracy
- **Slow convergence**: Takes too long to improve

## 🛠️ Practical Tips

### Starting Points (Good Defaults):
```python
GOOD_STARTING_HYPERPARAMS = {
    'learning_rate': 3e-5,
    'batch_size': 16,
    'epochs': 60,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'pooling_mode': 'mean',
    'gradient_accumulation_steps': 1,
}
```

### GPU Memory Optimization:
- **RTX 4090/3090**: batch_size=24-32
- **RTX 4080/3080**: batch_size=16-24  
- **RTX 4070/3070**: batch_size=8-16
- **RTX 4060/3060**: batch_size=4-8

### Learning Rate Guidelines:
- **Large dataset (>5000 samples)**: Start with 1e-5
- **Medium dataset (1000-5000 samples)**: Start with 3e-5
- **Small dataset (<1000 samples)**: Start with 5e-5

### Epoch Guidelines:
- **Strong baseline model**: 30-60 epochs
- **Fine-tuning from scratch**: 80-120 epochs
- **Small dataset**: 100-150 epochs (with early stopping)

## 🔬 Advanced Techniques

### 1. Learning Rate Scheduling
```python
# In training arguments:
'lr_scheduler_type': 'cosine',  # or 'linear', 'polynomial'
'warmup_steps': 1000,           # instead of warmup_ratio
```

### 2. Mixed Precision Training
```python
'fp16': True,  # Faster training, less memory
'bf16': False, # Alternative to fp16 (newer GPUs)
```

### 3. Gradient Clipping
```python
'max_grad_norm': 1.0,  # Prevents gradient explosion
```

### 4. Data Augmentation Parameters
```python
# In your preprocessing:
'augmentation_probability': 0.3,  # 30% of samples get augmented
'noise_level': 0.005,            # Background noise strength
'pitch_shift_range': (-2, 2),    # Pitch variation in semitones
```

## 📈 Optimization Workflow

### Phase 1: Baseline (Day 1)
1. Use default hyperparameters
2. Train one model to get baseline accuracy
3. Identify if you have overfitting/underfitting

### Phase 2: Quick Search (Day 1-2)
1. Run quick hyperparameter search
2. Focus on learning_rate and batch_size
3. Find 2-3 promising combinations

### Phase 3: Refinement (Day 2-3)
1. Take best combinations from Phase 2
2. Test variations around them
3. Add more hyperparameters (weight_decay, warmup_ratio)

### Phase 4: Final Optimization (Day 3-5)
1. Use Bayesian optimization or progressive search
2. Include all relevant hyperparameters
3. Run longer searches for final tuning

## 🎯 Expected Performance Levels

### Accuracy Benchmarks:
- **Excellent**: >90% accuracy
- **Good**: 85-90% accuracy
- **Acceptable**: 80-85% accuracy
- **Poor**: <80% accuracy

### Class-Specific Performance:
- **Long words (เย็ด, เหี้ย, ควย)**: Usually 85-95% accuracy
- **Short words (กู, มึง)**: Often 70-85% accuracy (harder)
- **Background/None class**: Should be >90% accuracy

## 🚨 Common Problems and Solutions

### Problem: Low Accuracy (<70%)
**Solutions:**
- Increase learning rate (try 5e-5 or 1e-4)
- Increase number of epochs
- Reduce weight_decay
- Check data quality

### Problem: Overfitting (Train >> Val accuracy)
**Solutions:**
- Increase weight_decay (0.05 → 0.1)
- Reduce learning rate
- Add more dropout
- Reduce number of epochs

### Problem: Unstable Training (Loss jumps)
**Solutions:**
- Reduce learning rate
- Increase batch size or gradient accumulation
- Add gradient clipping (max_grad_norm=1.0)
- Increase warmup_ratio

### Problem: Very Slow Training
**Solutions:**
- Increase learning rate
- Reduce warmup_ratio
- Use mixed precision (fp16=True)
- Increase batch size

### Problem: Out of Memory
**Solutions:**
- Reduce batch size
- Increase gradient_accumulation_steps
- Use gradient_checkpointing=True
- Reduce sequence length

## 📝 Hyperparameter Search Log Template

Keep track of your experiments:

```
Date: 2025-01-XX
Method: Quick Search
Trials: 20
Best Result:
  - Accuracy: 0.XXXX
  - Learning Rate: X.XeX
  - Batch Size: XX
  - Epochs: XX
  - Notes: [What worked well, what didn't]

Next Steps:
  - [ ] Try higher learning rate
  - [ ] Test different pooling modes
  - [ ] Run comprehensive search
```

## 🎓 Final Recommendations

### For Beginners:
1. Start with `simple_hyperparameter_search.py --mode quick`
2. Focus on learning_rate and batch_size first
3. Use the best result as baseline for further optimization

### For Experienced Users:
1. Use Bayesian optimization with Optuna
2. Include all relevant hyperparameters
3. Use progressive search for final models

### For Production Models:
1. Run multiple optimization rounds
2. Use cross-validation for robust results
3. Test final model on completely separate test set

Remember: **Good hyperparameters can improve accuracy by 10-20%**, making optimization time well spent!

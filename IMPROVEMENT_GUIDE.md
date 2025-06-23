# Improving Model Accuracy and Project Quality

This guide outlines steps to implement the improvements for your Thai profanity detection model. Follow these instructions to boost model accuracy and overall project quality.

## Overview of Improvements

1. **Data Augmentation**: Balance your dataset and make the model more robust to variations
2. **Enhanced Model Architecture**: Attention mechanism and optimized architecture for better learning
3. **Advanced Training Techniques**: Staged training with better hyperparameters
4. **Audio Preprocessing**: Better noise reduction for cleaner audio input
5. **Hyperparameter Optimization**: Systematic approach to finding optimal parameters
6. **Comprehensive Evaluation**: More detailed analysis of model performance

## Step 1: Install New Dependencies

First, ensure all required dependencies are installed:

```bash
pip install -r requirements.txt
```

Key new dependencies:
- `audiomentations`: For audio data augmentation
- `optuna`: For hyperparameter optimization
- `wandb`: For experiment tracking (optional)

## Step 2: Data Augmentation

Run the data augmentation script to balance your dataset:

```bash
python scripts/data_augmentation.py
```

This will:
- Analyze class distribution in your dataset
- Generate augmented samples for underrepresented classes
- Save a balanced dataset as `csv/balanced_main.csv`

## Step 3: Audio Preprocessing

Clean your audio files to remove background noise:

```bash
python audio_processing/noise_reduction.py
```

This creates clean versions of your audio files in `./main_clean` directory.

## Step 4: Model Training

Train the improved model with enhanced architecture and training techniques:

```bash
python scripts/improved_training.py
```

This script:
- Uses the balanced dataset
- Applies the improved model architecture with attention mechanism
- Implements staged training (feature extractor first, then fine-tuning)
- Uses better regularization and optimization techniques

## Step 5: Hyperparameter Optimization (Optional)

For best results, run hyperparameter optimization:

```bash
python scripts/hyperparameter_optimization.py
```

This may take several hours, but will find optimal:
- Learning rate
- Weight decay
- Dropout rate
- Batch size
- Warmup ratio

## Step 6: Comprehensive Evaluation

Run the advanced evaluation script to get detailed performance metrics:

```bash
python scripts/advanced_evaluation.py
```

This will generate:
- Detailed classification report
- Confusion matrix visualization
- Per-class performance metrics
- Confidence distribution analysis

## Explanation of Improvements

### 1. Data Augmentation

The original dataset had class imbalance issues with some labels like 'หี' having low recall. The data augmentation applies:
- Time stretching (speed variation)
- Pitch shifting
- Adding noise
- Gain adjustments
- Room simulation

This creates more diverse training examples and helps the model generalize better for underrepresented classes.

### 2. Enhanced Model Architecture

The new architecture includes:
- **Attention mechanism**: Helps focus on relevant parts of audio
- **Better feature aggregation**: Improved pooling of features
- **Deeper classification head**: More expressive model
- **Advanced regularization**: Multiple dropout layers to prevent overfitting

### 3. Advanced Training Techniques

Improvements include:
- **Staged training**: First freeze feature extractor, then fine-tune entire model
- **Better weighted loss**: Using improved class weights calculation
- **Mixup augmentation**: Combining samples during training for better robustness
- **Learning rate scheduling**: Proper warmup and decay
- **Gradient accumulation**: Allows effective larger batch sizes

### 4. Audio Preprocessing

The noise reduction applies:
- Spectral gating to remove background noise
- Adaptive noise profile estimation
- Peak normalization for better volume consistency

### 5. Evaluation

New evaluation metrics provide:
- Per-class performance analysis
- Visualization of confusion patterns
- Confidence score analysis
- Windowed evaluation for long audio files

## Expected Results

With these improvements, you should see:
1. Better overall accuracy (target: 15-25% improvement)
2. Higher recall for underrepresented classes
3. More stable training
4. Better generalization to new audio samples
5. More robust performance in noisy environments

## Monitoring Progress

The new training scripts include metrics logging that will show you:
- Per-epoch accuracy, F1, precision, and recall
- Per-class performance over time
- Learning curves to identify overfitting

Good luck with the implementation!

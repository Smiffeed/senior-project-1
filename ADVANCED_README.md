# Advanced Thai Profanity Detection Model 🚀

This repository contains state-of-the-art improvements for Thai profanity audio detection using advanced deep learning techniques.

## 🎯 Key Improvements Over Baseline

### 1. **Advanced Audio Preprocessing**
- **Spectral Gating Denoising**: Advanced noise reduction using spectral gating
- **Dynamic Range Compression**: Professional audio compression for consistent levels
- **Advanced Voice Activity Detection (VAD)**: Multi-feature VAD using energy, ZCR, and spectral centroid
- **Enhanced Pre-emphasis**: Improved spectral characteristics
- **Hamming Windowing**: Better frequency domain representation

### 2. **Multi-Modal Feature Extraction**
- **Mel-spectrograms**: Time-frequency representation
- **MFCC Features**: Cepstral coefficients for speech characteristics
- **Spectral Features**: Centroid, rolloff, bandwidth, zero-crossing rate
- **Chroma Features**: Harmonic content analysis
- **Temporal Features**: RMS energy and statistical moments
- **Feature Statistics**: Mean, std, max, min across time dimensions

### 3. **Advanced Data Augmentation**
- **SpecAugment**: Time and frequency masking in spectrograms
- **MixUp**: Blending spectrograms for better generalization
- **Audio Augmentation**: Noise injection, pitch shifting, time stretching
- **Adaptive Augmentation**: Context-aware augmentation strategies

### 4. **State-of-the-Art Model Architecture**
- **Multi-Scale Feature Fusion**: Combining global, local, and mid-range features
- **Transformer Encoder**: Multi-head self-attention for sequence modeling
- **Positional Encoding**: Better sequence understanding
- **Attention Pooling**: Weighted feature aggregation
- **Uncertainty Estimation**: Model confidence prediction

### 5. **Advanced Training Techniques**
- **Focal Loss**: Better handling of class imbalance
- **Contrastive Learning**: Learning better feature representations
- **Adversarial Training**: Improved model robustness
- **Progressive Training**: Gradual complexity increase
- **Curriculum Learning**: Easy-to-hard sample ordering
- **Knowledge Distillation**: Teacher-student learning (optional)

### 6. **Ensemble Methods**
- **Multi-Model Ensemble**: Combining predictions from multiple models
- **Uncertainty-Weighted Voting**: Using model confidence for ensemble weighting
- **Cross-Validation Ensemble**: Using models from different folds

### 7. **Advanced Evaluation**
- **Uncertainty Quantification**: Model confidence analysis
- **Calibration Analysis**: Prediction reliability assessment
- **Per-Class Analysis**: Detailed metrics for each profanity type
- **Error Analysis**: Understanding model failure modes
- **ROC/AUC Analysis**: Comprehensive performance metrics

## 📁 File Structure

```
senior-project-1/
├── scripts/
│   ├── advanced_model_improvements.py      # Core advanced techniques
│   ├── ultimate_model_training.py          # Main training pipeline
│   ├── comprehensive_evaluation.py         # Advanced evaluation suite
│   ├── improved_model_training.py          # Previous iteration
│   └── evaluate_windowed.py               # Basic evaluation
├── config/
│   └── advanced_training_config.json      # Configuration file
├── run_experiments.py                     # Experiment runner
├── csv/
│   └── main.csv                           # Dataset
├── models/                                # Trained models
├── experiments/                           # Experiment results
└── README.md                             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchaudio transformers
pip install librosa scikit-learn pandas numpy matplotlib seaborn
pip install scipy tqdm
```

### 2. Basic Training

```bash
# Run with default advanced configuration
python scripts/ultimate_model_training.py
```

### 3. Advanced Experiment Management

```bash
# Run ablation study to understand component contributions
python run_experiments.py --action ablation

# Run hyperparameter search
python run_experiments.py --action hyperparameter

# Run custom experiment
python run_experiments.py --action custom --name my_experiment --modifications '{"advanced_features.use_ensemble": false}'

# List all experiments
python run_experiments.py --action list

# Compare experiments
python run_experiments.py --action compare --experiments ablation_baseline ablation_full_advanced
```

### 4. Evaluation Only

```bash
# Evaluate trained models
python scripts/comprehensive_evaluation.py
```

## ⚙️ Configuration

The system is highly configurable through `config/advanced_training_config.json`. Key sections:

### Advanced Features
```json
{
  "advanced_features": {
    "use_spectral_augmentation": true,
    "use_contrastive_learning": true,
    "use_adversarial_training": true,
    "use_ensemble": true,
    "use_progressive_training": true,
    "use_focal_loss": true,
    "use_uncertainty_estimation": true
  }
}
```

### Training Configuration
```json
{
  "training_config": {
    "num_folds": 5,
    "batch_size": 8,
    "learning_rate": 3e-5,
    "num_epochs": 30,
    "early_stopping_patience": 5
  }
}
```

### Loss Weights
```json
{
  "loss_weights": {
    "contrastive_weight": 0.1,
    "adversarial_weight": 0.2,
    "focal_loss_gamma": 2.0
  }
}
```

## 🧪 Experiment Types

### 1. Ablation Study
Systematically removes features to understand their contribution:
- `baseline`: No advanced features
- `with_spectral_aug`: + Spectral augmentation
- `with_contrastive`: + Contrastive learning
- `with_adversarial`: + Adversarial training
- `with_focal_loss`: + Focal loss
- `full_advanced`: All features enabled

### 2. Hyperparameter Search
Tests different hyperparameter combinations:
- Learning rates: [1e-5, 3e-5, 5e-5]
- Contrastive weights: [0.05, 0.1, 0.2]
- Focal loss gamma: [1.0, 2.0, 3.0]

### 3. Custom Experiments
Create your own experiments with specific configurations.

## 📊 Expected Performance Improvements

Based on state-of-the-art techniques, you can expect:

1. **+5-10% Accuracy**: From advanced preprocessing and feature extraction
2. **+3-7% F1-Score**: From better class imbalance handling (Focal Loss)
3. **+2-5% Robustness**: From adversarial training and augmentation
4. **+3-8% Overall**: From ensemble methods
5. **Better Calibration**: More reliable confidence estimates

### Profanity Detection Specific Improvements:
- **Rare Class Detection**: Better performance on low-frequency profanity types
- **Context Understanding**: Improved detection in noisy environments
- **Confidence Estimation**: Know when the model is uncertain

## 🔧 Key Components Explained

### 1. MultiModalAudioClassifier
Combines Wav2Vec2 with additional audio features:
```python
# Combines multiple feature types
- Wav2Vec2 hidden states
- Spectral features (MFCC, mel-spec, etc.)
- Transformer-based classification
- Uncertainty estimation
```

### 2. Advanced Training Strategy
Multiple loss functions and training techniques:
```python
# Combined loss function
total_loss = classification_loss + 
             contrastive_weight * contrastive_loss +
             adversarial_weight * adversarial_loss
```

### 3. Progressive Training
Gradually increases model complexity:
```python
stages = [
    {'name': 'basic', 'epochs': 10, 'complexity': 'low'},
    {'name': 'intermediate', 'epochs': 15, 'complexity': 'medium'},
    {'name': 'advanced', 'epochs': 20, 'complexity': 'high'}
]
```

## 📈 Monitoring and Analysis

### 1. Training Metrics
- Loss components (classification, contrastive, adversarial)
- Learning rate scheduling
- Gradient norms
- Memory usage

### 2. Evaluation Metrics
- Overall accuracy and per-class performance
- Confusion matrices and error analysis
- Uncertainty calibration curves
- ROC curves and AUC scores

### 3. Visualization
- Training curves
- Feature importance plots
- Attention weight visualizations
- Error pattern analysis

## 🎯 Best Practices

### 1. Data Preparation
- Ensure balanced classes through oversampling
- Use stratified splits for consistent evaluation
- Validate audio file integrity

### 2. Training
- Start with baseline configuration
- Gradually enable advanced features
- Monitor for overfitting with early stopping
- Use cross-validation for robust evaluation

### 3. Evaluation
- Always use held-out test set
- Analyze per-class performance
- Consider uncertainty in predictions
- Compare multiple configurations

## 🚨 Troubleshooting

### Common Issues:

1. **Memory Errors**
   - Reduce batch size
   - Disable some advanced features
   - Use gradient accumulation

2. **Training Instability**
   - Lower learning rate
   - Reduce adversarial training weight
   - Enable gradient clipping

3. **Poor Performance**
   - Check data quality and labels
   - Verify preprocessing pipeline
   - Start with simpler configuration

4. **Slow Training**
   - Reduce model complexity
   - Use fewer augmentations
   - Disable ensemble training

## 📚 Research References

The implementations are based on cutting-edge research:

1. **SpecAugment**: Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
2. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection"
3. **Contrastive Learning**: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations"
4. **Adversarial Training**: Goodfellow et al., "Explaining and Harnessing Adversarial Examples"
5. **Curriculum Learning**: Bengio et al., "Curriculum Learning"

## 🤝 Contributing

To add new advanced techniques:

1. Implement in `advanced_model_improvements.py`
2. Integrate into `ultimate_model_training.py`
3. Add configuration options
4. Update evaluation metrics
5. Add experiments and documentation

## 📄 License

This project is part of a university senior project focused on advancing Thai language NLP capabilities.

---

**Happy experimenting! 🎉**

For questions or issues, please refer to the troubleshooting section or create detailed error reports with your configuration and logs.

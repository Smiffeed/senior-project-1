# Advanced Frame-Level Audio Evaluation System

This advanced evaluation system combines the best aspects of multiple evaluation approaches to provide comprehensive frame-level profanity detection analysis.

## 🎯 Key Features

### **Frame-Level Detection** (from `frame_level_censor.py`)
- Sliding window approach for continuous audio analysis
- Ensemble model support for improved accuracy
- Configurable window sizes and strides
- Confidence-based filtering

### **Advanced Audio Preprocessing** (from `comprehensive_evaluation_processor.py`)  
- Noise reduction with preemphasis filtering
- Audio normalization and length validation
- Robust error handling for audio loading
- Consistent feature extraction

### **Parallel Processing** (from `fixed_smart_parallel_processor.py`)
- ThreadPoolExecutor for efficient parallel evaluation
- System resource optimization
- Memory-conscious GPU management
- Progress tracking and error reporting

### **VAD Refinement** (from `vad_post_processor.py`)
- Voice Activity Detection for precise boundaries
- Overlapping prediction merging
- Speech segment refinement
- IoU-based evaluation metrics

## 📁 Files Overview

### Core Components
- **`advanced_frame_level_evaluator.py`** - Main evaluation system
- **`batch_advanced_evaluator.py`** - Batch processing with multiple configurations
- **`test_advanced_evaluator.py`** - Test suite with different scenarios

### Key Classes
- **`AdvancedAudioPreprocessor`** - Audio preprocessing pipeline
- **`EnhancedModelEnsemble`** - Model loading and ensemble predictions
- **`VADRefinementProcessor`** - VAD-based boundary refinement
- **`AdvancedFrameLevelEvaluator`** - Main evaluation orchestrator

## 🚀 Quick Start

### 1. Basic Evaluation
```bash
python advanced_frame_level_evaluator.py \
    --model_dir models/4_classes_max_steps \
    --ground_truth csv/eval_5labels.csv \
    --audio_dir dataset \
    --output_dir results/basic_evaluation
```

### 2. High Precision Evaluation
```bash
python advanced_frame_level_evaluator.py \
    --model_dir models/4_classes_max_steps \
    --ground_truth csv/eval_5labels.csv \
    --audio_dir dataset \
    --output_dir results/high_precision \
    --window_size 0.3 \
    --stride 0.15 \
    --confidence_threshold 0.7
```

### 3. Fast Evaluation
```bash
python advanced_frame_level_evaluator.py \
    --model_dir models/4_classes_max_steps \
    --ground_truth csv/eval_5labels.csv \
    --audio_dir dataset \
    --output_dir results/fast_evaluation \
    --window_size 1.0 \
    --stride 0.8 \
    --workers 4
```

## 🔧 Configuration Parameters

### Window Configuration
- **`--window_size`** (float): Window size in seconds (default: 0.5)
- **`--stride`** (float): Stride between windows in seconds (default: 0.25)

### Model Configuration  
- **`--model_dir`** (str): Directory containing trained models
- **`--model_name`** (str): Base HuggingFace model name
- **`--confidence_threshold`** (float): Minimum confidence for detection (default: 0.5)

### Processing Configuration
- **`--workers`** (int): Number of parallel workers (auto-detected)
- **`--max_files`** (int): Limit number of files for testing
- **`--file_pattern`** (str): Audio file pattern (default: "*.wav")

## 📊 Batch Evaluation

### Run All Configurations
```bash
python batch_advanced_evaluator.py \
    --model_dir models/4_classes_max_steps \
    --ground_truth csv/eval_5labels.csv \
    --audio_dir dataset \
    --output_dir batch_results
```

### List Available Configurations
```bash
python batch_advanced_evaluator.py --list_configs
```

### Run Specific Configurations
```bash
python batch_advanced_evaluator.py \
    --configs balanced_small_window high_precision_medium_window \
    --max_files 10
```

### Available Batch Configurations

| Configuration | Window | Stride | Confidence | Description |
|---------------|--------|--------|------------|-------------|
| `high_precision_small_window` | 0.3s | 0.15s | 0.7 | High precision, small window |
| `high_precision_medium_window` | 0.5s | 0.25s | 0.7 | High precision, medium window |
| `balanced_small_window` | 0.3s | 0.15s | 0.5 | Balanced precision, small window |
| `balanced_medium_window` | 0.5s | 0.25s | 0.5 | Balanced precision, medium window |
| `balanced_large_window` | 1.0s | 0.5s | 0.5 | Balanced precision, large window |
| `high_coverage_small_window` | 0.3s | 0.15s | 0.3 | High coverage, small window |
| `high_coverage_medium_window` | 0.5s | 0.25s | 0.3 | High coverage, medium window |
| `fast_large_window` | 1.0s | 0.8s | 0.5 | Fast evaluation, large window |
| `fast_very_large_window` | 2.0s | 1.5s | 0.5 | Very fast evaluation |

## 🧪 Testing

### Run Test Suite
```bash
# Run all tests
python test_advanced_evaluator.py --test all

# Run specific test
python test_advanced_evaluator.py --test precision

# Analyze existing results only
python test_advanced_evaluator.py --analyze_only
```

### Test Scenarios
- **Basic Test**: Default parameters with 5 files, 2 workers
- **Precision Test**: High precision settings (0.3s window, 0.7 confidence)
- **Fast Test**: Speed-optimized settings (1.0s window, 4 workers)

## 📈 Output Structure

### Directory Structure
```
output_dir/
├── evaluation_summary.csv      # Per-file results summary
├── evaluation_config.json      # Configuration and metadata
└── detailed_results/           # Detailed prediction files (optional)
```

### Key Metrics
- **Mean IoU**: Average Intersection over Union for all predictions
- **Binary F1**: Binary classification F1 score (profane vs non-profane)
- **Multiclass F1**: Multiclass F1 score for specific profanity types
- **Precision/Recall**: Standard classification metrics
- **Processing Stats**: Timing and system resource usage

### CSV Columns (evaluation_summary.csv)
- `audio_file`: Audio filename
- `raw_windows`: Total sliding windows processed
- `profanity_windows`: High-confidence profanity windows
- `merged_predictions`: After overlapping merge
- `refined_predictions`: After VAD refinement
- `mean_iou`: Mean IoU for this file
- `binary_f1`: Binary F1 score
- `multiclass_f1`: Multiclass F1 score
- `total_ground_truth`: Ground truth entries
- `total_predictions`: Final predictions

## 🔄 Processing Pipeline

### 1. **Audio Loading & Windowing**
- Load audio file and determine duration
- Generate sliding windows with specified size and stride
- Handle edge cases (short files, boundary windows)

### 2. **Frame-Level Prediction**
- Advanced audio preprocessing for each window
- Ensemble model prediction with uncertainty estimation
- Confidence-based filtering of high-confidence profanity

### 3. **Temporal Refinement**
- Merge overlapping predictions of same label
- Apply Voice Activity Detection for precise boundaries
- Split refined segments based on speech activity

### 4. **Evaluation Against Ground Truth**
- IoU calculation between predictions and ground truth
- Binary and multiclass classification metrics
- Per-class IoU analysis

## ⚙️ System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM (16GB recommended)
- CUDA-compatible GPU (optional but recommended)

### Dependencies
```bash
pip install torch torchaudio transformers librosa scikit-learn pandas numpy tqdm psutil
```

### Model Requirements
- Trained Wav2Vec2 model (HuggingFace format or PyTorch checkpoints)
- Feature extractor configuration
- Support for 5-class classification (none, เย็ด, กู, มึง, เหี้ย)

## 🎛️ Advanced Usage

### Custom Model Loading
The system supports multiple model formats:
- **HuggingFace Models**: Complete model directory with config.json
- **Fold-based Ensemble**: Multiple models in fold_1/, fold_2/, etc.
- **PyTorch Checkpoints**: .pth/.pt files with state dictionaries

### Memory Optimization
```bash
# For limited GPU memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use fewer workers
python advanced_frame_level_evaluator.py --workers 1
```

### Custom Audio Preprocessing
The `AdvancedAudioPreprocessor` can be extended:
```python
class CustomPreprocessor(AdvancedAudioPreprocessor):
    def advanced_preprocess_audio(self, file_path, start_time, end_time):
        # Custom preprocessing logic
        audio = super().advanced_preprocess_audio(file_path, start_time, end_time)
        # Additional processing...
        return audio
```

## 🔍 Troubleshooting

### Common Issues

**GPU Memory Errors**
```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""
python advanced_frame_level_evaluator.py --workers 1
```

**Audio Loading Errors**
- Ensure audio files are in supported formats (WAV, MP3, FLAC)
- Check audio file paths in ground truth CSV
- Verify audio file integrity

**Model Loading Errors**
- Verify model directory structure
- Check model compatibility with NUM_LABELS=5
- Ensure feature extractor configuration exists

**Performance Issues**
- Reduce window size or increase stride for faster processing
- Use fewer workers to reduce memory usage
- Enable test mode with `--max_files` for quick testing

### Debug Mode
```bash
# Enable verbose output
python -u advanced_frame_level_evaluator.py [options] 2>&1 | tee debug.log
```

## 📚 Integration with Existing Systems

### With Comprehensive Evaluation
```python
# Use the same preprocessing approach
from advanced_frame_level_evaluator import AdvancedAudioPreprocessor
preprocessor = AdvancedAudioPreprocessor()
```

### With VAD Post-processor
```python
# Use the same VAD refinement
from advanced_frame_level_evaluator import VADRefinementProcessor
vad_processor = VADRefinementProcessor()
```

### With Parallel Processing
```python
# Use the same threading approach
from concurrent.futures import ThreadPoolExecutor
```

## 🎯 Performance Tuning

### For Accuracy
- Use smaller windows (0.3s) with smaller strides (0.15s)
- Higher confidence thresholds (0.7)
- Enable VAD refinement
- Use ensemble models

### For Speed
- Use larger windows (1.0s+) with larger strides (0.8s+)
- Lower confidence thresholds (0.3)
- More parallel workers
- Skip VAD refinement for very fast processing

### For Coverage
- Lower confidence thresholds (0.3)
- Smaller strides for better overlap
- Process all confidence levels

## 📖 Examples

### Example 1: Research Evaluation
```bash
# Comprehensive evaluation for research
python batch_advanced_evaluator.py \
    --model_dir models/best_model \
    --ground_truth datasets/test_set.csv \
    --audio_dir datasets/test_audio \
    --output_dir research_results \
    --workers 4
```

### Example 2: Production Testing
```bash
# Fast evaluation for production testing
python advanced_frame_level_evaluator.py \
    --model_dir models/production_model \
    --ground_truth validation/ground_truth.csv \
    --audio_dir validation/audio \
    --output_dir production_test \
    --window_size 1.0 \
    --stride 0.8 \
    --confidence_threshold 0.5 \
    --max_files 50
```

### Example 3: Custom Configuration
```bash
# Custom configuration for specific needs
python advanced_frame_level_evaluator.py \
    --model_dir models/custom_model \
    --ground_truth data/gt.csv \
    --audio_dir data/audio \
    --output_dir custom_results \
    --window_size 0.4 \
    --stride 0.2 \
    --confidence_threshold 0.6 \
    --workers 2 \
    --file_pattern "*.flac"
```

---

## 🤝 Contributing

This evaluation system is designed to be modular and extensible. Key extension points:

- **Custom Preprocessors**: Extend `AdvancedAudioPreprocessor`
- **Custom Models**: Implement model loading in `EnhancedModelEnsemble`
- **Custom Metrics**: Add evaluation methods in `AdvancedFrameLevelEvaluator`
- **Custom VAD**: Extend `VADRefinementProcessor`

## 📝 Citation

If you use this evaluation system in your research, please cite the original components and methodologies that inspired this implementation.

---

**🎯 This system provides state-of-the-art frame-level audio evaluation combining the best practices from multiple approaches for comprehensive profanity detection analysis.**
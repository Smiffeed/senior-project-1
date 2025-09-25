# 🎯 Thai Profanity Detector Web App

A modern web application for detecting and censoring Thai profanity in audio and video files.

## Features

### 🎵 **File Support**
- **Audio Files**: WAV, MP3, M4A, AAC
- **Video Files**: MP4, AVI, MOV, MKV, FLV (audio extraction)
- **Upload Size**: Up to 200MB per file

### 🔍 **Detection Capabilities**
- Real-time Thai profanity detection
- 4 profanity classes: เย็ด, กู, มึง, เหี้ย
- Confidence scoring for each detection
- Precise timestamp identification

### 📊 **Visualization**
- Interactive timeline showing profanity locations
- Summary charts and statistics
- Detailed detection list with timestamps
- Audio comparison (original vs censored)

### 🔇 **Censoring Options**
- **Silence**: Replace profanity with silence
- **Beep**: Replace with customizable beep tone
- **Fade**: Gradual fade out/in effect

### 📁 **Export Features**
- Download censored audio files
- Export detailed JSON reports
- Processing timestamps and confidence scores

## Installation

### Prerequisites
- Python 3.8+ 
- FFmpeg (for video support)
- CUDA (optional, for GPU acceleration)

### Quick Start
1. **Clone/Download** this repository
2. **Run the launcher**:
   ```bash
   python launch_app.py
   ```
3. The launcher will:
   - Check dependencies
   - Install requirements automatically
   - Launch the web app in your browser

### Manual Installation
If you prefer manual installation:

```bash
# Install requirements
pip install -r requirements_app.txt

# Launch app
streamlit run profanity_detector_app.py
```

## Usage

### 1. **Upload File**
- Click "Choose an audio or video file"
- Select your file (max 200MB)
- Wait for processing

### 2. **Configure Settings** (Optional)
- **Model Selection**: Choose from available trained models
- **Detection Parameters**:
  - Window Size: Analysis window duration
  - Overlap: Window overlap amount
  - Confidence Threshold: Minimum confidence for detection
- **Censoring Method**: Choose silence, beep, or fade

### 3. **Process File**
- Click "Detect & Censor Profanity"
- Wait for analysis (may take 1-5 minutes depending on file size)

### 4. **View Results**
- Interactive timeline showing profanity locations
- Statistics and summary charts
- Detailed list with timestamps and confidence scores

### 5. **Download Results**
- Censored audio file (WAV format)
- Detailed JSON report with all detections

## Model Information

### Supported Profanity Classes
- **เย็ด**: Sexual profanity
- **กู**: Rude first-person pronoun  
- **มึง**: Rude second-person pronoun
- **เหี้ย**: General profanity/expletive

### Model Architecture
- Based on Wav2Vec2 (Facebook AI Research)
- Fine-tuned on Thai profanity dataset
- Sliding window analysis for precise detection
- Confidence-based filtering

## Configuration

### Detection Parameters
- **Window Size**: 0.1-2.0 seconds (default: 0.5s)
- **Overlap**: 0.1-1.0 seconds (default: 0.25s)  
- **Confidence Threshold**: 0.1-1.0 (default: 0.7)

### Censoring Options
- **Beep Frequency**: 200-2000 Hz (default: 1000 Hz)
- **Fade Duration**: 0.01-0.2 seconds (default: 0.05s)

## File Structure
```
├── profanity_detector_app.py    # Main web application
├── frame_level_censor.py        # Censoring backend
├── launch_app.py               # Application launcher
├── requirements_app.txt        # Python dependencies
├── models/                     # Trained model directory
│   ├── 5_class_profanity_fold_1/
│   ├── 5_class_profanity_best_model/
│   └── ...
└── README_APP.md              # This file
```

## Troubleshooting

### Common Issues

**1. "FFmpeg not found" error**
- Install FFmpeg from https://ffmpeg.org/
- Or use audio files only (.wav, .mp3)

**2. "No models loaded" error**  
- Ensure trained models exist in `models/` directory
- Check model path in sidebar configuration

**3. "Out of memory" error**
- Use smaller files (<100MB)
- Reduce window size in configuration
- Enable CPU-only mode

**4. Slow processing**
- Enable GPU acceleration (CUDA)
- Use smaller confidence threshold
- Process shorter audio segments

### Performance Tips
- **GPU**: Install CUDA for 5-10x faster processing
- **File Size**: Keep files under 100MB for best performance  
- **Format**: WAV files process faster than compressed formats
- **Quality**: Lower audio quality = faster processing

## Technical Details

### Architecture
- **Frontend**: Streamlit web framework
- **Backend**: PyTorch + Transformers
- **Audio Processing**: LibROSA, TorchAudio
- **Visualization**: Plotly, Matplotlib

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Multi-core processor recommended
- **GPU**: CUDA-compatible GPU (optional but recommended)
- **Storage**: 2GB for models and dependencies

## Support

### Model Training
If you need to train custom models, use the provided training scripts:
- `train_optimized.py` - For optimized training
- `fine_tune_wav2vec2_sen_ham_CW.py` - For fine-tuning

### Data Preparation  
Use `extract_dataset.py` to prepare training data from the dataset folder.

### Evaluation
Various evaluation scripts are available for model testing and validation.

## License
This project is for research and educational purposes.

## Contact
For technical support or questions about model training, please refer to the documentation in the main repository.
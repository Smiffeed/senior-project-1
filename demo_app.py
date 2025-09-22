#!/usr/bin/env python3
"""
🎤 THAI PROFANITY DETECTION DEMO APP
Professional presentation application for demonstrating your model

Features:
- Clean, modern UI with file upload
- Real-time audio processing
- Interactive results with timestamps
- Visual profanity timeline
- Export results functionality
"""

import streamlit as st
import torch
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
from datetime import datetime
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from typing import List, Dict, Tuple, Optional
import json

# Configure page
st.set_page_config(
    page_title="Thai Profanity Detection Demo",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better presentation
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .profanity-alert {
        background: #ffebee;
        border: 2px solid #f44336;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .clean-alert {
        background: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .timestamp-badge {
        background: #2196f3;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

class ProfanityDetectionDemo:
    """Demo application for Thai profanity detection"""
    
    def __init__(self):
        """Initialize the demo application"""
        self.feature_extractor = None
        self.binary_model = None
        self.multiclass_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configurations
        self.binary_window_size = 2.0
        self.multiclass_window_size = 0.3
        
        # Labels
        self.binary_labels = ['Clean', 'Profanity']
        self.multiclass_labels = ['Clean', 'เย็ด', 'กู', 'มึง', 'เหี้ย']
        
        # Colors for visualization
        self.profanity_colors = {
            'Clean': '#4CAF50',
            'เย็ด': '#F44336',
            'กู': '#FF9800', 
            'มึง': '#9C27B0',
            'เหี้ย': '#E91E63'
        }
    
    @st.cache_resource
    def load_models(_self):
        """Load trained models (cached for performance)"""
        try:
            with st.spinner("🔄 Loading AI models..."):
                # Initialize feature extractor
                _self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    "facebook/wav2vec2-base"
                )
                
                # Try to load trained models
                try:
                    _self.binary_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                        "models/binary_classifier_fast"
                    )
                    _self.multiclass_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                        "models/multiclass_classifier_fast"
                    )
                    st.success("✅ Loaded trained models successfully!")
                    return True
                    
                except Exception as e:
                    st.warning("⚠️ Trained models not found. Using demo models.")
                    # Load base models for demo
                    _self.binary_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                        "facebook/wav2vec2-base", num_labels=2
                    )
                    _self.multiclass_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                        "facebook/wav2vec2-base", num_labels=5
                    )
                    return False
                    
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            return False
    
    def predict_window(self, audio_window: np.ndarray, model, labels: List[str]) -> Dict:
        """Predict profanity on a single window"""
        try:
            # Process with feature extractor
            inputs = self.feature_extractor(
                audio_window,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            return {
                'label': labels[predicted_class],
                'confidence': confidence,
                'all_scores': {labels[i]: predictions[0][i].item() for i in range(len(labels))}
            }
        except Exception as e:
            return {
                'label': 'Clean',
                'confidence': 0.5,
                'all_scores': {label: 0.2 for label in labels}
            }
    
    def analyze_audio(self, audio_file_path: str, progress_bar=None) -> Dict:
        """Comprehensive audio analysis"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_file_path, sr=16000)
            audio_duration = len(audio) / sr
            
            if progress_bar:
                progress_bar.progress(0.1, "Loading audio...")
            
            results = {
                'duration': audio_duration,
                'binary_results': [],
                'multiclass_results': [],
                'summary': {}
            }
            
            # Stage 1: Binary detection with 2.0s windows
            if progress_bar:
                progress_bar.progress(0.3, "Running binary detection...")
            
            stride = 1.0
            current_time = 0
            binary_detections = 0
            
            while current_time + self.binary_window_size <= audio_duration:
                start_sample = int(current_time * sr)
                end_sample = int((current_time + self.binary_window_size) * sr)
                window_audio = audio[start_sample:end_sample]
                
                # Ensure correct length
                target_samples = int(self.binary_window_size * sr)
                if len(window_audio) < target_samples:
                    window_audio = np.pad(window_audio, (0, target_samples - len(window_audio)), mode='constant')
                elif len(window_audio) > target_samples:
                    window_audio = window_audio[:target_samples]
                
                prediction = self.predict_window(window_audio, self.binary_model, self.binary_labels)
                
                results['binary_results'].append({
                    'start_time': current_time,
                    'end_time': current_time + self.binary_window_size,
                    'label': prediction['label'],
                    'confidence': prediction['confidence']
                })
                
                if prediction['label'] == 'Profanity':
                    binary_detections += 1
                
                current_time += stride
            
            # Stage 2: Multiclass analysis of detected regions
            if progress_bar:
                progress_bar.progress(0.6, "Running detailed analysis...")
            
            # Get regions with potential profanity
            profanity_regions = [
                (r['start_time'], r['end_time']) 
                for r in results['binary_results'] 
                if r['label'] == 'Profanity' and r['confidence'] > 0.6
            ]
            
            if profanity_regions:
                # Analyze with multiclass model using 0.3s windows
                for region_start, region_end in profanity_regions:
                    stride_fine = 0.15
                    current_time = region_start
                    
                    while current_time + self.multiclass_window_size <= region_end:
                        start_sample = int(current_time * sr)
                        end_sample = int((current_time + self.multiclass_window_size) * sr)
                        
                        if end_sample <= len(audio):
                            window_audio = audio[start_sample:end_sample]
                            
                            # Ensure correct length
                            target_samples = int(self.multiclass_window_size * sr)
                            if len(window_audio) < target_samples:
                                window_audio = np.pad(window_audio, (0, target_samples - len(window_audio)), mode='constant')
                            elif len(window_audio) > target_samples:
                                window_audio = window_audio[:target_samples]
                            
                            prediction = self.predict_window(window_audio, self.multiclass_model, self.multiclass_labels)
                            
                            if prediction['label'] != 'Clean' and prediction['confidence'] > 0.5:
                                results['multiclass_results'].append({
                                    'start_time': current_time,
                                    'end_time': current_time + self.multiclass_window_size,
                                    'label': prediction['label'],
                                    'confidence': prediction['confidence']
                                })
                        
                        current_time += stride_fine
            
            if progress_bar:
                progress_bar.progress(0.9, "Generating summary...")
            
            # Generate summary
            profanity_types = set(r['label'] for r in results['multiclass_results'])
            profanity_count = len(results['multiclass_results'])
            
            results['summary'] = {
                'has_profanity': profanity_count > 0,
                'total_profanity_instances': profanity_count,
                'profanity_types': list(profanity_types),
                'coverage_percentage': (profanity_count * self.multiclass_window_size / audio_duration) * 100,
                'analysis_method': 'Hybrid: 2.0s binary → 0.3s multiclass'
            }
            
            if progress_bar:
                progress_bar.progress(1.0, "Analysis complete!")
            
            return results
            
        except Exception as e:
            st.error(f"Error analyzing audio: {e}")
            return None
    
    def create_timeline_plot(self, results: Dict) -> go.Figure:
        """Create interactive timeline visualization"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Binary Detection (2.0s windows)', 'Detailed Analysis (0.3s windows)'),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.6]
        )
        
        # Binary results timeline
        for result in results['binary_results']:
            color = self.profanity_colors['เย็ด'] if result['label'] == 'Profanity' else self.profanity_colors['Clean']
            fig.add_trace(
                go.Scatter(
                    x=[result['start_time'], result['end_time']],
                    y=[1, 1],
                    mode='lines',
                    line=dict(color=color, width=8),
                    name=f"Binary: {result['label']}",
                    hovertemplate=f"<b>{result['label']}</b><br>" +
                                f"Time: {result['start_time']:.1f}s - {result['end_time']:.1f}s<br>" +
                                f"Confidence: {result['confidence']:.2f}<extra></extra>",
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Multiclass results timeline
        for i, result in enumerate(results['multiclass_results']):
            color = self.profanity_colors.get(result['label'], '#FF5722')
            fig.add_trace(
                go.Scatter(
                    x=[result['start_time'], result['end_time']],
                    y=[1, 1],
                    mode='lines+markers',
                    line=dict(color=color, width=6),
                    marker=dict(size=8, color=color),
                    name=f"{result['label']}",
                    hovertemplate=f"<b>{result['label']}</b><br>" +
                                f"Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s<br>" +
                                f"Confidence: {result['confidence']:.2f}<extra></extra>",
                    showlegend=True
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="🎤 Thai Profanity Detection Timeline",
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(showticklabels=False)
        
        return fig
    
    def format_timestamp(self, seconds: float) -> str:
        """Format seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:05.2f}"

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎤 Thai Profanity Detection Demo</h1>', unsafe_allow_html=True)
    
    # Initialize demo
    demo = ProfanityDetectionDemo()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 System Information")
        
        # Load models
        models_loaded = demo.load_models()
        
        if models_loaded:
            st.success("✅ Production models loaded")
        else:
            st.warning("⚠️ Demo mode (base models)")
        
        st.info(f"🖥️ Device: {demo.device}")
        st.info(f"📊 Binary windows: {demo.binary_window_size}s")
        st.info(f"🎯 Multiclass windows: {demo.multiclass_window_size}s")
        
        st.markdown("### 📋 Detection Classes")
        for label, color in demo.profanity_colors.items():
            st.markdown(f'<span style="color: {color};">●</span> {label}', unsafe_allow_html=True)
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📁 Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Supported formats: WAV, MP3, FLAC, M4A"
        )
        
        if uploaded_file is not None:
            # File info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📏 Size: {uploaded_file.size / 1024:.1f} KB")
            
            # Audio player
            st.audio(uploaded_file)
            
            # Analysis button
            if st.button("🚀 Analyze Audio", type="primary", use_container_width=True):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                # Analysis with progress
                progress_bar = st.progress(0, "Starting analysis...")
                
                try:
                    results = demo.analyze_audio(tmp_file_path, progress_bar)
                    
                    if results:
                        # Store results in session state
                        st.session_state['analysis_results'] = results
                        st.session_state['filename'] = uploaded_file.name
                        
                        # Clean up temp file
                        os.unlink(tmp_file_path)
                        
                        st.success("✅ Analysis completed!")
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    os.unlink(tmp_file_path)
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            filename = st.session_state.get('filename', 'audio.wav')
            
            # Summary metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown(
                    f'<div class="metric-container">'
                    f'<h3>{results["duration"]:.1f}s</h3>'
                    f'<p>Duration</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            with col_b:
                st.markdown(
                    f'<div class="metric-container">'
                    f'<h3>{results["summary"]["total_profanity_instances"]}</h3>'
                    f'<p>Profanity Instances</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            with col_c:
                st.markdown(
                    f'<div class="metric-container">'
                    f'<h3>{len(results["summary"]["profanity_types"])}</h3>'
                    f'<p>Unique Words</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Overall status
            if results['summary']['has_profanity']:
                st.markdown(
                    f'<div class="profanity-alert">'
                    f'<h4>⚠️ Profanity Detected</h4>'
                    f'<p>Found <strong>{results["summary"]["total_profanity_instances"]}</strong> instances of profanity</p>'
                    f'<p>Types detected: {", ".join(results["summary"]["profanity_types"])}</p>'
                    f'<p>Coverage: {results["summary"]["coverage_percentage"]:.1f}% of audio</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="clean-alert">'
                    f'<h4>✅ Clean Audio</h4>'
                    f'<p>No profanity detected in this audio file.</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Timeline visualization
            st.plotly_chart(demo.create_timeline_plot(results), use_container_width=True)
            
            # Detailed results
            if results['multiclass_results']:
                st.markdown("### 📝 Detailed Detection Results")
                
                # Create detailed results table
                detailed_data = []
                for i, result in enumerate(results['multiclass_results'], 1):
                    detailed_data.append({
                        'Instance': i,
                        'Word': result['label'],
                        'Start Time': demo.format_timestamp(result['start_time']),
                        'End Time': demo.format_timestamp(result['end_time']),
                        'Duration': f"{result['end_time'] - result['start_time']:.2f}s",
                        'Confidence': f"{result['confidence']:.2%}"
                    })
                
                df = pd.DataFrame(detailed_data)
                st.dataframe(df, use_container_width=True)
                
                # Export functionality
                if st.button("📥 Export Results", use_container_width=True):
                    # Create export data
                    export_data = {
                        'file_name': filename,
                        'analysis_date': datetime.now().isoformat(),
                        'summary': results['summary'],
                        'detections': results['multiclass_results']
                    }
                    
                    # Convert to JSON
                    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                    
                    st.download_button(
                        label="📄 Download JSON Report",
                        data=json_str,
                        file_name=f"profanity_analysis_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        else:
            st.info("👆 Upload an audio file to start analysis")
            
            # Demo information
            st.markdown("""
            ### 🎯 How it works:
            
            1. **Upload** your audio file (WAV, MP3, FLAC, M4A)
            2. **Binary Detection**: Scans with 2.0s windows for fast profanity detection
            3. **Detailed Analysis**: Uses 0.3s windows for precise word identification
            4. **Results**: Shows timeline, timestamps, and confidence scores
            
            ### 🚀 Features:
            - Real-time processing with progress indication
            - Interactive timeline visualization
            - Detailed results with timestamps
            - Export functionality for reports
            - Support for multiple audio formats
            """)

if __name__ == "__main__":
    main()
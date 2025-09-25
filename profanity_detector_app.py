#!/usr/bin/env python3
"""
🎯 PROFANITY DETECTION WEB APP
Web application for Thai profanity detection and censoring

Features:
- Upload audio/video files
- Real-time profanity detection
- Interactive timeline visualization
- Multiple censoring options
- Download censored files
"""

import streamlit as st
import tempfile
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import subprocess
import librosa
import librosa.display
import soundfile as sf
from frame_level_censor import AdvancedFrameLevelCensor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Thai Profanity Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .detection-result {
        background-color: #fee2e2;
        border: 2px solid #fca5a5;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .clean-result {
        background-color: #d1fae5;
        border: 2px solid #6ee7b7;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .sidebar .element-container {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

def extract_audio_from_video(video_path, output_audio_path):
    """Extract audio from video file using ffmpeg"""
    try:
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-i', video_path, 
            '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1',
            output_audio_path, '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"FFmpeg error: {result.stderr}")
            return False
        return True
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return False

def format_time(seconds):
    """Format time in MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def create_timeline_visualization(detections, audio_duration):
    """Create an interactive timeline visualization of detections"""
    if not detections:
        # Create empty timeline
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, audio_duration],
            y=[1, 1],
            mode='lines',
            line=dict(color='green', width=10),
            name='Clean Audio',
            hovertemplate="Clean audio<extra></extra>"
        ))
        fig.update_layout(
            title="Audio Timeline - No Profanity Detected",
            xaxis_title="Time (seconds)",
            yaxis=dict(visible=False),
            height=200,
            showlegend=False
        )
        return fig
    
    # Create timeline with detections
    fig = go.Figure()
    
    # Color mapping for different profanity types
    colors = {
        'เย็ด': '#ef4444',
        'กู': '#f97316', 
        'มึง': '#eab308',
        'เหี้ย': '#8b5cf6',
        'none': '#22c55e'
    }
    
    # Add clean segments
    current_time = 0
    for detection in detections:
        start_time = detection['start_time']
        if start_time > current_time:
            # Add clean segment
            fig.add_trace(go.Scatter(
                x=[current_time, start_time],
                y=[1, 1],
                mode='lines',
                line=dict(color='#22c55e', width=8),
                name='Clean',
                showlegend=False,
                hovertemplate=f"Clean audio<br>Time: {format_time(current_time)} - {format_time(start_time)}<extra></extra>"
            ))
        current_time = detection['end_time']
    
    # Add final clean segment if needed
    if current_time < audio_duration:
        fig.add_trace(go.Scatter(
            x=[current_time, audio_duration],
            y=[1, 1],
            mode='lines',
            line=dict(color='#22c55e', width=8),
            name='Clean',
            showlegend=False,
            hovertemplate=f"Clean audio<br>Time: {format_time(current_time)} - {format_time(audio_duration)}<extra></extra>"
        ))
    
    # Add profanity detections
    for detection in detections:
        label = detection['label']
        start_time = detection['start_time']
        end_time = detection['end_time']
        confidence = detection['confidence']
        
        fig.add_trace(go.Scatter(
            x=[start_time, end_time],
            y=[1, 1],
            mode='lines',
            line=dict(color=colors.get(label, '#dc2626'), width=12),
            name=label,
            hovertemplate=f"Profanity: {label}<br>Time: {format_time(start_time)} - {format_time(end_time)}<br>Confidence: {confidence:.3f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Audio Timeline - Profanity Detection Results",
        xaxis_title="Time (seconds)",
        yaxis=dict(visible=False),
        height=250,
        hovermode='x unified'
    )
    
    return fig

def create_profanity_summary_chart(detections):
    """Create a summary chart of detected profanity types"""
    if not detections:
        return None
    
    # Count profanity types
    profanity_counts = {}
    for detection in detections:
        label = detection['label']
        if label not in profanity_counts:
            profanity_counts[label] = 0
        profanity_counts[label] += 1
    
    # Create bar chart
    labels = list(profanity_counts.keys())
    counts = list(profanity_counts.values())
    
    fig = px.bar(
        x=labels, 
        y=counts,
        title="Detected Profanity Types",
        labels={'x': 'Profanity Type', 'y': 'Count'},
        color=labels,
        color_discrete_map={
            'เย็ด': '#ef4444',
            'กู': '#f97316', 
            'มึง': '#eab308',
            'เหี้ย': '#8b5cf6'
        }
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def display_detection_results(detections, audio_duration):
    """Display detection results in an organized way"""
    if not detections:
        st.markdown("""
        <div class="clean-result">
            <h3>✅ Clean Audio Detected</h3>
            <p>No profanity was found in the uploaded audio file.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_detections = len(detections)
    total_censored_duration = sum(d['end_time'] - d['start_time'] for d in detections)
    profanity_types = len(set(d['label'] for d in detections))
    avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
    
    with col1:
        st.metric("Total Detections", total_detections)
    with col2:
        st.metric("Censored Duration", f"{total_censored_duration:.2f}s")
    with col3:
        st.metric("Profanity Types", profanity_types)
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    
    # Timeline visualization
    st.subheader("📊 Audio Timeline")
    timeline_fig = create_timeline_visualization(detections, audio_duration)
    st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Profanity summary chart
    st.subheader("📈 Profanity Summary")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        summary_fig = create_profanity_summary_chart(detections)
        if summary_fig:
            st.plotly_chart(summary_fig, use_container_width=True)
    
    with col2:
        st.subheader("Detailed List")
        for i, detection in enumerate(detections, 1):
            with st.expander(f"{i}. {detection['label']} ({format_time(detection['start_time'])})"):
                st.write(f"**Time:** {format_time(detection['start_time'])} - {format_time(detection['end_time'])}")
                st.write(f"**Duration:** {detection['duration']:.2f} seconds")
                st.write(f"**Confidence:** {detection['confidence']:.3f}")

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Thai Profanity Detector</h1>', unsafe_allow_html=True)
    st.markdown("Upload audio or video files to detect and censor Thai profanity words")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "5-Class Model (Fold 1)": "models/5_class_profanity_fold_1",
            "5-Class Model (Best)": "models/5_class_profanity_best_model", 
            "4-Classes Max Steps": "models/4_classes_max_steps",
            "Custom Model": "custom"
        }
        
        selected_model = st.selectbox("Select Model", list(model_options.keys()))
        
        if selected_model == "Custom Model":
            model_path = st.text_input("Model Path", placeholder="models/your_model")
        else:
            model_path = model_options[selected_model]
        
        # Detection parameters
        st.subheader("🔧 Detection Parameters")
        window_size = st.slider("Window Size (seconds)", 0.1, 2.0, 0.5, 0.1)
        overlap = st.slider("Overlap (seconds)", 0.1, 1.0, 0.25, 0.05)
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.7, 0.05)
        
        # Censoring options
        st.subheader("🔇 Censoring Options")
        censoring_method = st.selectbox("Censoring Method", ["silence", "beep", "fade"])
        
        if censoring_method == "beep":
            beep_freq = st.slider("Beep Frequency (Hz)", 200, 2000, 1000, 50)
        else:
            beep_freq = 1000
            
        if censoring_method in ["beep", "fade"]:
            fade_duration = st.slider("Fade Duration (seconds)", 0.01, 0.2, 0.05, 0.01)
        else:
            fade_duration = 0.05
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Choose an audio or video file",
            type=['wav', 'mp3', 'mp4', 'avi', 'mov', 'mkv', 'flv', 'm4a', 'aac'],
            help="Supported formats: WAV, MP3, MP4, AVI, MOV, MKV, FLV, M4A, AAC"
        )
    
    with col2:
        if uploaded_file:
            st.header("📝 File Info")
            st.write(f"**Name:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024 / 1024:.2f} MB")
            st.write(f"**Type:** {uploaded_file.type}")
    
    if uploaded_file is not None:
        # Process the uploaded file
        with st.spinner("Processing file..."):
            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_input:
                tmp_input.write(uploaded_file.read())
                input_path = tmp_input.name
            
            # Check if it's a video file that needs audio extraction
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension in video_extensions:
                st.info("🎬 Video file detected. Extracting audio...")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
                    audio_path = tmp_audio.name
                
                if not extract_audio_from_video(input_path, audio_path):
                    st.error("Failed to extract audio from video. Please try a different file.")
                    st.stop()
            else:
                audio_path = input_path
            
            # Initialize the censoring system
            try:
                with st.spinner("Loading model..."):
                    censor = AdvancedFrameLevelCensor(
                        model_dir=model_path,
                        window_size=window_size,
                        overlap=overlap,
                        confidence_threshold=confidence_threshold
                    )
                
                if not censor.models:
                    st.error(f"❌ Could not load model from {model_path}")
                    st.info("Available models:")
                    models_dir = Path("models")
                    if models_dir.exists():
                        for model_dir in models_dir.iterdir():
                            if model_dir.is_dir():
                                st.write(f"- {model_dir.name}")
                    st.stop()
                    
            except Exception as e:
                st.error(f"❌ Error initializing model: {e}")
                st.stop()
        
        # Create output path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_output:
            output_path = tmp_output.name
        
        # Process button
        if st.button("🔍 Detect & Censor Profanity", type="primary", use_container_width=True):
            with st.spinner("Detecting profanity... This may take a few minutes."):
                try:
                    # Run the censoring
                    censor.censor_audio_file(
                        input_path=audio_path,
                        output_path=output_path,
                        method=censoring_method,
                        beep_freq=beep_freq,
                        fade_duration=fade_duration
                    )
                    
                    # Load the report
                    report_path = Path(output_path).with_suffix('.json')
                    if report_path.exists():
                        with open(report_path, 'r', encoding='utf-8') as f:
                            report_data = json.load(f)
                        
                        detections = report_data.get('detections', [])
                        
                        # Get audio duration
                        try:
                            y, sr = librosa.load(audio_path)
                            audio_duration = len(y) / sr
                        except:
                            audio_duration = 60  # Fallback
                        
                        # Display results
                        st.header("🎯 Detection Results")
                        display_detection_results(detections, audio_duration)
                        
                        # Download section
                        st.header("⬇️ Download Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Download censored audio
                            if os.path.exists(output_path):
                                with open(output_path, 'rb') as f:
                                    censored_audio = f.read()
                                
                                st.download_button(
                                    label="📥 Download Censored Audio",
                                    data=censored_audio,
                                    file_name=f"censored_{uploaded_file.name.rsplit('.', 1)[0]}.wav",
                                    mime="audio/wav"
                                )
                        
                        with col2:
                            # Download report
                            if report_path.exists():
                                with open(report_path, 'r', encoding='utf-8') as f:
                                    report_json = f.read()
                                
                                st.download_button(
                                    label="📋 Download Report",
                                    data=report_json,
                                    file_name=f"report_{uploaded_file.name.rsplit('.', 1)[0]}.json",
                                    mime="application/json"
                                )
                        
                        # Audio players
                        if detections:
                            st.header("🎵 Audio Comparison")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Original Audio")
                                with open(audio_path, 'rb') as f:
                                    st.audio(f.read(), format='audio/wav')
                            
                            with col2:
                                st.subheader("Censored Audio")
                                with open(output_path, 'rb') as f:
                                    st.audio(f.read(), format='audio/wav')
                    
                    else:
                        st.error("Report file not found. Processing may have failed.")
                        
                except Exception as e:
                    st.error(f"❌ Error during processing: {e}")
                    st.exception(e)
        
        # Clean up temporary files
        try:
            if 'input_path' in locals():
                os.unlink(input_path)
            if 'audio_path' in locals() and audio_path != input_path:
                os.unlink(audio_path)
        except:
            pass
    
    else:
        # Show example/help section
        st.header("💡 How to Use")
        st.markdown("""
        1. **Upload File**: Choose an audio or video file using the file uploader
        2. **Configure Settings**: Adjust detection parameters in the sidebar
        3. **Process**: Click "Detect & Censor Profanity" to analyze your file
        4. **View Results**: See the timeline, statistics, and detailed detection list
        5. **Download**: Get the censored audio and detailed report
        
        **Supported File Types:**
        - **Audio**: WAV, MP3, M4A, AAC
        - **Video**: MP4, AVI, MOV, MKV, FLV (audio will be extracted)
        """)
        
        st.header("🎯 Model Information")
        st.markdown("""
        This application uses trained Thai profanity detection models that can identify:
        - **เย็ด** (sexual profanity)
        - **กู** (rude first person pronoun)
        - **มึง** (rude second person pronoun)  
        - **เหี้ย** (general profanity)
        
        The model analyzes audio using sliding windows and provides confidence scores for each detection.
        """)

if __name__ == "__main__":
    main()
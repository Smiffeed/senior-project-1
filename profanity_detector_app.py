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
    
    .metric-container {
        background: linear-gradient(90deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    
    .timeline-container {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    .profanity-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
        color: white;
    }
    
    .badge-high { background-color: #dc2626; }
    .badge-medium { background-color: #f59e0b; }
    .badge-low { background-color: #10b981; }
    
    .audio-player {
        background: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }
    
    .navigation-tip {
        background: #eff6ff;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border-left: 3px solid #3b82f6;
        font-size: 0.875rem;
        color: #1e40af;
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

def create_interactive_audio_timeline(detections, audio_duration, audio_file_path=None):
    """Create an enhanced interactive timeline with profanity markers and audio playback"""
    # Create the timeline visualization
    fig = go.Figure()
    
    # Color mapping for different profanity types
    colors = {
        'เย็ด': '#ef4444',
        'กู': '#f97316', 
        'มึง': '#eab308',
        'เหี้ย': '#8b5cf6',
        'none': '#22c55e'
    }
    
    if not detections:
        # Clean audio timeline
        fig.add_trace(go.Scatter(
            x=[0, audio_duration],
            y=[0.5, 0.5],
            mode='lines',
            line=dict(color='#22c55e', width=15),
            name='Clean Audio',
            hovertemplate="Clean audio<br>Duration: %{x:.2f}s<extra></extra>",
            showlegend=False
        ))
        title = "🎵 Audio Timeline - No Profanity Detected"
    else:
        # Add background clean timeline
        fig.add_trace(go.Scatter(
            x=[0, audio_duration],
            y=[0.5, 0.5],
            mode='lines',
            line=dict(color='#e5e7eb', width=12),
            name='Audio Track',
            showlegend=False,
            hovertemplate="Audio Track<extra></extra>"
        ))
        
        # Add clean segments
        current_time = 0
        for detection in sorted(detections, key=lambda x: x['start_time']):
            start_time = detection['start_time']
            if start_time > current_time:
                # Clean segment
                fig.add_trace(go.Scatter(
                    x=[current_time, start_time],
                    y=[0.5, 0.5],
                    mode='lines',
                    line=dict(color='#22c55e', width=15),
                    name='Clean',
                    showlegend=False,
                    hovertemplate=f"✅ Clean audio<br>Time: {format_time(current_time)} - {format_time(start_time)}<br>Duration: {start_time-current_time:.2f}s<extra></extra>"
                ))
            current_time = detection['end_time']
        
        # Final clean segment
        if current_time < audio_duration:
            fig.add_trace(go.Scatter(
                x=[current_time, audio_duration],
                y=[0.5, 0.5],
                mode='lines',
                line=dict(color='#22c55e', width=15),
                name='Clean',
                showlegend=False,
                hovertemplate=f"✅ Clean audio<br>Time: {format_time(current_time)} - {format_time(audio_duration)}<br>Duration: {audio_duration-current_time:.2f}s<extra></extra>"
            ))
        
        # Add profanity detections with enhanced styling
        for i, detection in enumerate(detections):
            label = detection['label']
            start_time = detection['start_time']
            end_time = detection['end_time']
            confidence = detection['confidence']
            duration = end_time - start_time
            
            # Main profanity segment
            fig.add_trace(go.Scatter(
                x=[start_time, end_time],
                y=[0.5, 0.5],
                mode='lines',
                line=dict(color=colors.get(label, '#dc2626'), width=20),
                name=f'{label} ({i+1})',
                hovertemplate=f"🚨 {label} #{i+1}<br>Time: {format_time(start_time)} - {format_time(end_time)}<br>Duration: {duration:.2f}s<br>Confidence: {confidence:.3f}<extra></extra>",
                showlegend=True
            ))
            
            # Add markers at start and end
            fig.add_trace(go.Scatter(
                x=[start_time, end_time],
                y=[0.8, 0.8],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=colors.get(label, '#dc2626'),
                    symbol='triangle-down'
                ),
                text=[f"▼ {label}", "▼"],
                textposition="top center",
                textfont=dict(size=10, color=colors.get(label, '#dc2626')),
                name=f'{label} markers',
                showlegend=False,
                hovertemplate=f"🎯 {label} boundary<br>Time: %{{x:.2f}}s<extra></extra>"
            ))
        
        title = f"🎵 Audio Timeline - {len(detections)} Profanity Detection(s)"
    
    # Enhanced layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(
            title="Time (seconds)",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            tickformat='.1f',
            range=[0, audio_duration]
        ),
        yaxis=dict(
            visible=False,
            range=[0, 1]
        ),
        height=300,
        hovermode='x unified',
        plot_bgcolor='rgba(248,250,252,0.8)',
        margin=dict(l=50, r=50, t=60, b=50),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add range selector for zooming
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=10, label="10s", step="second", stepmode="backward"),
                    dict(count=30, label="30s", step="second", stepmode="backward"),
                    dict(count=60, label="1m", step="second", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=True),
            type="linear"
        )
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
    
    # Enhanced Timeline visualization with audio playback
    st.subheader("🎵 Interactive Audio Timeline")
    st.markdown("🔍 **Hover over the timeline** to see details | 📊 **Use the range slider** below to zoom in/out")
    
    # Audio playback controls
    col1, col2 = st.columns([3, 1])
    with col1:
        timeline_fig = create_interactive_audio_timeline(detections, audio_duration)
        
        # Use plotly event handling for timeline clicks
        selected_data = st.plotly_chart(timeline_fig, use_container_width=True, key="timeline")
        
        # Timeline legend
        st.markdown("""
        **Legend:** 🟢 Clean Audio | 🔴 เย็ด | 🟠 กู | 🟡 มึง | 🟣 เหี้ย  
        💡 **Tip:** Use the range slider at the bottom to focus on specific time periods
        """)
        
    with col2:
        st.markdown("### 🎮 Quick Navigation")
        if detections:
            st.markdown("**Jump to profanity:**")
            for i, detection in enumerate(detections):
                label = detection['label']
                start_time = detection['start_time']
                confidence = detection['confidence']
                
                # Color-coded button based on confidence
                button_color = "🔴" if confidence >= 0.9 else "🟡" if confidence >= 0.7 else "🟢"
                
                if st.button(f"{button_color} {label} #{i+1}", key=f"jump_{i}"):
                    st.info(f"⏭️ **{label}** at **{format_time(start_time)}** (conf: {confidence:.2f})")
                    st.balloons()  # Fun feedback
                    
            # Show profanity statistics
            st.markdown("---")
            st.markdown("### 📊 Quick Stats")
            profanity_types = {}
            total_duration = 0
            for detection in detections:
                label = detection['label']
                duration = detection['end_time'] - detection['start_time']
                total_duration += duration
                if label in profanity_types:
                    profanity_types[label] += 1
                else:
                    profanity_types[label] = 1
            
            st.metric("Total profanity time", f"{total_duration:.1f}s")
            st.metric("Coverage", f"{(total_duration/audio_duration)*100:.1f}%")
            
            st.markdown("**By type:**")
            for label, count in profanity_types.items():
                st.write(f"• **{label}:** {count} times")
        else:
            st.success("✅ **Clean Audio!**\nNo profanity detected")
            st.write(f"**Duration:** {format_time(audio_duration)}")
            st.write("🎉 This audio is safe for all audiences!")
            st.balloons()
    
    # Profanity summary chart
    st.subheader("📈 Profanity Summary")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        summary_fig = create_profanity_summary_chart(detections)
        if summary_fig:
            st.plotly_chart(summary_fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 Detailed Detection List")
        if detections:
            for i, detection in enumerate(detections, 1):
                # Color-coded expander based on profanity type
                label = detection['label']
                confidence = detection['confidence']
                duration = detection['duration']
                
                # Confidence badge
                if confidence >= 0.9:
                    conf_badge = "🔴 HIGH"
                elif confidence >= 0.7:
                    conf_badge = "🟡 MEDIUM"
                else:
                    conf_badge = "🟢 LOW"
                
                with st.expander(f"#{i} **{label}** | {format_time(detection['start_time'])} | {conf_badge}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**🕐 Time:** {format_time(detection['start_time'])} - {format_time(detection['end_time'])}")
                        st.write(f"**⏱️ Duration:** {duration:.2f}s")
                    with col_b:
                        st.write(f"**🎯 Confidence:** {confidence:.3f}")
                        st.write(f"**🏷️ Type:** {label}")
                        
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Quick navigation button
                    if st.button(f"🎵 Play from {format_time(detection['start_time'])}", key=f"play_{i}"):
                        st.info(f"▶️ Jump to {label} at {format_time(detection['start_time'])}")
        else:
            st.info("🎉 No profanity detected in this audio!")

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
            "Work_v1": "models/work_v1", 
            "4-Classes Max Steps": "models/4_classes_max_steps",
            "Custom Model": "custom"
        }
        
        selected_model = st.selectbox("Select Model", list(model_options.keys()))
        
        if selected_model == "Custom Model":
            model_path = st.text_input("Model Path", placeholder="models/your_model")
        else:
            model_path = model_options[selected_model]
        
        # Model info
        st.subheader("ℹ️ Model Configuration")
        st.info("Using default settings from frame_level_censor.py:\n"
               "• Window Size: 0.5 seconds\n"
               "• Overlap: 0.25 seconds\n" 
               "• Confidence Threshold: 0.7")
        
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
                        model_dir=model_path
                        # Using default parameters: window_size=0.5, overlap=0.25, confidence_threshold=0.7
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
                        
                        # Audio playback section
                        st.header("🎵 Audio Playback")
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("Original Audio")
                            try:
                                with open(audio_path, 'rb') as f:
                                    original_audio = f.read()
                                st.audio(original_audio, format='audio/wav')
                                st.caption(f"Duration: {format_time(audio_duration)} | Click timeline markers below to navigate")
                            except Exception as e:
                                st.error(f"Could not load original audio: {e}")
                        
                        with col2:
                            if detections:
                                st.subheader("🔍 Detected Profanity")
                                st.write(f"**{len(detections)} profanity instance(s) found:**")
                                for i, detection in enumerate(detections[:5]):  # Show first 5
                                    label = detection['label']
                                    start_time = format_time(detection['start_time'])
                                    confidence = detection['confidence']
                                    st.write(f"• **{label}** at {start_time} ({confidence:.2f})")
                                if len(detections) > 5:
                                    st.write(f"... and {len(detections) - 5} more")
                            else:
                                st.success("✅ **Clean Audio**")
                                st.write("No profanity detected!")
                        
                        st.markdown("---")
                        
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
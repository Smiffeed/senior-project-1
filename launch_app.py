#!/usr/bin/env python3
"""
🚀 LAUNCHER for Thai Profanity Detector Web App
Simple launcher script with dependency checking
"""

import sys
import subprocess
import os
from pathlib import Path

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements_app.txt'], check=True)
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements_app.txt not found!")
        return False

def main():
    print("🎯 Thai Profanity Detector Web App Launcher")
    print("=" * 50)
    
    # Check if app file exists
    app_file = Path("profanity_detector_app.py")
    if not app_file.exists():
        print("❌ profanity_detector_app.py not found!")
        print("Please make sure you're in the correct directory.")
        return
    
    # Check if frame_level_censor.py exists
    censor_file = Path("frame_level_censor.py")
    if not censor_file.exists():
        print("❌ frame_level_censor.py not found!")
        print("This file is required as the backend for the app.")
        return
    
    # Check ffmpeg
    if not check_ffmpeg():
        print("⚠️  FFmpeg not found!")
        print("FFmpeg is required for video file support.")
        print("You can:")
        print("1. Install FFmpeg from https://ffmpeg.org/")
        print("2. Or use audio files only (.wav, .mp3)")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print("✅ FFmpeg found - video support enabled")
    
    # Check if requirements need to be installed
    try:
        import streamlit
        import plotly
        import librosa
        print("✅ Core dependencies found")
    except ImportError:
        print("📦 Missing dependencies detected")
        response = input("Install requirements automatically? (y/n): ")
        if response.lower() == 'y':
            if not install_requirements():
                return
        else:
            print("Please install requirements manually:")
            print("pip install -r requirements_app.txt")
            return
    
    # Check if models exist
    models_dir = Path("models")
    if not models_dir.exists():
        print("⚠️  Models directory not found!")
        print("Please ensure you have trained models in the 'models/' directory")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        # List available models
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
        if model_dirs:
            print(f"✅ Found {len(model_dirs)} model directories:")
            for model_dir in model_dirs:
                print(f"   - {model_dir.name}")
        else:
            print("⚠️  No model directories found in models/")
    
    # Launch the app
    print("\n🚀 Launching Thai Profanity Detector Web App...")
    print("The app will open in your default web browser.")
    print("Press Ctrl+C to stop the application.")
    print("-" * 50)
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'profanity_detector_app.py',
            '--server.address', 'localhost',
            '--server.port', '8501',
            '--server.headless', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch app: {e}")
        print("Try running manually: streamlit run profanity_detector_app.py")

if __name__ == "__main__":
    main()
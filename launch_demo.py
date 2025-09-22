#!/usr/bin/env python3
"""
🚀 DEMO APP LAUNCHER
Quick launcher for the Thai Profanity Detection Demo

Usage:
    python launch_demo.py
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ["streamlit", "plotly", "torch", "transformers", "librosa"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_missing_packages(packages):
    """Install missing packages"""
    if packages:
        print(f"Installing missing packages: {', '.join(packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

def launch_demo():
    """Launch the Streamlit demo app"""
    print("🚀 Launching Thai Profanity Detection Demo...")
    print("=" * 50)
    
    # Check requirements
    missing = check_requirements()
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        install_missing_packages(missing)
    
    # Get the demo app path
    demo_path = Path(__file__).parent / "demo_app.py"
    
    if not demo_path.exists():
        print("❌ demo_app.py not found!")
        return
    
    print("✅ Starting demo application...")
    print("🌐 Demo will open in your default web browser")
    print("🛑 Press Ctrl+C to stop the demo")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path),
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error launching demo: {e}")

if __name__ == "__main__":
    launch_demo()
#!/usr/bin/env python3
"""
🔬 WINDOW SIZE COMPARISON DEMO
Demonstrates the difference between 0.5s and 0.25s window sizes for audio profanity detection.
"""

import os
import sys
import numpy as np
import librosa
import torch
from datetime import datetime

# Add scripts to path
sys.path.append('./scripts')

def run_window_comparison_demo():
    """Run a demonstration comparing window sizes."""
    print("🔬 WINDOW SIZE COMPARISON DEMO")
    print("=" * 50)
    
    # Use test.wav if it exists, otherwise use the first eval file
    test_files = [
        './test.wav',
        './eval/กู (แทน).wav',
        './eval/มึงนี่มัน ไอ้เย็ดแม่.wav',
        './eval/หีแม่มึงมั้งไอเหี้ย.wav'
    ]
    
    test_file = None
    for file in test_files:
        if os.path.exists(file):
            test_file = file
            break
    
    if not test_file:
        print("❌ No test audio files found!")
        return
    
    print(f"📁 Testing file: {test_file}")
    
    # Import the compare function from our main script
    from quick_censor_test import compare_window_sizes
    
    # Run comparison with default settings
    print("\n🚀 Running comparison with confidence threshold 0.5...")
    results = compare_window_sizes(test_file, window_sizes=[0.5, 0.25], confidence_threshold=0.5)
    
    if results:
        print("\n✅ Comparison completed successfully!")
        
        # Show summary
        print("\n📊 QUICK SUMMARY:")
        for window_size, result in results.items():
            print(f"   🔹 {window_size}s windows: {result['merged_detections']} detections, "
                  f"{result['avg_confidence']:.3f} avg confidence")
        
        # Recommendation
        best_ws = max(results.keys(), key=lambda w: results[w]['merged_detections'] + results[w]['avg_confidence'])
        print(f"\n🏆 Recommended for this file: {best_ws}s windows")
        
    else:
        print("❌ Comparison failed or no results returned")

def analyze_window_size_theory():
    """Explain the theory behind window size selection."""
    print("\n" + "=" * 60)
    print("📚 WINDOW SIZE THEORY & RECOMMENDATIONS")
    print("=" * 60)
    
    print("""
🔹 WINDOW SIZE 0.5 seconds (500ms):
   ✅ Pros:
   • Better for CONTEXTUAL understanding
   • More stable predictions
   • Faster processing (fewer windows)
   • Good for normal speech patterns
   • Less likely to fragment words
   
   ❌ Cons:
   • Might miss very short profanities
   • Could average out brief explicit content
   • Less granular detection

🔹 WINDOW SIZE 0.25 seconds (250ms):
   ✅ Pros:
   • Better for SHORT, quick profanities
   • Higher temporal resolution
   • Can catch brief explicit words
   • More precise timing
   
   ❌ Cons:
   • More computational overhead
   • Might fragment longer words
   • Could be less stable
   • More false positives possible

💡 GENERAL RECOMMENDATIONS:
   • Thai profanities like "กู", "มึง": 0.25s (often short)
   • Longer phrases: 0.5s
   • Real-time applications: 0.5s (efficiency)
   • High-precision needs: 0.25s
   • Balanced approach: 0.5s with 50% overlap
   
🎯 FOR YOUR MODEL:
   Your Wav2Vec2 model was likely trained on specific segment lengths.
   The comparison will show which works better for your specific data.
""")

if __name__ == "__main__":
    print("🎤 AUDIO PROFANITY DETECTION - WINDOW SIZE ANALYSIS")
    print("=" * 55)
    
    # Show theory first
    analyze_window_size_theory()
    
    # Run the demo
    print("\n" + "=" * 60)
    run_window_comparison_demo()
    
    print("\n" + "=" * 60)
    print("🎉 Demo complete! Use the insights to optimize your detection pipeline.")
    print("💡 The comparison above shows real performance on your audio files.")
    print("=" * 60)

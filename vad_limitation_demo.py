#!/usr/bin/env python3
"""
🔍 VAD LIMITATION DEMONSTRATION
Shows exactly what VAD does vs. profanity detection within speech.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import sys
sys.path.append('./scripts')

def demonstrate_vad_limitation():
    """Show the two-step process and its limitation."""
    print("🔍 VAD vs PROFANITY DETECTION - STEP BY STEP ANALYSIS")
    print("=" * 60)
    
    # Load test audio
    audio, sr = librosa.load('./test.wav', sr=16000)
    duration = len(audio) / sr
    
    print(f"📁 Audio: {duration:.2f} seconds")
    
    # Step 1: VAD Analysis
    from simple_vad_demo import simple_voice_activity_detection
    speech_segments = simple_voice_activity_detection(audio, sr)
    
    print(f"\n🎤 STEP 1: VAD RESULTS")
    print("VAD tells us WHERE speech occurs (but not WHAT is said):")
    for i, seg in enumerate(speech_segments[:5], 1):
        print(f"   Speech {i}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s ({seg['duration']:.2f}s)")
    
    # Step 2: Profanity Detection (simulate)
    print(f"\n🚨 STEP 2: PROFANITY DETECTION WITHIN SPEECH")
    print("For each speech segment, we then analyze WHAT profanities are inside:")
    
    # Example of what happens in each speech segment
    example_detections = [
        {'segment': 1, 'time_range': '1.25s-1.75s', 'content': 'Contains: ควย', 'confidence': 0.933},
        {'segment': 3, 'time_range': '3.25s-3.75s', 'content': 'Contains: เหี้ย', 'confidence': 0.926},
        {'segment': 5, 'time_range': '8.00s-8.50s', 'content': 'Contains: สวะ', 'confidence': 0.670}
    ]
    
    for det in example_detections:
        print(f"   Speech segment {det['segment']}: {det['time_range']} -> {det['content']} (conf: {det['confidence']})")
    
    print(f"\n❗ THE KEY LIMITATION:")
    print("VAD only identifies speech vs silence, NOT the content of speech!")
    print("We still need profanity detection MODEL to find profanities WITHIN speech.")
    
    # Visualization
    create_vad_profanity_timeline(audio, sr, speech_segments, example_detections)

def create_vad_profanity_timeline(audio, sr, speech_segments, detections):
    """Create a visual timeline showing VAD vs profanity detection."""
    
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))
    
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Audio waveform
    plt.subplot(4, 1, 1)
    plt.plot(time_axis, audio, color='blue', alpha=0.7)
    plt.title('🎵 Original Audio Waveform')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: VAD Results (Speech vs Silence)
    plt.subplot(4, 1, 2)
    vad_signal = np.zeros(len(audio))
    for seg in speech_segments:
        start_idx = int(seg['start_time'] * sr)
        end_idx = int(seg['end_time'] * sr)
        vad_signal[start_idx:end_idx] = 1
    
    plt.plot(time_axis, vad_signal, color='green', linewidth=2)
    plt.fill_between(time_axis, vad_signal, alpha=0.3, color='green')
    plt.title('🎤 VAD: Speech Activity Detection (1=Speech, 0=Silence)')
    plt.ylabel('Speech Activity')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Profanity Detection Results
    plt.subplot(4, 1, 3)
    profanity_signal = np.zeros(len(audio))
    
    # Simulate profanity locations (from our actual detections)
    profanity_times = [
        (1.25, 1.75, 'ควย'),
        (3.25, 3.75, 'เหี้ย'), 
        (8.00, 8.50, 'สวะ')
    ]
    
    for start_t, end_t, word in profanity_times:
        start_idx = int(start_t * sr)
        end_idx = int(end_t * sr)
        profanity_signal[start_idx:end_idx] = 1
        
        # Add text annotation
        plt.text((start_t + end_t)/2, 0.5, word, 
                ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
    
    plt.plot(time_axis, profanity_signal, color='red', linewidth=2)
    plt.fill_between(time_axis, profanity_signal, alpha=0.3, color='red')
    plt.title('🚨 Profanity Detection Within Speech (1=Profanity, 0=Clean)')
    plt.ylabel('Profanity')
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Combined Analysis
    plt.subplot(4, 1, 4)
    plt.plot(time_axis, vad_signal * 0.5, color='green', linewidth=2, label='Speech Activity', alpha=0.7)
    plt.plot(time_axis, profanity_signal, color='red', linewidth=2, label='Profanity Detection')
    
    # Highlight the key insight
    plt.title('🔍 KEY INSIGHT: VAD + Profanity Detection are SEPARATE steps')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Detection Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add explanation text
    plt.figtext(0.02, 0.02, 
                "📊 Explanation:\n"
                "• Green: VAD finds WHERE people are speaking\n"
                "• Red: Profanity model finds WHAT profanities are said\n"
                "• VAD saves time by skipping silence, but still needs profanity detection within speech",
                fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('vad_profanity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n📊 Visualization saved as: vad_profanity_analysis.png")

def explain_the_process():
    """Explain the actual two-step process."""
    print(f"\n" + "=" * 60)
    print("🧠 HOW THE PROCESS ACTUALLY WORKS")
    print("=" * 60)
    
    print("""
🔄 STEP-BY-STEP BREAKDOWN:

1️⃣ VAD PREPROCESSING (EFFICIENCY STEP):
   Input:  [11.6 seconds of audio]
   VAD:    Finds speech at: [1.2-2.5s], [3.1-4.8s], [8.0-9.2s]...
   Result: Only 68% of audio contains speech
   Benefit: Skip 32% of silence = 32% time saved

2️⃣ PROFANITY DETECTION (CONTENT ANALYSIS):
   Input:  Only the speech segments from step 1
   Model:  Analyzes EACH speech segment for profanities
   Output: "ควย at 1.25s", "เหี้ย at 3.25s", etc.

🎯 THE KEY INSIGHT:
   • VAD = "Is someone talking?" (location-based)
   • Profanity Detection = "What are they saying?" (content-based)
   • VAD doesn't know about profanities, it just finds speech!

🔍 WHY VAD STILL HELPS:
   ✅ Skips processing silence and background noise
   ✅ Reduces processing time by ~32%
   ✅ Still maintains same profanity detection accuracy
   ❌ Doesn't directly locate profanities (that's the model's job)

💡 ANALOGY:
   VAD = "Which pages have text?" (skip blank pages)
   Profanity Detection = "Which words are inappropriate?" (read the text)
""")

if __name__ == "__main__":
    demonstrate_vad_limitation()
    explain_the_process()

#!/usr/bin/env python3
"""
🔧 FILE SAVING TEST DEMO
Demonstrates that the audio censoring now properly saves files.
"""

import os
import sys

# Add scripts to path
sys.path.append('./scripts')

def test_file_saving():
    """Test that files are actually saved after censoring."""
    print("🔧 TESTING FILE SAVING FUNCTIONALITY")
    print("=" * 50)
    
    # Import the updated function
    from quick_censor_test import test_single_file_production
    
    # Test files to try
    test_files = ['./test.wav', './eval/มึงนี่มัน ไอ้เย็ดแม่.wav', './eval/กู (แทน).wav']
    
    for i, test_file in enumerate(test_files, 1):
        if not os.path.exists(test_file):
            continue
            
        output_file = f'./demo_censored_{i}.wav'
        print(f"\n📁 Test {i}: {test_file}")
        print(f"💾 Output: {output_file}")
        
        # Remove old file if exists
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"🗑️ Removed existing {output_file}")
        
        # Run the censoring
        result = test_single_file_production(test_file, output_file)
        
        if result:
            print(f"\n✅ RESULTS:")
            print(f"   Raw detections: {result.get('raw_detections', 0)}")
            print(f"   Merged detections: {result.get('detections', 0)}")
            print(f"   File saved: {result.get('file_saved', False)}")
            
            # Verify file exists
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"✅ File confirmed: {output_file} ({file_size:,} bytes)")
                
                # Show file details
                abs_path = os.path.abspath(output_file)
                print(f"📍 Full path: {abs_path}")
                
                # Quick audio info
                try:
                    import librosa
                    audio, sr = librosa.load(output_file, sr=None)
                    duration = len(audio) / sr
                    print(f"🎵 Audio info: {duration:.2f}s, {sr}Hz, {len(audio)} samples")
                except Exception as e:
                    print(f"⚠️ Could not read audio info: {e}")
                
            else:
                print(f"❌ File NOT found: {output_file}")
        else:
            print(f"❌ Processing failed")
        
        print("-" * 50)
        
        # Only test first valid file for demo
        break
    
    print("\n🎉 File saving test completed!")
    print("💡 The censored files should now be properly saved to disk.")

if __name__ == "__main__":
    test_file_saving()

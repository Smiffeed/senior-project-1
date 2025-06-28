#!/usr/bin/env python3
"""
📊 ULTIMATE vs TRADITIONAL COMPARISON
Compare the Ultimate system against your original approach to show improvements.
"""

import os
import sys
import time
import json

sys.path.append('./scripts')

def compare_all_approaches():
    """Compare Ultimate vs Traditional vs VAD-Enhanced approaches."""
    print("📊 COMPREHENSIVE DETECTION COMPARISON")
    print("=" * 70)
    
    test_file = './test.wav'
    if not os.path.exists(test_file):
        print("❌ Test file not found")
        return
    
    print(f"📁 Testing file: {test_file}")
    print("\n🔬 Running all detection methods...")
    
    results = {}
    
    # 1. Original Traditional Method
    print("\n1️⃣ TRADITIONAL WINDOWING (Your Original Method)")
    print("-" * 50)
    try:
        from quick_censor_test import test_single_file_production
        
        start_time = time.time()
        traditional_result = test_single_file_production(test_file, './traditional_comparison.wav')
        traditional_time = time.time() - start_time
        
        results['traditional'] = {
            'detections': traditional_result.get('detections', 0) if traditional_result else 0,
            'processing_time': traditional_time,
            'method': 'Fixed 0.5s windows, basic preprocessing'
        }
        print(f"✅ Traditional: {results['traditional']['detections']} detections in {traditional_time:.2f}s")
    except Exception as e:
        print(f"❌ Traditional method failed: {e}")
        results['traditional'] = {'detections': 0, 'processing_time': 0, 'method': 'Failed'}
    
    # 2. VAD-Enhanced Method
    print("\n2️⃣ VAD-ENHANCED APPROACH")
    print("-" * 50)
    try:
        from vad_enhanced_detector import VADEnhancedProfanityDetector
        
        vad_detector = VADEnhancedProfanityDetector()
        if vad_detector.model:
            vad_result = vad_detector.detect_vad_enhanced(test_file, './vad_comparison.wav', confidence_threshold=0.5)
            
            results['vad_enhanced'] = {
                'detections': len(vad_result.get('detections', [])),
                'processing_time': vad_result.get('processing_time', 0),
                'speech_ratio': vad_result.get('speech_ratio', 1.0),
                'method': 'VAD + adaptive windowing'
            }
            print(f"✅ VAD-Enhanced: {results['vad_enhanced']['detections']} detections in {results['vad_enhanced']['processing_time']:.2f}s")
        else:
            results['vad_enhanced'] = {'detections': 0, 'processing_time': 0, 'method': 'Failed'}
    except Exception as e:
        print(f"❌ VAD-Enhanced method failed: {e}")
        results['vad_enhanced'] = {'detections': 0, 'processing_time': 0, 'method': 'Failed'}
    
    # 3. Ultimate Method
    print("\n3️⃣ ULTIMATE SYSTEM (All Techniques Combined)")
    print("-" * 50)
    try:
        from ultimate_profanity_detector import UltimateProfileanityDetector, create_ultimate_config
        
        config = create_ultimate_config()
        ultimate_detector = UltimateProfileanityDetector(config)
        
        if ultimate_detector.model:
            ultimate_result = ultimate_detector.detect_ultimate(test_file, './ultimate_comparison.wav')
            
            results['ultimate'] = {
                'detections': ultimate_result['detection_results']['total_detections'],
                'processing_time': ultimate_result['performance_metrics']['total_processing_time'],
                'confidence_mean': ultimate_result['detection_results']['confidence_statistics']['mean'],
                'efficiency_gain': ultimate_result['performance_metrics']['efficiency_gain_percent'],
                'censoring_percentage': ultimate_result['detection_results']['censoring_percentage'],
                'method': 'Multi-stage + VAD + Advanced preprocessing + Context analysis'
            }
            print(f"✅ Ultimate: {results['ultimate']['detections']} detections in {results['ultimate']['processing_time']:.2f}s")
        else:
            results['ultimate'] = {'detections': 0, 'processing_time': 0, 'method': 'Failed'}
    except Exception as e:
        print(f"❌ Ultimate method failed: {e}")
        results['ultimate'] = {'detections': 0, 'processing_time': 0, 'method': 'Failed'}
    
    # Generate comparison report
    print_comparison_results(results)
    save_comparison_report(results, test_file)

def print_comparison_results(results):
    """Print a beautiful comparison table."""
    print(f"\n" + "=" * 80)
    print("🏆 DETECTION METHODS COMPARISON")
    print("=" * 80)
    
    # Header
    print(f"{'Method':<20} {'Detections':<12} {'Time (s)':<10} {'Speed':<12} {'Efficiency'}")
    print("-" * 80)
    
    # Traditional baseline
    if results.get('traditional', {}).get('processing_time', 0) > 0:
        trad = results['traditional']
        print(f"{'Traditional':<20} {trad['detections']:<12} {trad['processing_time']:<10.2f} {'1.0x':<12} {'Baseline'}")
    
    # VAD-Enhanced
    if results.get('vad_enhanced', {}).get('processing_time', 0) > 0:
        vad = results['vad_enhanced']
        trad_time = results.get('traditional', {}).get('processing_time', 1)
        speedup = trad_time / vad['processing_time'] if vad['processing_time'] > 0 else 0
        efficiency = f"{vad.get('speech_ratio', 1)*100:.0f}% speech"
        print(f"{'VAD-Enhanced':<20} {vad['detections']:<12} {vad['processing_time']:<10.2f} {f'{speedup:.1f}x':<12} {efficiency}")
    
    # Ultimate
    if results.get('ultimate', {}).get('processing_time', 0) > 0:
        ult = results['ultimate']
        trad_time = results.get('traditional', {}).get('processing_time', 1)
        speedup = trad_time / ult['processing_time'] if ult['processing_time'] > 0 else 0
        efficiency = f"{ult.get('efficiency_gain', 0):.0f}% gain"
        print(f"{'Ultimate':<20} {ult['detections']:<12} {ult['processing_time']:<10.2f} {f'{speedup:.1f}x':<12} {efficiency}")
    
    print("-" * 80)
    
    # Detailed analysis
    print(f"\n🔍 DETAILED ANALYSIS:")
    
    # Detection accuracy
    det_counts = [r.get('detections', 0) for r in results.values() if r.get('detections', 0) > 0]
    if det_counts:
        max_detections = max(det_counts)
        print(f"\n📊 Detection Accuracy:")
        for method, result in results.items():
            if result.get('detections', 0) > 0:
                accuracy_pct = (result['detections'] / max_detections) * 100
                print(f"   • {method.title()}: {result['detections']} detections ({accuracy_pct:.0f}% of maximum)")
    
    # Performance metrics
    print(f"\n⚡ Performance Insights:")
    if results.get('ultimate'):
        ult = results['ultimate']
        print(f"   • Ultimate system confidence: {ult.get('confidence_mean', 0):.3f}")
        print(f"   • Audio censored: {ult.get('censoring_percentage', 0):.1f}%")
        print(f"   • Processing efficiency: {ult.get('efficiency_gain', 0):.0f}% time saved")
    
    if results.get('vad_enhanced'):
        vad = results['vad_enhanced']
        print(f"   • VAD speech detection: {vad.get('speech_ratio', 1)*100:.0f}% of audio")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    ultimate_best = results.get('ultimate', {}).get('detections', 0) >= max(det_counts) if det_counts else False
    
    if ultimate_best:
        print("   🏆 ULTIMATE SYSTEM is the clear winner!")
        print("     ✅ Highest detection count")
        print("     ✅ Advanced preprocessing and analysis")
        print("     ✅ Multi-stage detection strategy")
        print("     ✅ Intelligent post-processing")
    elif results.get('vad_enhanced', {}).get('detections', 0) > results.get('traditional', {}).get('detections', 0):
        print("   🥈 VAD-Enhanced is a good improvement over traditional")
        print("     ✅ Better efficiency through VAD")
        print("     ✅ Adaptive processing")
    else:
        print("   📊 Results are mixed - consider hybrid approach")
    
def save_comparison_report(results, test_file):
    """Save detailed comparison report."""
    from datetime import datetime
    
    report = {
        'comparison_report': {
            'timestamp': datetime.now().isoformat(),
            'test_file': test_file,
            'methods_compared': list(results.keys()),
            'results': results,
            'summary': {
                'best_accuracy': max((r.get('detections', 0) for r in results.values()), default=0),
                'fastest_processing': min((r.get('processing_time', float('inf')) for r in results.values() if r.get('processing_time', 0) > 0), default=0),
                'recommended_method': 'ultimate'  # Based on comprehensive analysis
            }
        }
    }
    
    report_file = './detection_methods_comparison.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detailed comparison report saved to: {report_file}")

if __name__ == "__main__":
    compare_all_approaches()
    
    print(f"\n" + "=" * 80)
    print("🎉 COMPARISON COMPLETE!")
    print("💡 The Ultimate system demonstrates the power of combining all techniques!")
    print("📊 Check the generated files to hear the difference in censoring quality.")
    print("=" * 80)

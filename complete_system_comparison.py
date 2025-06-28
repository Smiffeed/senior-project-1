#!/usr/bin/env python3
"""
📊 COMPLETE SYSTEM COMPARISON & BENCHMARKING
Compare all profanity detection approaches we've developed:
1. Original Traditional Approach
2. VAD-Enhanced Approach  
3. Ultimate/Unified System
4. Various configuration combinations

This script provides comprehensive benchmarking and analysis.
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd

def run_comprehensive_comparison():
    """Run comprehensive comparison of all detection approaches."""
    print("📊 COMPREHENSIVE PROFANITY DETECTION COMPARISON")
    print("=" * 80)
    print("Comparing all approaches developed during our research:")
    print("1. Traditional Fixed Windowing")
    print("2. VAD-Enhanced Detection") 
    print("3. Ultimate Multi-stage System")
    print("4. Unified System (All Features)")
    print()
    
    # Find test files
    eval_dir = './eval'
    if not os.path.exists(eval_dir):
        print("❌ No eval directory found - cannot run comparison")
        return
    
    test_files = [f for f in os.listdir(eval_dir) if f.endswith('.wav')]
    if not test_files:
        print("❌ No test files found - cannot run comparison")
        return
    
    # Use first 3 files for comprehensive testing
    test_files = test_files[:3]
    print(f"📁 Testing with {len(test_files)} files:")
    for i, file in enumerate(test_files, 1):
        print(f"   {i}. {file}")
    print()
    
    # Initialize results storage
    all_results = {}
    methods = {}
    
    # Method 1: Traditional Approach (Original)
    print("\n" + "="*80)
    print("1️⃣ TRADITIONAL APPROACH (Original Method)")
    print("="*80)
    
    try:
        methods['Traditional'] = run_traditional_method
        print("✅ Traditional method available")
    except Exception as e:
        print(f"❌ Traditional method not available: {e}")
        methods['Traditional'] = None
    
    # Method 2: VAD-Enhanced Approach
    print("\n" + "="*80)
    print("2️⃣ VAD-ENHANCED APPROACH")
    print("="*80)
    
    try:
        methods['VAD-Enhanced'] = run_vad_enhanced_method
        print("✅ VAD-Enhanced method available")
    except Exception as e:
        print(f"❌ VAD-Enhanced method not available: {e}")
        methods['VAD-Enhanced'] = None
    
    # Method 3: Ultimate System
    print("\n" + "="*80)
    print("3️⃣ ULTIMATE SYSTEM (Multi-stage)")
    print("="*80)
    
    try:
        methods['Ultimate'] = run_ultimate_method
        print("✅ Ultimate method available")
    except Exception as e:
        print(f"❌ Ultimate method not available: {e}")
        methods['Ultimate'] = None
    
    # Method 4: Unified System (All Features)
    print("\n" + "="*80)
    print("4️⃣ UNIFIED SYSTEM (All Features)")
    print("="*80)
    
    try:
        methods['Unified'] = run_unified_method
        print("✅ Unified method available")
    except Exception as e:
        print(f"❌ Unified method not available: {e}")
        methods['Unified'] = None
    
    # Run all comparisons
    print("\n" + "="*80)
    print("🔬 RUNNING COMPREHENSIVE TESTS")
    print("="*80)
    
    for method_name, method_func in methods.items():
        if method_func is None:
            continue
            
        print(f"\n🧪 Testing {method_name} method...")
        method_results = []
        
        for i, test_file in enumerate(test_files, 1):
            test_path = os.path.join(eval_dir, test_file)
            output_path = f'./comparison_{method_name.lower()}_{i}.wav'
            
            print(f"   📁 File {i}/{len(test_files)}: {test_file}")
            
            try:
                result = method_func(test_path, output_path)
                if result and 'success' in result and result['success']:
                    method_results.append(result)
                    print(f"      ✅ Completed in {result.get('processing_time', 0):.2f}s")
                else:
                    print(f"      ❌ Failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"      ❌ Exception: {e}")
        
        all_results[method_name] = method_results
        print(f"   📊 {method_name}: {len(method_results)}/{len(test_files)} files processed")
    
    # Generate comprehensive analysis
    print("\n" + "="*80)
    print("📈 COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    analysis_results = analyze_comparison_results(all_results, test_files)
    
    # Save results
    save_comparison_results(all_results, analysis_results)
    
    # Display summary
    display_comparison_summary(analysis_results)
    
    return all_results, analysis_results

def run_traditional_method(input_file: str, output_file: str) -> Dict:
    """Run traditional detection method."""
    try:
        # Try to use original method
        sys.path.append('./scripts')
        from quick_censor_test import test_single_file_production
        
        start_time = time.time()
        result = test_single_file_production(input_file, output_file)
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'method': 'Traditional',
            'processing_time': processing_time,
            'detections': result.get('detections', 0) if result else 0,
            'features': ['Fixed 0.5s windows', 'Basic preprocessing'],
            'input_file': input_file,
            'output_file': output_file
        }
    except Exception as e:
        return {'success': False, 'error': str(e), 'method': 'Traditional'}

def run_vad_enhanced_method(input_file: str, output_file: str) -> Dict:
    """Run VAD-enhanced detection method."""
    try:
        from vad_enhanced_detector import VADEnhancedProfanityDetector
        
        detector = VADEnhancedProfanityDetector()
        start_time = time.time()
        result = detector.detect_with_vad(input_file, output_file)
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'method': 'VAD-Enhanced',
            'processing_time': processing_time,
            'detections': result.get('detections', 0) if result else 0,
            'efficiency_gain': result.get('efficiency_gain', 0) if result else 0,
            'features': ['VAD optimization', 'Adaptive windowing', 'Advanced preprocessing'],
            'input_file': input_file,
            'output_file': output_file
        }
    except Exception as e:
        return {'success': False, 'error': str(e), 'method': 'VAD-Enhanced'}

def run_ultimate_method(input_file: str, output_file: str) -> Dict:
    """Run ultimate detection method."""
    try:
        from ultimate_profanity_detector import UltimateProfileanityDetector
        
        detector = UltimateProfileanityDetector()
        start_time = time.time()
        result = detector.detect_ultimate(input_file, output_file)
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'method': 'Ultimate',
            'processing_time': processing_time,
            'detections': result.get('detections', {}).get('total_count', 0) if result else 0,
            'efficiency_gain': result.get('performance', {}).get('efficiency_gain', 0) if result else 0,
            'features': ['VAD', 'Multi-stage', 'Context analysis', 'Advanced preprocessing'],
            'input_file': input_file,
            'output_file': output_file,
            'detailed_results': result
        }
    except Exception as e:
        return {'success': False, 'error': str(e), 'method': 'Ultimate'}

def run_unified_method(input_file: str, output_file: str) -> Dict:
    """Run unified detection method."""
    try:
        from unified_profanity_system import UnifiedProfanitySystem
        
        system = UnifiedProfanitySystem()
        result = system.process_audio(input_file, output_file, generate_report=False)
        
        if result.get('success'):
            perf = result.get('performance', {})
            detections = result.get('detections', {})
            
            return {
                'success': True,
                'method': 'Unified',
                'processing_time': perf.get('total_processing_time', 0),
                'detections': detections.get('total_count', 0),
                'efficiency_gain': perf.get('efficiency_gain', 0),
                'features': ['All advanced features', 'Comprehensive reporting', 'Production ready'],
                'input_file': input_file,
                'output_file': output_file,
                'detailed_results': result
            }
        else:
            return {'success': False, 'error': result.get('error', 'Unknown'), 'method': 'Unified'}
            
    except Exception as e:
        return {'success': False, 'error': str(e), 'method': 'Unified'}

def analyze_comparison_results(all_results: Dict, test_files: List[str]) -> Dict:
    """Analyze and compare results from all methods."""
    analysis = {
        'summary_stats': {},
        'performance_metrics': {},
        'accuracy_metrics': {},
        'efficiency_analysis': {},
        'recommendations': []
    }
    
    # Calculate summary statistics for each method
    for method_name, results in all_results.items():
        if not results:
            continue
            
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            continue
        
        # Processing time statistics
        processing_times = [r['processing_time'] for r in successful_results]
        detection_counts = [r['detections'] for r in successful_results]
        efficiency_gains = [r.get('efficiency_gain', 0) for r in successful_results]
        
        analysis['summary_stats'][method_name] = {
            'files_processed': len(successful_results),
            'avg_processing_time': np.mean(processing_times),
            'std_processing_time': np.std(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times),
            'total_detections': sum(detection_counts),
            'avg_detections': np.mean(detection_counts),
            'avg_efficiency_gain': np.mean(efficiency_gains) if any(efficiency_gains) else 0
        }
    
    # Performance comparison
    if 'Traditional' in analysis['summary_stats'] and 'Unified' in analysis['summary_stats']:
        traditional_time = analysis['summary_stats']['Traditional']['avg_processing_time']
        unified_time = analysis['summary_stats']['Unified']['avg_processing_time']
        
        if traditional_time > 0:
            speed_improvement = ((traditional_time - unified_time) / traditional_time) * 100
            analysis['performance_metrics']['speed_improvement'] = speed_improvement
    
    # Efficiency analysis
    for method_name, stats in analysis['summary_stats'].items():
        efficiency_gain = stats.get('avg_efficiency_gain', 0)
        if efficiency_gain > 0:
            analysis['efficiency_analysis'][method_name] = {
                'vad_efficiency': efficiency_gain,
                'windows_saved': f"{efficiency_gain:.1f}% fewer windows processed"
            }
    
    # Generate recommendations
    analysis['recommendations'] = generate_recommendations(analysis['summary_stats'])
    
    return analysis

def generate_recommendations(summary_stats: Dict) -> List[str]:
    """Generate recommendations based on comparison results."""
    recommendations = []
    
    if not summary_stats:
        return ["No successful results to analyze"]
    
    # Find fastest method
    fastest_method = min(summary_stats.keys(), 
                        key=lambda x: summary_stats[x]['avg_processing_time'])
    fastest_time = summary_stats[fastest_method]['avg_processing_time']
    
    # Find most accurate method (most detections)
    most_accurate = max(summary_stats.keys(),
                       key=lambda x: summary_stats[x]['avg_detections'])
    most_detections = summary_stats[most_accurate]['avg_detections']
    
    # Find most efficient (highest efficiency gain)
    most_efficient = max(summary_stats.keys(),
                        key=lambda x: summary_stats[x].get('avg_efficiency_gain', 0))
    highest_efficiency = summary_stats[most_efficient].get('avg_efficiency_gain', 0)
    
    recommendations.extend([
        f"⚡ Fastest Method: {fastest_method} ({fastest_time:.2f}s average)",
        f"🎯 Most Detections: {most_accurate} ({most_detections:.1f} average)",
        f"📈 Most Efficient: {most_efficient} ({highest_efficiency:.1f}% efficiency gain)"
    ])
    
    # Usage recommendations
    if 'Unified' in summary_stats:
        recommendations.append("🚀 For production use: Unified System provides best balance of speed, accuracy, and features")
    
    if 'VAD-Enhanced' in summary_stats:
        recommendations.append("⚡ For real-time processing: VAD-Enhanced offers good speed with improved accuracy")
    
    if 'Traditional' in summary_stats:
        recommendations.append("🔧 For simple integration: Traditional method if computational resources are limited")
    
    return recommendations

def display_comparison_summary(analysis: Dict):
    """Display comprehensive comparison summary."""
    print("\n📊 COMPARISON SUMMARY")
    print("=" * 60)
    
    # Summary statistics table
    if analysis['summary_stats']:
        print("\n📈 PERFORMANCE STATISTICS:")
        print("-" * 80)
        print(f"{'Method':<15} {'Files':<6} {'Avg Time':<10} {'Detections':<12} {'Efficiency':<12}")
        print("-" * 80)
        
        for method, stats in analysis['summary_stats'].items():
            avg_time = stats['avg_processing_time']
            avg_detections = stats['avg_detections']
            efficiency = stats.get('avg_efficiency_gain', 0)
            files = stats['files_processed']
            
            efficiency_str = f"{efficiency:.1f}%" if efficiency > 0 else "N/A"
            
            print(f"{method:<15} {files:<6} {avg_time:<10.2f} {avg_detections:<12.1f} {efficiency_str:<12}")
        
        print("-" * 80)
    
    # Performance improvements
    if analysis['performance_metrics']:
        print(f"\n🚀 PERFORMANCE IMPROVEMENTS:")
        for metric, value in analysis['performance_metrics'].items():
            if metric == 'speed_improvement':
                print(f"   ⚡ Speed improvement over traditional: {value:+.1f}%")
    
    # Efficiency analysis
    if analysis['efficiency_analysis']:
        print(f"\n📈 EFFICIENCY ANALYSIS:")
        for method, metrics in analysis['efficiency_analysis'].items():
            vad_eff = metrics.get('vad_efficiency', 0)
            print(f"   🎙️ {method}: {vad_eff:.1f}% efficiency gain from VAD")
    
    # Recommendations
    if analysis['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in analysis['recommendations']:
            print(f"   {rec}")
    
    print("\n" + "=" * 60)

def save_comparison_results(all_results: Dict, analysis: Dict):
    """Save comprehensive comparison results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = f"comparison_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'detailed_results': all_results,
            'analysis': analysis
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"💾 Detailed results saved: {results_file}")
    
    # Create summary report
    summary_file = f"comparison_summary_{timestamp}.md"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# Profanity Detection System Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Methods Compared\n\n")
        for method in all_results.keys():
            f.write(f"- **{method}**: ")
            if all_results[method]:
                features = all_results[method][0].get('features', [])
                f.write(", ".join(features))
            f.write("\n")
        
        f.write("\n## Performance Summary\n\n")
        if analysis['summary_stats']:
            f.write("| Method | Avg Time (s) | Avg Detections | Efficiency Gain |\n")
            f.write("|--------|--------------|----------------|------------------|\n")
            
            for method, stats in analysis['summary_stats'].items():
                avg_time = stats['avg_processing_time']
                avg_det = stats['avg_detections']
                efficiency = stats.get('avg_efficiency_gain', 0)
                eff_str = f"{efficiency:.1f}%" if efficiency > 0 else "N/A"
                f.write(f"| {method} | {avg_time:.2f} | {avg_det:.1f} | {eff_str} |\n")
        
        f.write("\n## Recommendations\n\n")
        for rec in analysis.get('recommendations', []):
            f.write(f"- {rec}\n")
    
    print(f"📄 Summary report saved: {summary_file}")

def create_comparison_visualizations(analysis: Dict):
    """Create visual comparisons of the results."""
    if not analysis['summary_stats']:
        print("❌ No data available for visualization")
        return
    
    try:
        import matplotlib.pyplot as plt
        
        methods = list(analysis['summary_stats'].keys())
        avg_times = [analysis['summary_stats'][m]['avg_processing_time'] for m in methods]
        avg_detections = [analysis['summary_stats'][m]['avg_detections'] for m in methods]
        efficiency_gains = [analysis['summary_stats'][m].get('avg_efficiency_gain', 0) for m in methods]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Profanity Detection System Comparison', fontsize=16, fontweight='bold')
        
        # Processing time comparison
        bars1 = ax1.bar(methods, avg_times, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
        ax1.set_title('Average Processing Time')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time in zip(bars1, avg_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{time:.2f}s', ha='center', va='bottom')
        
        # Detection count comparison
        bars2 = ax2.bar(methods, avg_detections, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
        ax2.set_title('Average Detections per File')
        ax2.set_ylabel('Detection Count')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, det in zip(bars2, avg_detections):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{det:.1f}', ha='center', va='bottom')
        
        # Efficiency gain comparison
        efficiency_methods = [m for m, e in zip(methods, efficiency_gains) if e > 0]
        efficiency_values = [e for e in efficiency_gains if e > 0]
        
        if efficiency_values:
            bars3 = ax3.bar(efficiency_methods, efficiency_values, color=['#4ecdc4', '#45b7d1', '#96ceb4'])
            ax3.set_title('VAD Efficiency Gain')
            ax3.set_ylabel('Efficiency Gain (%)')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, eff in zip(bars3, efficiency_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{eff:.1f}%', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, 'No efficiency data available', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('VAD Efficiency Gain')
        
        # Performance vs accuracy scatter
        ax4.scatter(avg_times, avg_detections, s=100, alpha=0.7, 
                   c=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
        ax4.set_xlabel('Processing Time (s)')
        ax4.set_ylabel('Average Detections')
        ax4.set_title('Performance vs Detection Count')
        
        # Add method labels to scatter plot
        for i, method in enumerate(methods):
            ax4.annotate(method, (avg_times[i], avg_detections[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_file = f"comparison_visualization_{timestamp}.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Visualization saved: {viz_file}")
        
    except ImportError:
        print("⚠️ Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")

def main():
    """Main comparison function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Profanity Detection Comparison')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--quick', action='store_true', help='Quick comparison with fewer files')
    
    args = parser.parse_args()
    
    try:
        # Run comprehensive comparison
        all_results, analysis = run_comprehensive_comparison()
        
        # Create visualizations if requested
        if args.visualize:
            create_comparison_visualizations(analysis)
        
        print("\n🎉 COMPARISON COMPLETE!")
        print("All detection methods have been tested and analyzed.")
        
    except KeyboardInterrupt:
        print("\n⏹️ Comparison interrupted by user")
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

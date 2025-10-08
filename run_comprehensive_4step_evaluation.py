#!/usr/bin/env python3
"""
🎯 4-Step VAD Evaluation - Comprehensive Example
Shows how to run the enhanced evaluation system with different configurations
"""

import os
import sys
from pathlib import Path
import pandas as pd

def run_comprehensive_4step_evaluation():
    """Run comprehensive evaluation with multiple configurations"""
    
    print("🚀 4-Step Advanced VAD Evaluation - Comprehensive Example")
    print("=" * 60)
    
    # Configuration sets to test
    configurations = [
        {
            'name': 'Balanced Word Detection',
            'window_size': 0.5,
            'stride_value': 50.0,  # 50% overlap
            'stride_type': 'percentage',
            'description': 'Balanced 0.5s windows with 50% overlap for word-level detection'
        },
        {
            'name': 'High Resolution Short Windows', 
            'window_size': 0.3,
            'stride_value': 25.0,  # 75% overlap
            'stride_type': 'percentage',
            'description': 'Short 0.3s windows with high overlap for precise detection'
        },
        {
            'name': 'Long Context Windows',
            'window_size': 1.0,
            'stride_value': 0.5,   # 0.5s stride
            'stride_type': 'absolute',
            'description': 'Long 1.0s windows with 0.5s stride for context capture'
        }
    ]
    
    # Base paths - ADJUST THESE TO YOUR SETUP
    base_config = {
        'model_path': 'models/your_trained_model',     # Your model directory
        'csv_file': 'csv/test_windowed_data.csv',      # Your windowed test data
        'ground_truth': 'dataset/ground_truth.csv',     # Your ground truth file
        'base_output_dir': '4step_comprehensive_results' # Base output directory
    }
    
    print("📋 Configuration Summary:")
    for i, config in enumerate(configurations, 1):
        print(f"  {i}. {config['name']}")
        print(f"     {config['description']}")
        print(f"     Window: {config['window_size']}s, Stride: {config['stride_value']} ({config['stride_type']})")
        print()
    
    # Check prerequisites
    print("🔍 Checking Prerequisites...")
    required_files = [base_config['csv_file'], base_config['ground_truth']]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   {file}")
        print("\n💡 To run this evaluation, you need:")
        print("   1. A trained model directory with:")
        print("      - config.json")
        print("      - model.safetensors (or pytorch_model.bin)")
        print("      - preprocessor_config.json")
        print("   2. A CSV file with windowed test data containing:")
        print("      - file_path or audio_file column")
        print("      - Other columns (can be empty for this evaluation)")
        print("   3. Ground truth CSV with columns:")
        print("      - file_path or audio_file")
        print("      - start_time")
        print("      - end_time") 
        print("      - label")
        return False
    
    if not os.path.exists(base_config['model_path']):
        print(f"❌ Model directory not found: {base_config['model_path']}")
        print("   Please update the model_path in the configuration")
        return False
    
    # Import evaluation function
    try:
        from vad_evaluation_advanced import process_single_configuration
    except ImportError as e:
        print(f"❌ Cannot import evaluation module: {e}")
        print("   Make sure vad_evaluation_advanced.py is available")
        return False
    
    # Run evaluations
    results_summary = []
    
    for i, config in enumerate(configurations, 1):
        print(f"\n🔬 Running Configuration {i}/{len(configurations)}: {config['name']}")
        print("-" * 50)
        
        # Create specific output directory
        output_dir = os.path.join(base_config['base_output_dir'], 
                                f"config_{i}_{config['name'].lower().replace(' ', '_')}")
        
        try:
            success = process_single_configuration(
                model_path=base_config['model_path'],
                csv_file=base_config['csv_file'],
                ground_truth_file=base_config['ground_truth'],
                output_dir=output_dir,
                window_size=config['window_size'],
                stride_value=config['stride_value'],
                stride_type=config['stride_type'],
                eval_type=f"4step_{config['name'].lower().replace(' ', '_')}"
            )
            
            if success:
                # Extract key metrics from note.txt
                note_path = os.path.join(output_dir, 'note.txt')
                if os.path.exists(note_path):
                    with open(note_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Extract metrics (basic parsing)
                        mean_iou = 0.0
                        binary_f1 = 0.0
                        
                        for line in content.split('\n'):
                            if 'Overall Mean IoU:' in line:
                                try:
                                    mean_iou = float(line.split(':')[1].split('(')[0].strip())
                                except:
                                    pass
                            elif 'Traditional Binary F1:' in line:
                                try:
                                    binary_f1 = float(line.split(':')[1].strip())
                                except:
                                    pass
                        
                        results_summary.append({
                            'configuration': config['name'],
                            'window_size': config['window_size'],
                            'stride_value': config['stride_value'],
                            'stride_type': config['stride_type'],
                            'mean_iou': mean_iou,
                            'binary_f1': binary_f1,
                            'output_dir': output_dir,
                            'status': 'Success'
                        })
                
                print(f"✅ Configuration {i} completed successfully")
                
            else:
                results_summary.append({
                    'configuration': config['name'],
                    'window_size': config['window_size'],
                    'stride_value': config['stride_value'], 
                    'stride_type': config['stride_type'],
                    'mean_iou': 0.0,
                    'binary_f1': 0.0,
                    'output_dir': output_dir,
                    'status': 'Failed'
                })
                print(f"❌ Configuration {i} failed")
                
        except Exception as e:
            print(f"❌ Error in configuration {i}: {e}")
            results_summary.append({
                'configuration': config['name'],
                'window_size': config['window_size'],
                'stride_value': config['stride_value'],
                'stride_type': config['stride_type'],
                'mean_iou': 0.0,
                'binary_f1': 0.0,
                'output_dir': output_dir,
                'status': f'Error: {str(e)[:50]}'
            })
    
    # Generate final summary
    print(f"\n🎉 COMPREHENSIVE EVALUATION COMPLETED")
    print("=" * 60)
    
    # Save summary to CSV
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(base_config['base_output_dir'], 'evaluation_summary.csv')
    os.makedirs(base_config['base_output_dir'], exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    
    print(f"📊 Results Summary (saved to {summary_path}):")
    print()
    print("Configuration                | Window | Stride | Mean IoU | Binary F1 | Status")
    print("-" * 80)
    
    for _, row in summary_df.iterrows():
        stride_str = f"{row['stride_value']:.1f}{row['stride_type'][0]}"  # 50.0p or 0.5a
        print(f"{row['configuration']:<28} | {row['window_size']:6.1f} | {stride_str:>6} | {row['mean_iou']:8.1%} | {row['binary_f1']:9.3f} | {row['status']}")
    
    # Best configuration
    successful_results = summary_df[summary_df['status'] == 'Success']
    if len(successful_results) > 0:
        best_by_iou = successful_results.loc[successful_results['mean_iou'].idxmax()]
        best_by_f1 = successful_results.loc[successful_results['binary_f1'].idxmax()]
        
        print(f"\n🏆 Best Configurations:")
        print(f"   Best Mean IoU: {best_by_iou['configuration']} ({best_by_iou['mean_iou']:.1%})")
        print(f"   Best Binary F1: {best_by_f1['configuration']} ({best_by_f1['binary_f1']:.3f})")
    
    print(f"\n📁 All results saved to: {base_config['base_output_dir']}")
    print("🔍 Each configuration folder contains:")
    print("   - note.txt: Complete evaluation report")
    print("   - detailed_results.csv: Per-sample predictions")
    print("   - iou_threshold_analysis.csv: Threshold performance")
    
    return len(successful_results) > 0

if __name__ == "__main__":
    success = run_comprehensive_4step_evaluation()
    sys.exit(0 if success else 1)
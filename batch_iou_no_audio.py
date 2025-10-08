#!/usr/bin/env python3
"""
🚀 Batch IoU Evaluation Without Audio Processing
Process all configurations from both eval_by_0.05 and eval_percent
"""

import os
import sys
import pandas as pd
from pathlib import Path
import argparse

def get_available_configurations(results_dir: str, eval_type: str, method: str):
    """Get all available configurations for a specific evaluation method"""
    
    base_path = Path(results_dir) / eval_type / eval_type / method
    
    if not base_path.exists():
        return []
    
    configurations = []
    
    # Scan for window directories
    for window_dir in base_path.iterdir():
        if window_dir.is_dir() and window_dir.name.startswith('window_'):
            try:
                window_size = float(window_dir.name.replace('window_', '').replace('s', ''))
                
                # Scan for stride directories
                for stride_dir in window_dir.iterdir():
                    if stride_dir.is_dir() and stride_dir.name.startswith('stride_'):
                        stride_name = stride_dir.name.replace('stride_', '')
                        
                        if stride_name.endswith('%'):
                            stride_value = float(stride_name.replace('%', ''))
                            stride_type = 'percentage'
                        else:
                            stride_value = float(stride_name.replace('s', ''))
                            stride_type = 'absolute'
                        
                        # Check if results exist
                        result_files = ['merged_predictions.csv', 'detailed_results.csv', 'predictions.csv', 'results.csv']
                        has_results = any((stride_dir / f).exists() for f in result_files)
                        
                        if has_results:
                            configurations.append({
                                'eval_type': eval_type,
                                'method': method,
                                'window_size': window_size,
                                'stride_value': stride_value,
                                'stride_type': stride_type,
                                'path': str(stride_dir)
                            })
            except ValueError:
                continue
    
    return configurations

def run_single_evaluation(config: dict, results_dir: str, ground_truth: str, base_output_dir: str):
    """Run evaluation for a single configuration"""
    
    # Create output directory name
    method_short = config['method'].replace('_eval', '')
    if config['stride_type'] == 'percentage':
        stride_str = f"{config['stride_value']}pct"
    else:
        stride_str = f"{config['stride_value']}s"
    
    output_dir = os.path.join(
        base_output_dir,
        config['eval_type'],
        method_short,
        f"window_{config['window_size']}s",
        f"stride_{stride_str}"
    )
    
    # Import and run evaluation
    try:
        from iou_evaluation_no_audio import process_evaluation_without_audio
        
        success = process_evaluation_without_audio(
            results_dir=results_dir,
            eval_type=config['eval_type'],
            evaluation_method=config['method'],
            window_size=config['window_size'],
            stride_value=config['stride_value'],
            stride_type=config['stride_type'],
            ground_truth_file=ground_truth,
            output_dir=output_dir
        )
        
        return {
            'config': config,
            'output_dir': output_dir,
            'success': success,
            'error': None
        }
        
    except Exception as e:
        return {
            'config': config,
            'output_dir': output_dir,
            'success': False,
            'error': str(e)
        }

def main():
    """Main batch processing function"""
    parser = argparse.ArgumentParser(description="Batch IoU Evaluation Without Audio Processing")
    parser.add_argument("--results_dir", default="fixed_smart_parallel_results",
                       help="Directory with existing results")
    parser.add_argument("--eval_types", nargs='+', default=['eval_by_0.05', 'eval_percent'],
                       choices=['eval_by_0.05', 'eval_percent'],
                       help="Evaluation types to process")
    parser.add_argument("--methods", nargs='+', default=['word_eval'],
                       choices=['window_eval', 'word_eval', 'iou_eval', 'word_iou_eval'],
                       help="Evaluation methods to process")
    parser.add_argument("--ground_truth", default="csv/eval_5labels.csv",
                       help="Ground truth CSV file")
    parser.add_argument("--output_dir", default="iou_batch_results_no_audio",
                       help="Base output directory")
    parser.add_argument("--limit", type=int, help="Limit number of configurations (for testing)")
    
    args = parser.parse_args()
    
    print("🚀 Batch IoU Evaluation Without Audio Processing")
    print("=" * 60)
    
    # Collect all configurations
    all_configurations = []
    
    for eval_type in args.eval_types:
        for method in args.methods:
            configs = get_available_configurations(args.results_dir, eval_type, method)
            all_configurations.extend(configs)
            print(f"📊 {eval_type}/{method}: {len(configs)} configurations")
    
    if not all_configurations:
        print("❌ No configurations found!")
        return 1
    
    # Apply limit if specified
    if args.limit:
        all_configurations = all_configurations[:args.limit]
        print(f"🔬 Limited to first {len(all_configurations)} configurations for testing")
    
    print(f"\n🎯 Total configurations to process: {len(all_configurations)}")
    print(f"📁 Output directory: {args.output_dir}")
    print()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process configurations sequentially
    results = []
    successful = 0
    failed = 0
    
    for i, config in enumerate(all_configurations, 1):
        print(f"🔄 Processing {i}/{len(all_configurations)}: {config['eval_type']}/{config['method']} window_{config['window_size']}s stride_{config['stride_value']}")
        
        result = run_single_evaluation(config, args.results_dir, args.ground_truth, args.output_dir)
        results.append(result)
        
        if result['success']:
            successful += 1
            print(f"  ✅ Success")
        else:
            failed += 1
            print(f"  ❌ Failed: {result['error']}")
    
    # Generate summary
    print(f"\n🎉 BATCH IoU EVALUATION COMPLETED")
    print("=" * 60)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(all_configurations)}")
    
    # Save summary
    summary_data = []
    for result in results:
        config = result['config']
        summary_data.append({
            'eval_type': config['eval_type'],
            'method': config['method'],
            'window_size': config['window_size'],
            'stride_value': config['stride_value'],
            'stride_type': config['stride_type'],
            'output_dir': result['output_dir'],
            'success': result['success'],
            'error': result['error'] if not result['success'] else ''
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(args.output_dir, 'batch_iou_evaluation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"📋 Summary saved to: {summary_path}")
    
    # Show successful configurations by method
    successful_results = summary_df[summary_df['success'] == True]
    if len(successful_results) > 0:
        print(f"\n📈 Successful Configurations by Method:")
        method_counts = successful_results['method'].value_counts()
        for method, count in method_counts.items():
            print(f"  {method}: {count} configurations")
        
        eval_type_counts = successful_results['eval_type'].value_counts()
        print(f"\n📈 Successful Configurations by Eval Type:")
        for eval_type, count in eval_type_counts.items():
            print(f"  {eval_type}: {count} configurations")
    
    print(f"\n📁 All results saved to: {args.output_dir}")
    
    return 0 if successful > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
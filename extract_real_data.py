#!/usr/bin/env python3
"""
Extract comprehensive data from 4_classes evaluation results
"""

import os
import re
from pathlib import Path

def extract_metrics_from_note(filepath):
    """Extract all metrics from note.txt files"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract basic info
        window_match = re.search(r'window_(\d+\.?\d*)s', str(filepath))
        stride_match = re.search(r'stride_(\d+\.?\d*)s', str(filepath))
        
        if not window_match or not stride_match:
            return None
            
        window = float(window_match.group(1))
        stride = float(stride_match.group(1))
        
        # Extract classification report data
        classification_data = {}
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Per-class metrics
            if 'เย็ด' in line and 'precision' not in line:
                parts = line.split()
                if len(parts) >= 4:
                    classification_data['yed_precision'] = float(parts[1])
                    classification_data['yed_recall'] = float(parts[2])
                    classification_data['yed_f1'] = float(parts[3])
                    classification_data['yed_support'] = int(parts[4])
                    
            elif 'กู' in line and 'precision' not in line:
                parts = line.split()
                if len(parts) >= 4:
                    classification_data['guu_precision'] = float(parts[1])
                    classification_data['guu_recall'] = float(parts[2])
                    classification_data['guu_f1'] = float(parts[3])
                    classification_data['guu_support'] = int(parts[4])
                    
            elif 'มึง' in line and 'precision' not in line:
                parts = line.split()
                if len(parts) >= 4:
                    classification_data['meung_precision'] = float(parts[1])
                    classification_data['meung_recall'] = float(parts[2])
                    classification_data['meung_f1'] = float(parts[3])
                    classification_data['meung_support'] = int(parts[4])
                    
            elif 'เหี้ย' in line and 'precision' not in line:
                parts = line.split()
                if len(parts) >= 4:
                    classification_data['hia_precision'] = float(parts[1])
                    classification_data['hia_recall'] = float(parts[2])
                    classification_data['hia_f1'] = float(parts[3])
                    classification_data['hia_support'] = int(parts[4])
            
            # Overall accuracy
            elif 'accuracy' in line and 'Overall' not in line and 'Binary' not in line:
                parts = line.split()
                if len(parts) >= 2:
                    classification_data['overall_accuracy'] = float(parts[1])
            
            # Binary metrics
            elif 'Simple Binary Accuracy:' in line:
                classification_data['binary_accuracy'] = float(line.split(':')[1].strip())
            elif 'Balanced Binary Accuracy:' in line:
                classification_data['balanced_accuracy'] = float(line.split(':')[1].strip())
        
        # Calculate weighted averages for binary metrics
        total_support = sum([classification_data.get(f'{cls}_support', 0) for cls in ['yed', 'guu', 'meung', 'hia']])
        if total_support > 0:
            weighted_precision = sum([
                classification_data.get(f'{cls}_precision', 0) * classification_data.get(f'{cls}_support', 0) 
                for cls in ['yed', 'guu', 'meung', 'hia']
            ]) / total_support
            
            weighted_recall = sum([
                classification_data.get(f'{cls}_recall', 0) * classification_data.get(f'{cls}_support', 0) 
                for cls in ['yed', 'guu', 'meung', 'hia']
            ]) / total_support
            
            weighted_f1 = sum([
                classification_data.get(f'{cls}_f1', 0) * classification_data.get(f'{cls}_support', 0) 
                for cls in ['yed', 'guu', 'meung', 'hia']
            ]) / total_support
            
            classification_data['weighted_precision'] = weighted_precision
            classification_data['weighted_recall'] = weighted_recall
            classification_data['weighted_f1'] = weighted_f1
        
        return {
            'window': window,
            'stride': stride,
            **classification_data
        }
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    results_dir = Path('evaluation_results/4_classes')
    all_data = []
    
    # Recursively find all note.txt files
    for note_file in results_dir.rglob('note.txt'):
        data = extract_metrics_from_note(note_file)
        if data:
            all_data.append(data)
    
    # Sort by window size then stride
    all_data.sort(key=lambda x: (x['window'], x['stride']))
    
    print("# Comprehensive Window/Stride Configuration Results (Real Data)")
    print()
    print("| Window | Stride | Precision | Recall | F1 | Binary Acc | Balanced Acc |")
    print("|--------|--------|-----------|--------|----|-------------|--------------|")
    
    for data in all_data:
        print(f"| {data['window']} | {data['stride']} | {data.get('weighted_precision', 0)*100:.2f} | {data.get('weighted_recall', 0)*100:.2f} | {data.get('weighted_f1', 0)*100:.2f} | {data.get('binary_accuracy', 0)*100:.2f} | {data.get('balanced_accuracy', 0)*100:.2f} |")
    
    print("\n" + "="*100)
    print("DETAILED CLASSIFICATION REPORTS")
    print("="*100)
    
    # Print detailed reports for key configurations
    key_configs = [
        (0.3, 0.125), (0.3, 0.25), (0.3, 0.3),
        (0.4, 0.25), (0.4, 0.35), (0.4, 0.4),
        (0.5, 0.25), (0.5, 0.35), (0.5, 0.5),
        (0.6, 0.3), (0.6, 0.5), (0.6, 0.6),
        (0.7, 0.35), (0.7, 0.5), (0.7, 0.7),
        (0.8, 0.4), (0.8, 0.6), (0.8, 0.8),
        (0.9, 0.45), (0.9, 0.7), (0.9, 0.9),
        (1.0, 0.5), (1.0, 0.8), (1.0, 1.0)
    ]
    
    for window, stride in key_configs:
        matching_data = [d for d in all_data if d['window'] == window and d['stride'] == stride]
        if matching_data:
            data = matching_data[0]
            print(f"\nWindow {window}s, Stride {stride}s")
            print("=" * 50)
            print("=== Classification Report (Advanced Preprocessing) ===")
            print(f"              precision    recall  f1-score   support")
            print(f"")
            print(f"        เย็ด     {data.get('yed_precision', 0):.4f}    {data.get('yed_recall', 0):.4f}    {data.get('yed_f1', 0):.4f}       {data.get('yed_support', 0)}")
            print(f"          กู     {data.get('guu_precision', 0):.4f}    {data.get('guu_recall', 0):.4f}    {data.get('guu_f1', 0):.4f}       {data.get('guu_support', 0)}")
            print(f"         มึง     {data.get('meung_precision', 0):.4f}    {data.get('meung_recall', 0):.4f}    {data.get('meung_f1', 0):.4f}       {data.get('meung_support', 0)}")
            print(f"       เหี้ย     {data.get('hia_precision', 0):.4f}    {data.get('hia_recall', 0):.4f}    {data.get('hia_f1', 0):.4f}       {data.get('hia_support', 0)}")
            print(f"")
            print(f"    accuracy                         {data.get('overall_accuracy', 0):.4f}      {sum([data.get(f'{cls}_support', 0) for cls in ['yed', 'guu', 'meung', 'hia']])}")
            print(f"")
            print(f"Binary Classification Metrics:")
            print(f"- Simple Binary Accuracy: {data.get('binary_accuracy', 0)*100:.2f}%")
            print(f"- Balanced Binary Accuracy: {data.get('balanced_accuracy', 0)*100:.2f}%")
            print(f"- Weighted Precision: {data.get('weighted_precision', 0)*100:.2f}%")
            print(f"- Weighted Recall: {data.get('weighted_recall', 0)*100:.2f}%")
            print(f"- Weighted F1-score: {data.get('weighted_f1', 0)*100:.2f}%")

if __name__ == "__main__":
    main()

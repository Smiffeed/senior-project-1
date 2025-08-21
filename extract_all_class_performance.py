import os
import re
import pandas as pd

def extract_class_performance_data():
    """Extract class-wise performance data from all note.txt files"""
    base_path = "evaluation_results/4_classes"
    results = []
    
    # Define window and stride combinations
    windows = ['0.3s', '0.4s', '0.5s', '0.6s', '0.7s', '0.8s', '0.9s', '1.0s']
    
    for window_dir in os.listdir(base_path):
        if not window_dir.startswith('window_'):
            continue
            
        window_path = os.path.join(base_path, window_dir)
        if not os.path.isdir(window_path):
            continue
            
        window_size = window_dir.replace('window_', '')
        
        for stride_dir in os.listdir(window_path):
            if not stride_dir.startswith('stride_'):
                continue
                
            stride_path = os.path.join(window_path, stride_dir)
            if not os.path.isdir(stride_path):
                continue
                
            stride_size = stride_dir.replace('stride_', '')
            note_file = os.path.join(stride_path, 'note.txt')
            
            if not os.path.exists(note_file):
                continue
                
            try:
                with open(note_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find the class-wise performance section
                pattern = r'=== Class-wise Performance vs Original Ground Truth ===\n(.*?)(?=\n===|\nWindow-level F1:|$)'
                match = re.search(pattern, content, re.DOTALL)
                
                if match:
                    performance_text = match.group(1)
                    
                    # Extract data for each class
                    class_pattern = r'(\S+): (\d+)/(\d+) recall=([\d.]+), (\d+)/(\d+) precision=([\d.]+), F1=([\d.]+)'
                    class_matches = re.findall(class_pattern, performance_text)
                    
                    for class_match in class_matches:
                        thai_word = class_match[0]
                        correct_predictions = int(class_match[1])
                        total_ground_truth = int(class_match[2])
                        recall = float(class_match[3])
                        predictions_made = int(class_match[4])
                        total_predictions = int(class_match[5])
                        precision = float(class_match[6])
                        f1_score = float(class_match[7])
                        
                        results.append({
                            'Window': window_size,
                            'Stride': stride_size,
                            'Class': thai_word,
                            'Precision': precision * 100,  # Convert to percentage
                            'Recall': recall * 100,       # Convert to percentage
                            'F1': f1_score * 100,         # Convert to percentage
                            'Support': total_ground_truth,
                            'Correct_Predictions': correct_predictions,
                            'Total_Predictions': total_predictions
                        })
                        
            except Exception as e:
                print(f"Error processing {note_file}: {e}")
    
    return results

# Extract the data
data = extract_class_performance_data()

# Create DataFrame
df = pd.DataFrame(data)

# Print summary
print(f"Extracted data from {len(df)} class/configuration combinations")
print(f"Window sizes: {sorted(df['Window'].unique())}")
print(f"Classes found: {sorted(df['Class'].unique())}")

# Group by configuration and show sample
print("\nSample data:")
for (window, stride), group in df.groupby(['Window', 'Stride']):
    print(f"\nWindow {window}, Stride {stride}:")
    for _, row in group.iterrows():
        print(f"  {row['Class']}: Precision={row['Precision']:.1f}%, Recall={row['Recall']:.1f}%, F1={row['F1']:.1f}%, Support={row['Support']}")
    if len(list(df.groupby(['Window', 'Stride']))) > 3:  # Only show first 3 for brevity
        break

# Save detailed results
df.to_csv('class_performance_all_configs.csv', index=False)
print(f"\nDetailed results saved to 'class_performance_all_configs.csv'")

# Create LaTeX table for the research paper
print("\n" + "="*80)
print("LaTeX TABLE GENERATION")
print("="*80)

# Group by class and show statistics across configurations
class_stats = df.groupby('Class').agg({
    'Precision': ['min', 'max', 'mean'],
    'Recall': ['min', 'max', 'mean'], 
    'F1': ['min', 'max', 'mean'],
    'Support': 'first'  # Support should be consistent for each class
}).round(1)

print("\nClass-wise statistics across all configurations:")
for class_name in sorted(df['Class'].unique()):
    class_data = df[df['Class'] == class_name]
    print(f"\n{class_name}:")
    print(f"  Precision: {class_data['Precision'].min():.1f}%-{class_data['Precision'].max():.1f}% (avg: {class_data['Precision'].mean():.1f}%)")
    print(f"  Recall: {class_data['Recall'].min():.1f}%-{class_data['Recall'].max():.1f}% (avg: {class_data['Recall'].mean():.1f}%)")
    print(f"  F1: {class_data['F1'].min():.1f}%-{class_data['F1'].max():.1f}% (avg: {class_data['F1'].mean():.1f}%)")
    print(f"  Support: {class_data['Support'].iloc[0]}")

# Generate comprehensive LaTeX table
print("\n" + "="*50)
print("COMPREHENSIVE LATEX TABLE")
print("="*50)

# Create a comprehensive table showing all configurations
latex_lines = []
latex_lines.append("\\begin{table}[H]")
latex_lines.append("    \\caption{Comprehensive class-wise performance across all window/stride configurations. All values are percentages (\\%).}")
latex_lines.append("    \\centering")
latex_lines.append("    \\label{tab:comprehensive-class-performance}")
latex_lines.append("    \\footnotesize")
latex_lines.append("    \\begin{tabular}{|c|c|c|c|c|c|c|}")
latex_lines.append("        \\hline")
latex_lines.append("        \\textbf{Window} & \\textbf{Stride} & \\textbf{Class} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{Support} \\\\")
latex_lines.append("        \\hline")

# Group by window size for better organization
for window in sorted(df['Window'].unique(), key=lambda x: float(x.replace('s', ''))):
    window_data = df[df['Window'] == window].sort_values(['Stride', 'Class'])
    
    latex_lines.append(f"        \\multicolumn{{7}}{{|c|}}{{\\textbf{{{window} Window Configurations}}}} \\\\")
    latex_lines.append("        \\hline")
    
    for (stride, class_name), group in window_data.groupby(['Stride', 'Class']):
        row = group.iloc[0]
        # Convert Thai text for LaTeX
        if class_name == 'เย็ด':
            latex_class = '\\textthai{เย็ด}'
        elif class_name == 'กู':
            latex_class = '\\textthai{กู}'
        elif class_name == 'มึง':
            latex_class = '\\textthai{มึง}'
        elif class_name == 'เหี้ย':
            latex_class = '\\textthai{เหี้ย}'
        else:
            latex_class = class_name
            
        latex_lines.append(f"        {window} & {stride} & {latex_class} & {row['Precision']:.1f} & {row['Recall']:.1f} & {row['F1']:.1f} & {row['Support']} \\\\")
    
    latex_lines.append("        \\hline")

latex_lines.append("    \\end{tabular}")
latex_lines.append("\\end{table}")

print("\n".join(latex_lines))

# Also create a summary table by class
print("\n" + "="*50)
print("SUMMARY LATEX TABLE BY CLASS")
print("="*50)

summary_latex = []
summary_latex.append("\\begin{table}[H]")
summary_latex.append("    \\caption{Summary of class-wise performance statistics across all configurations.}")
summary_latex.append("    \\centering")
summary_latex.append("    \\label{tab:class-performance-summary}")
summary_latex.append("    \\footnotesize")
summary_latex.append("    \\begin{tabular}{|l|c|c|c|c|}")
summary_latex.append("        \\hline")
summary_latex.append("        \\textbf{Class} & \\textbf{Precision Range (\\%)} & \\textbf{Recall Range (\\%)} & \\textbf{F1 Range (\\%)} & \\textbf{Support} \\\\")
summary_latex.append("        \\hline")

for class_name in sorted(df['Class'].unique()):
    class_data = df[df['Class'] == class_name]
    
    if class_name == 'เย็ด':
        latex_class = '\\textthai{เย็ด} (yed)'
    elif class_name == 'กู':
        latex_class = '\\textthai{กู} (guu)'
    elif class_name == 'มึง':
        latex_class = '\\textthai{มึง} (meung)'
    elif class_name == 'เหี้ย':
        latex_class = '\\textthai{เหี้ย} (hia)'
    else:
        latex_class = class_name
    
    prec_range = f"{class_data['Precision'].min():.1f}-{class_data['Precision'].max():.1f}"
    recall_range = f"{class_data['Recall'].min():.1f}-{class_data['Recall'].max():.1f}"
    f1_range = f"{class_data['F1'].min():.1f}-{class_data['F1'].max():.1f}"
    support = class_data['Support'].iloc[0]
    
    summary_latex.append(f"        {latex_class} & {prec_range} & {recall_range} & {f1_range} & {support} \\\\")

summary_latex.append("        \\hline")
summary_latex.append("    \\end{tabular}")
summary_latex.append("\\end{table}")

print("\n".join(summary_latex))

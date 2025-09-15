import pandas as pd
import numpy as np

def label_window_hybrid(window_start, window_end, profanity_segments):
    """
    Hybrid labeling approach:
    1. If any profanity word is COMPLETELY contained in window → profanity
    2. If profanity overlap >= 50% of profanity word length → profanity  
    3. Otherwise → none
    """
    for segment in profanity_segments:
        prof_start, prof_end = segment['start_time'], segment['end_time']
        prof_length = prof_end - prof_start
        
        # Check if profanity word is completely contained
        if prof_start >= window_start and prof_end <= window_end:
            return segment['label']  # Complete word in window
        
        # Check overlap percentage relative to profanity word length
        overlap_start = max(window_start, prof_start)
        overlap_end = min(window_end, prof_end)
        overlap_duration = max(0, overlap_end - overlap_start)
        
        if overlap_duration >= 0.5 * prof_length:  # >=50% of WORD captured
            return segment['label']
    
    return 'none'

def process_windows_majority(df, window_size=0.5, step_size=0.25):
    windows = []
    for file_path in df['file_path'].unique():
        file_segments = df[df['file_path'] == file_path]
        audio_end = file_segments['end_time'].max()
        start = 0
        while start < audio_end:
            end = start + window_size
            if start >= audio_end:
                break
            
            # Find all segments that overlap with this window
            overlapping = file_segments[
                ~((file_segments['end_time'] <= start) | 
                  (file_segments['start_time'] >= end))
            ]
            
            # Get profanity segments only
            profanity_segments = overlapping[overlapping['label'] != 'none']
            
            # Use hybrid labeling approach
            if len(profanity_segments) > 0:
                profanity_list = []
                for _, segment in profanity_segments.iterrows():
                    profanity_list.append({
                        'start_time': segment['start_time'],
                        'end_time': segment['end_time'],
                        'label': segment['label']
                    })
                label = label_window_hybrid(start, end, profanity_list)
            else:
                label = 'none'
            
            windows.append({
                'file_path': file_path,
                'start_time': round(start, 3),
                'end_time': round(end, 3),
                'label': label
            })
            start += step_size
    return pd.DataFrame(windows)

# Read the original CSV
df = pd.read_csv('./csv/eval_5labels.csv')

# Test with one stride first to verify hybrid approach works
print("Testing hybrid labeling approach...")
print(f"Original data: {len(df)} rows")
print(f"Original profanity instances:")
for label in ['เย็ด', 'กู', 'มึง', 'เหี้ย']:
    count = len(df[df['label'] == label])
    print(f"  {label}: {count}")

# Test with stride 0.25s
windowed_df = process_windows_majority(df, window_size=0.5, step_size=0.25)
windowed_df = windowed_df.drop_duplicates(subset=['file_path', 'start_time', 'end_time'], keep='first')
windowed_df = windowed_df.sort_values(['file_path', 'start_time']).reset_index(drop=True)

print(f"\nHybrid windowed data: {len(windowed_df)} rows")
print(f"Hybrid windowed profanity instances:")
for label in ['เย็ด', 'กู', 'มึง', 'เหี้ย']:
    count = len(windowed_df[windowed_df['label'] == label])
    print(f"  {label}: {count}")

# Save test result
windowed_df.to_csv(f'./csv/stride_0.25s_hybrid.csv', index=False)

print(f"\n✅ Hybrid approach test saved to: ./csv/eval_0.5s/stride_0.25s_hybrid.csv")

# Generate all strides if test looks good
print("Generating all stride files with hybrid approach...")
for stride in [0.36]:
    windowed_df = process_windows_majority(df, window_size=0.4, step_size=stride)
    windowed_df = windowed_df.drop_duplicates(subset=['file_path', 'start_time', 'end_time'], keep='first')
    windowed_df = windowed_df.sort_values(['file_path', 'start_time']).reset_index(drop=True)
    windowed_df.to_csv(f'./csv/eval_percent/window_0.4s/stride_{(stride/0.4)*100}%.csv', index=False)
    print(f"  Generated: stride_{stride}s.csv")

print("Hybrid windowing approach complete.")
print("Hybrid approach:")
print("1. Complete profanity words in window → labeled as profanity")
print("2. ≥50% of profanity word captured → labeled as profanity")
print("3. Otherwise → labeled as 'none'")

import pandas as pd
import numpy as np

def process_windows(df, window_size=0.5, step_size=0.25):
    windows = []
    
    # Process each audio file separately
    for file_path in df['file_path'].unique():
        file_segments = df[df['file_path'] == file_path]
        
        # Find the end time of the audio
        audio_end = file_segments['end_time'].max()
        
        # Generate windows
        start = 0
        while start < audio_end:
            end = start + window_size
            
            # Find all segments that overlap with this window
            overlapping = file_segments[
                ~((file_segments['end_time'] <= start) | 
                  (file_segments['start_time'] >= end))
            ]
            
            # Determine if there's any profanity in this window
            # If any part of the window contains profanity, label it as profanity
            label = 'none'
            for _, segment in overlapping.iterrows():
                if segment['label'] != 'none':
                    label = segment['label']
                    break
            
            windows.append({
                'file_path': file_path,
                'start_time': round(start, 3),
                'end_time': round(end, 3),
                'label': label
            })
            
            start += step_size
            
    return pd.DataFrame(windows)

# Read the original CSV
df = pd.read_csv('./csv/eval.csv')

# Process into windows
windowed_df = process_windows(df)

# Save the processed data
windowed_df.to_csv('./csv/eval_windowed.csv', index=False) 
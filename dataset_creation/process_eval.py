import pandas as pd
import numpy as np

def process_windows(df, window_size=0.5, step_size=0.25):
    windows = []
    
    # Process each audio file separately
    for file_path in df['file_path'].unique():
        file_segments = df[df['file_path'] == file_path]
        
        # Find the end time of the audio
        audio_end = file_segments['end_time'].max()
        
        # Get all profanity instances to ensure they're covered
        profanity_segments = file_segments[file_segments['label'] != 'none']
        
        # Create a set to track all window start times we need
        window_starts = set()
        
        # Regular grid windows
        start = 0
        while start < audio_end:
            window_starts.add(round(start, 3))
            start += step_size
        
        # Add windows centered on each profanity instance to ensure coverage
        for _, prof_seg in profanity_segments.iterrows():
            prof_center = (prof_seg['start_time'] + prof_seg['end_time']) / 2
            
            # Add window centered on profanity
            centered_start = prof_center - window_size / 2
            window_starts.add(round(max(0, centered_start), 3))
            
            # Add window starting at profanity start
            window_starts.add(round(max(0, prof_seg['start_time']), 3))
            
            # Add window ending at profanity end
            ending_start = prof_seg['end_time'] - window_size
            window_starts.add(round(max(0, ending_start), 3))
        
        # Generate windows for all start times
        for start in sorted(window_starts):
            end = start + window_size
            
            # Skip if window goes beyond audio
            if start >= audio_end:
                continue
                
            # Find all segments that overlap with this window
            overlapping = file_segments[
                ~((file_segments['end_time'] <= start) | 
                  (file_segments['start_time'] >= end))
            ]
            
            # Calculate profanity overlap with more lenient rules
            profanity_overlap = 0
            profanity_labels = {}  # Track overlap duration for each profanity type
            
            for _, segment in overlapping.iterrows():
                if segment['label'] != 'none':
                    # Calculate overlap between window and profanity segment
                    overlap_start = max(start, segment['start_time'])
                    overlap_end = min(end, segment['end_time'])
                    overlap_duration = max(0, overlap_end - overlap_start)
                    
                    # Track total profanity overlap and per-label overlap
                    profanity_overlap += overlap_duration
                    if segment['label'] not in profanity_labels:
                        profanity_labels[segment['label']] = 0
                    profanity_labels[segment['label']] += overlap_duration
            
            # Use very lenient rule for short profanity words: any overlap >= 0.02 seconds (20ms)
            label = 'none'
            if profanity_overlap >= 0.02:  # Much more lenient threshold
                # Choose the profanity type with the most overlap
                label = max(profanity_labels, key=profanity_labels.get)
            
            windows.append({
                'file_path': file_path,
                'start_time': round(start, 3),
                'end_time': round(end, 3),
                'label': label
            })
            
    return pd.DataFrame(windows)

# Read the original CSV
df = pd.read_csv('./csv/eval.csv')

# Process into windows
windowed_df = process_windows(df)

# Remove duplicate windows (same file, start, end times) and keep the one with profanity if any
windowed_df = windowed_df.drop_duplicates(subset=['file_path', 'start_time', 'end_time'], keep='first')

# Sort for better organization
windowed_df = windowed_df.sort_values(['file_path', 'start_time']).reset_index(drop=True)

# Save the processed data
windowed_df.to_csv('./csv/eval_windowed_0.8s.csv', index=False)

# Verification
print(f"Original profanity instances: {len(df[df['label'] != 'none'])}")
print(f"Windowed profanity instances: {len(windowed_df[windowed_df['label'] != 'none'])}")
print(f"Total windows created: {len(windowed_df)}")
print(f"Unique profanity labels in original: {set(df[df['label'] != 'none']['label'].unique())}")
print(f"Unique profanity labels in windowed: {set(windowed_df[windowed_df['label'] != 'none']['label'].unique())}")

# Check coverage for each profanity type
print("\nProfanity coverage by type:")
for label in df[df['label'] != 'none']['label'].unique():
    orig_count = len(df[df['label'] == label])
    windowed_count = len(windowed_df[windowed_df['label'] == label])
    print(f"{label}: {orig_count} -> {windowed_count} (coverage: {windowed_count/orig_count*100:.1f}%)") 
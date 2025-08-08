import pandas as pd
import numpy as np

def process_windows_majority(df, window_size=0.5, step_size=0.7):
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
            # Calculate overlap duration for each label
            label_durations = {}
            for _, segment in overlapping.iterrows():
                overlap_start = max(start, segment['start_time'])
                overlap_end = min(end, segment['end_time'])
                overlap_duration = max(0, overlap_end - overlap_start)
                if overlap_duration > 0:
                    label = segment['label']
                    if label not in label_durations:
                        label_durations[label] = 0
                    label_durations[label] += overlap_duration
            # Assign label if any profanity label covers >= 50% of window
            label = 'none'
            for lbl, dur in label_durations.items():
                if lbl != 'none' and dur >= 0.5 * window_size:
                    label = lbl
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

# Example usage for one stride (repeat for 0.25 to 0.7)
for stride in [0.25, 0.3, 0.4, 0.5, 0.6, 0.7]:
    windowed_df = process_windows_majority(df, window_size=0.5, step_size=stride)
    windowed_df = windowed_df.drop_duplicates(subset=['file_path', 'start_time', 'end_time'], keep='first')
    windowed_df = windowed_df.sort_values(['file_path', 'start_time']).reset_index(drop=True)
    windowed_df.to_csv(f'./csv/eval_windowed_majority_{stride}s.csv', index=False)

print("Majority overlap window labeling complete for strides 0.25 to 0.7.")

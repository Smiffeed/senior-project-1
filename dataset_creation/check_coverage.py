import pandas as pd

# Read both datasets
df_orig = pd.read_csv('csv/eval.csv')
df_windowed = pd.read_csv('csv/eval_windowed_0.3s.csv')

print("=== COMPARISON RESULTS ===")
print(f"Original eval.csv:")
print(f"  Total rows: {len(df_orig)}")
print(f"  Profanity instances: {len(df_orig[df_orig['label'] != 'none'])}")

print(f"\nUpdated eval_windowed_0.3s.csv:")
print(f"  Total rows: {len(df_windowed)}")
print(f"  Profanity instances: {len(df_windowed[df_windowed['label'] != 'none'])}")

improvement = len(df_windowed[df_windowed['label'] != 'none']) - len(df_orig[df_orig['label'] != 'none'])
print(f"\nImprovement: {improvement} more profanity windows")
print(f"Coverage ratio: {len(df_windowed[df_windowed['label'] != 'none']) / len(df_orig[df_orig['label'] != 'none']):.2f}x")

print(f"\n=== LABEL DISTRIBUTION ===")
print("Original:")
for label in sorted(df_orig[df_orig['label'] != 'none']['label'].value_counts().index):
    count = len(df_orig[df_orig['label'] == label])
    print(f"  {label}: {count}")

print("\nWindowed:")
for label in sorted(df_windowed[df_windowed['label'] != 'none']['label'].value_counts().index):
    count = len(df_windowed[df_windowed['label'] == label])
    print(f"  {label}: {count}")

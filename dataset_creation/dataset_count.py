import pandas as pd

# Load the CSV file
df = pd.read_csv('./csv/main.csv')

# Count the occurrences of each word
word_counts = df['label'].value_counts()

# Count profanity vs non-profanity
profanity_count = df[df['label'] != 'none'].shape[0]
non_profanity_count = df[df['label'] == 'none'].shape[0]

# Count the number of unique datasets
dataset_counts = df['file_path'].nunique()

# Calculate the total time for each word
df['Duration'] = df['end_time'] - df['start_time']
total_time_per_word = df.groupby('label')['Duration'].sum()

print("Word Counts:")
print(word_counts)
print("\nProfanity vs Non-Profanity:")
print(f"Profanity instances: {profanity_count}")
print(f"Non-profanity instances: {non_profanity_count}")
print(f"Total instances: {profanity_count + non_profanity_count}")
print("\nNumber of Unique Datasets:")
print(dataset_counts)
print("\nTotal Time for Each Word:")
print(total_time_per_word)

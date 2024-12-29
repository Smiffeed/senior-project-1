import pandas as pd

# Load the CSV file
df = pd.read_csv('profanity_dataset_word.csv')

# Count the occurrences of each word
word_counts = df['Label'].value_counts()

# Count the number of unique datasets
dataset_counts = df['File Name'].nunique()

# Calculate the total time for each word
df['Duration'] = df['End Time (s)'] - df['Start Time (s)']
total_time_per_word = df.groupby('Label')['Duration'].sum()

print("Word Counts:")
print(word_counts)
print("\nNumber of Unique Datasets:")
print(dataset_counts)
print("\nTotal Time for Each Word:")
print(total_time_per_word)

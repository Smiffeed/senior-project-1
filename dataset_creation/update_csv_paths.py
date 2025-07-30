import pandas as pd

# Define the path to the CSV file
csv_file_path = 'csv/main.csv'

# Read the CSV file into a pandas DataFrame
try:
    df = pd.read_csv(csv_file_path)

    # Perform the string replacement on all columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(r'.\\main\\', './main/', regex=True)

    # Write the updated DataFrame back to the CSV file
    df.to_csv(csv_file_path, index=False)

    print(f"Successfully updated paths in {csv_file_path}")

except FileNotFoundError:
    print(f"Error: The file {csv_file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

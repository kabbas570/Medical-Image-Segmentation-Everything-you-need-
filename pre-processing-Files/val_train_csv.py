import pandas as pd

# Assuming csv_1 and csv_2 are the filenames of your CSV files
csv_1 = r"C:\My_Data\M2M Data\data\train.csv"
csv_2 = r"C:\My_Data\M2M Data\data\data_2\five_fold\sep\new_split_mix\F1\val\F1_val.csv"

# Read the CSV files into pandas DataFrames
df1 = pd.read_csv(csv_1)
df2 = pd.read_csv(csv_2)

# Find rows in df1 that are not in df2 based on a specified column (adjust column name as needed)
df3 = pd.merge(df1, df2, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)

# Write the result to csv_3
csv_3 = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\new_split_mix\F1\val/csv_3.csv'
df3.to_csv(csv_3, index=False)

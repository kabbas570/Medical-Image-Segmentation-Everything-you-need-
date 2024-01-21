import numpy as np
import pandas as pd

# Replace the following lines with your actual data
# Example arrays
array1 = np.random.randint(1, 6, size=(4, 4))


# Simulate missing and extra values
array1[2:4, 2:4] = 0


# Create a Pandas DataFrame
columns = ['Image','Grade-2 %', 'Grade-3 %', 'Grade-4 %', 'Grade-5 %', 'Grade-6 %',  'Total Rust %', 'Pixels > 5']
df = pd.DataFrame(columns=columns)

# Iterate through each array
for i, array in enumerate([array1], start=1):
    # Count the number of pixels for each label
    label_counts = np.bincount(array.flatten(), minlength=6)[1:]
    
    

    # Calculate the percentage of pixels for each label
    total_pixels = array.size
    percentages = (label_counts / np.sum(label_counts)) * 100

    # Count the total number of pixels greater than 5
    pixels_gt_5 = np.sum(array > 5)
    
    rust_percntg = np.sum(label_counts) / total_pixels * 100

    # Append the percentages to the DataFrame
    row_data = ['image_56']  + list(percentages) + [rust_percntg, pixels_gt_5]
    df.loc[f'Array {i}'] = row_data

# Save the DataFrame to a CSV file
df.to_csv('label_percentages.csv', index_label='Array')

output_file_path = r'C:\Users\Abbas Khan\Downloads/dataframe_file.csv'

# Write the DataFrame to a CSV file (change the format as needed)
df.to_csv(output_file_path, index=False)

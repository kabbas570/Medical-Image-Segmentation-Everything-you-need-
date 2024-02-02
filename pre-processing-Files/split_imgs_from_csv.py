import pandas as pd
import os
import shutil

csv_1 = r"C:\My_Data\M2M Data\data\data_2\five_fold\sep\new_split_mix\F1\train\F1_train.csv"
df1 = pd.read_csv(csv_1)
n = df1['SUBJECT_CODE']

file_names_to_copy = []
for i in n:
    k = str(i).zfill(3)
    print(k)
    k1 = k + '_LA_ED.nii.gz'
    k2 = k + '_LA_ES.nii.gz'
    file_names_to_copy.append(k1)
    file_names_to_copy.append(k2)
    

source_folder = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\all_train\imgs'
destination_folder = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\new_split_mix\F1\train\imgs'
os.makedirs(destination_folder, exist_ok=True)

for file_name in file_names_to_copy:
    source_path = os.path.join(source_folder, file_name)
    destination_path = os.path.join(destination_folder, file_name)
    
    # Check if the file exists before copying
    if os.path.isfile(source_path):
        shutil.copy2(source_path, destination_path)
        print(f"File '{file_name}' copied successfully.")
    else:
        print(f"File '{file_name}' not found in the source folder.")

print("Copying completed.")


file_names_to_copy = []
for i in n:
    k = str(i).zfill(3)
    print(k)
    k1 = k + '_LA_ED_gt.nii.gz'
    k2 = k + '_LA_ES_gt.nii.gz'
    file_names_to_copy.append(k1)
    file_names_to_copy.append(k2)
    

source_folder = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\all_train\gts'
destination_folder = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\new_split_mix\F1\train\gts'
os.makedirs(destination_folder, exist_ok=True)

for file_name in file_names_to_copy:
    source_path = os.path.join(source_folder, file_name)
    destination_path = os.path.join(destination_folder, file_name)
    
    # Check if the file exists before copying
    if os.path.isfile(source_path):
        shutil.copy2(source_path, destination_path)
        print(f"File '{file_name}' copied successfully.")
    else:
        print(f"File '{file_name}' not found in the source folder.")

print("Copying completed.")

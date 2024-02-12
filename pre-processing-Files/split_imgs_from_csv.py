import pandas as pd
import os
import shutil


source_folder_img = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\SA_Data\all_train\imgs'
source_folder_gt = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\SA_Data\all_train\gts'

fold = str(1)

csv_1 = r"C:\My_Data\M2M Data\data\data_2\five_fold\sep\SA_Data\F" + fold + "/train\F" + fold + "_train.csv"
destination_folder_img = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\SA_Data\F' + fold + '/'+ 'train\imgs'
destination_folder_gt = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\SA_Data\F' + fold + '/'+ 'train\gts'




df1 = pd.read_csv(csv_1)
n = df1['SUBJECT_CODE']

img_file_names_to_copy = []
for i in n:
    k = str(i).zfill(3)
    k1 = k + '_SA_ED.nii.gz'
    k2 = k + '_SA_ES.nii.gz'
    img_file_names_to_copy.append(k1)
    img_file_names_to_copy.append(k2)
    
os.makedirs(destination_folder_img, exist_ok=True)
for file_name in img_file_names_to_copy:
    source_path = os.path.join(source_folder_img, file_name)
    destination_path = os.path.join(destination_folder_img, file_name)
    
    # Check if the file exists before copying
    if os.path.isfile(source_path):
        shutil.copy2(source_path, destination_path)
        print(f"File '{file_name}' copied successfully.")
    else:
        print(f"File '{file_name}' not found in the source folder.")

print("Copying completed.")




gt_file_names_to_copy = []
for i in n:
    k = str(i).zfill(3)
    print(k)
    k1 = k + '_SA_ED_gt.nii.gz'
    k2 = k + '_SA_ES_gt.nii.gz'
    gt_file_names_to_copy.append(k1)
    gt_file_names_to_copy.append(k2)
    
    
os.makedirs(destination_folder_gt, exist_ok=True)

for file_name in gt_file_names_to_copy:
    source_path = os.path.join(source_folder_gt, file_name)
    destination_path = os.path.join(destination_folder_gt, file_name)
    
    # Check if the file exists before copying
    if os.path.isfile(source_path):
        shutil.copy2(source_path, destination_path)
        print(f"File '{file_name}' copied successfully.")
    else:
        print(f"File '{file_name}' not found in the source folder.")

print("Copying completed.")

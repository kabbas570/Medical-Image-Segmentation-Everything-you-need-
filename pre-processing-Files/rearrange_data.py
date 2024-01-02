import os
import shutil

def copy_files_with_suffix(source_dir, dest_dir, suffix):
    # Iterate through folders in the source directory
    for root, dirs, files in os.walk(source_dir):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            files_in_folder = os.listdir(folder_path)

            # Find .nii.gz files with the specified suffix
            matching_files = [f for f in files_in_folder if f.endswith('.nii.gz') and f.endswith(suffix + '.nii.gz')]

            if matching_files:
                # Copy matching files to the destination folder
                for file in matching_files:
                    src_file = os.path.join(folder_path, file)
                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy(src_file, dest_file)


fold = str(2)

split = 'val'

source_directory1 = r'C:\My_Data\M2M Data\data\data_2\five_fold\F'+fold+'/'+ split

# Example usage:
source_directory = r'C:\My_Data\M2M Data\data\data_2\five_fold\F'+fold+'/' + split
destination_directory = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\F'+fold+'/val\imgs'
file_suffix = '_LA_ES'  # Replace with your desired suffix
copy_files_with_suffix(source_directory, destination_directory, file_suffix)

# Example usage:
source_directory = r'C:\My_Data\M2M Data\data\data_2\five_fold\F'+fold+'/' + split
destination_directory = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\F'+fold+'/val\imgs'
file_suffix = '_LA_ED'  # Replace with your desired suffix
copy_files_with_suffix(source_directory, destination_directory, file_suffix)


# Example usage:
source_directory = r'C:\My_Data\M2M Data\data\data_2\five_fold\F'+fold+'/' + split
destination_directory = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\F'+fold+'/val\gts'
file_suffix = '_LA_ES_gt'  # Replace with your desired suffix
copy_files_with_suffix(source_directory, destination_directory, file_suffix)

# Example usage:
source_directory = r'C:\My_Data\M2M Data\data\data_2\five_fold\F'+fold+'/' + split
destination_directory = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\F'+fold+'/val\gts'
file_suffix = '_LA_ED_gt'  # Replace with your desired suffix
copy_files_with_suffix(source_directory, destination_directory, file_suffix)


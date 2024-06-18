import os
import pandas as pd

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import SimpleITK as sitk
import os
import torch
import matplotlib.pyplot as plt
import os

NUM_WORKERS=0
PIN_MEMORY=True



def one_hot(df):
    
    df['Sex'] = df['Sex'].replace({'Male': 1, 'Female': 0})
    df['RaceC'] = df['RaceC'].replace({'African':0, 'White/Europid' :1, 'Mixed/other' :2,'South Asian':3, 'Oriental':4})
    df['SMOKER'] = df['SMOKER'].replace({'Smoker':1 ,'No smoker':0})
    df['Vascular'] = df['Vascular'].replace({'Yes':1, 'No':0})
    df['Coronary'] = df['Coronary'].replace({'Yes':1, 'No':0})
    df['DIABSUBE'] = df['DIABSUBE'].replace({'Yes':1, 'No':0})
    
    return df
    
class Dataset_val(Dataset): 
    def __init__(self, csv_folder):
        self.csv_folder = csv_folder
        self.files_name = os.listdir(csv_folder)
    def __len__(self):
       return len(self.files_name)
    def __getitem__(self, index):

        path = os.path.join(self.csv_folder,str(self.files_name[index]))        
        df = pd.read_csv(path)
        
        extracted_columns = df[['visit_1', 'visit_2', 'visit_3', 'visit_4']]
        extracted_columns = extracted_columns.replace({'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0, 'African':0, 'White/Europid' :1, 'Mixed/other' :2,'South Asian':3, 'Oriental':4})
        extracted_columns = extracted_columns.apply(pd.to_numeric, errors='coerce').fillna(0)
        numpy_array = extracted_columns.to_numpy()        
        tensor = torch.tensor(numpy_array, dtype=torch.float32)
    
        x = tensor[:-1]  # Selecting all rows except the last one
        y = tensor[-1]   # Selecting the last row

        return tensor,x,y
        
def Data_Loader(csv_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_val(csv_folder=csv_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

csv_path = r'C:\My_Data\BRC_Project\data\workstream_3\data_csv\data1/' 
train_loader = Data_Loader(csv_path,batch_size = 2)

a = iter(train_loader)
a1 = next(a) 
inp = a1[0].numpy()
x = a1[1].numpy()
y = a1[2].numpy()






# df = pd.read_csv(r'C:\My_Data\BRC_Project\data\workstream_3\data_csv\data1/50018-00233.csv')
# extracted_columns = df[['visit_1', 'visit_2', 'visit_3', 'visit_4']]
# extracted_columns = extracted_columns.replace({'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0})
# extracted_columns = extracted_columns.replace({'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0, 'African':0, 'White/Europid' :1, 'Mixed/other' :2,'South Asian':3, 'Oriental':4})

# numpy_array = extracted_columns.to_numpy()
# print(type(numpy_array))  # This will show the type to confirm it's a NumPy array



# import pandas as pd

# # Sample DataFrame for demonstration
# data = {
#     'visit_1': [1, 2, 3],
#     'visit_2': [4, 5, 6],
#     'visit_3': [7, 8, 9],
#     'visit_4': [10, 11, 12],
#     'other_column': [13, 14, 15]
# }

# df = pd.DataFrame(data)

# # Extract specific columns
# extracted_columns = df[['visit_1', 'visit_2', 'visit_3', 'visit_4']]

# # Convert to numpy array
# numpy_array = extracted_columns.to_numpy()

# print(numpy_array)
# print(type(numpy_array))  # This will show the type to confirm it's a NumPy array

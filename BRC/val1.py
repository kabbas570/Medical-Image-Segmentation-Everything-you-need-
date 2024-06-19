import torch.nn as nn
import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import csv

# Define the RNN models
class RNN1(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, output_size=1):
        super(RNN1, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size, bias=True)
        self.h2o = nn.Linear(hidden_size, output_size, bias=True)
        self.act = nn.ReLU()

    def forward(self, input):
        hidden = self.act(self.i2h(input))
        output = self.h2o(hidden)
        output = self.act(output)
        return output, hidden

class RNN2(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, output_size=1):
        super(RNN2, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size, bias=True)
        self.h2o = nn.Linear(hidden_size, output_size, bias=True)
        self.act = nn.ReLU()

    def forward(self, input, hidden):
        hidden = self.act(self.i2h(input)) + hidden
        output = self.h2o(hidden)
        output = self.act(output)
        return output, hidden

class RNN3(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, output_size=1):
        super(RNN3, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size, bias=True)
        self.h2o = nn.Linear(2 * hidden_size, output_size, bias=True)
        self.act = nn.ReLU()

    def forward(self, input, hidden):
        hidden_new = self.act(self.i2h(input))
        hidden_ = torch.cat([hidden_new, hidden], dim=2)
        output = self.h2o(hidden_)
        return output, hidden

class Net(nn.Module):
    def __init__(self, n_channels=1):
        super(Net, self).__init__()
        self.model1 = RNN1()
        self.model2 = RNN2()
        self.model3 = RNN2()
        self.model4 = RNN2()

    def forward(self, x):
        inp1 = x[:, 0:1, :]
        inp2 = x[:, 1:2, :]
        inp3 = x[:, 2:3, :]
        inp4 = x[:, 3:4, :]

        out1, h1 = self.model1(inp1)
        out2, h2 = self.model2(inp2, h1)
        out3, h3 = self.model3(inp3, h2)
        out4, _ = self.model4(inp4, h3)

        return out1, out2, out3, out4

# Dataset class
class Dataset_val(Dataset):
    def __init__(self, csv_folder):
        self.csv_folder = csv_folder
        self.files_name = os.listdir(csv_folder)
        self.scaler = MinMaxScaler()

        # Pre-fit the scaler
        self._fit_scaler()

    def _fit_scaler(self):
        all_data = []
        for file_name in self.files_name:
            path = os.path.join(self.csv_folder, file_name)
            df = pd.read_csv(path)
            extracted_columns = df[['visit_1', 'visit_2', 'visit_3', 'visit_4']]
            extracted_columns = extracted_columns.replace({
                'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0,
                'African': 0, 'White/Europid': 1, 'Mixed/other': 2,
                'South Asian': 3, 'Oriental': 4
            })
            extracted_columns = extracted_columns.apply(pd.to_numeric, errors='coerce').fillna(0)
            all_data.append(extracted_columns.to_numpy())
        all_data = np.concatenate(all_data, axis=0)
        self.scaler.fit(all_data)

    def __len__(self):
        return len(self.files_name)

    def __getitem__(self, index):
        path = os.path.join(self.csv_folder, str(self.files_name[index]))
        df = pd.read_csv(path)
        extracted_columns = df[['visit_1', 'visit_2', 'visit_3', 'visit_4']]
        extracted_columns = extracted_columns.replace({
            'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0,
            'African': 0, 'White/Europid': 1, 'Mixed/other': 2,
            'South Asian': 3, 'Oriental': 4
        })
        extracted_columns = extracted_columns.apply(pd.to_numeric, errors='coerce').fillna(0)
        numpy_array = extracted_columns.to_numpy()

        # Normalize data
        normalized_array = self.scaler.transform(numpy_array)
        tensor = torch.tensor(normalized_array, dtype=torch.float32)

        x = tensor[:-1]
        y = tensor[-1]

        y = torch.unsqueeze(y, 0)
        x = x.transpose(0, 1)

        return x, y

def Data_Loader(csv_folder, batch_size, num_workers=0, pin_memory=True):
    dataset = Dataset_val(csv_folder=csv_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    return data_loader

# Error checking function
def check_error(loader, model, device, scaler):
    model.eval()
    overall_error = 0
    error_1 = 0
    error_2 = 0
    error_3 = 0
    error_4 = 0
    
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['y4', 'p4'])

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)

            p1, p2, p3, p4 = model(x)
            y = y[0,:]
            y5_orig = scaler.inverse_transform(y).reshape(y.shape)
            
            print(p1.shape)
            p = torch.cat([p1,p2,p3,p4], dim=2)
            p = p[0,:]
            print(p.shape)
            print(y5_orig.shape)
            
            p5_pred = scaler.inverse_transform(p).reshape(p.shape)
            
            y1, y2, y3, y4 = y5_orig[:, 0:1], y5_orig[:,  1:2], y5_orig[:,  2:3], y5_orig[:, 3:4]
            p1, p2, p3, p4 = p5_pred[:, 0:1], p5_pred[:,  1:2], p5_pred[:,  2:3], p5_pred[:, 3:4]
            
           
            
            # Ensure the correct shape for inverse transformation
            #y4_np = y4.cpu().numpy().reshape(-1, 1)
            #/p4_np = p4.cpu().numpy().reshape(-1, 1)
            #y = y[0,:]
            #y5_orig = scaler.inverse_transform(y).reshape(y.shape)
            #print(y5_orig.shape)
            
            
            #y4_orig = scaler.inverse_transform(y4_np).reshape(p4.shape)
            #p4_orig = scaler.inverse_transform(p4_np).reshape(p4.shape)
            
            
            # error_1 += torch.sum(torch.abs(p1 - y1)).item()
            # error_2 += torch.sum(torch.abs(p2 - y2)).item()
            # error_3 += torch.sum(torch.abs(p3 - y3)).item()
            # error_4 += torch.sum(torch.abs(p5_pred - y5_orig)).item()
            
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                for y_val, p_val in zip(y4.flatten(), p4.flatten()):
                    writer.writerow([y_val, p_val])

    overall_error = (error_1 + error_2 + error_3 + error_4) / 4
    print(f"Error_1  : {error_1/len(loader)}")
    print(f"Error_2  : {error_2/len(loader)}")
    print(f"Error_3  : {error_3/len(loader)}")
    print(f"Error_4  : {error_4/len(loader)}")
    print(f"Overall_Error  : {overall_error/len(loader)}")

    return overall_error / len(loader)

# Evaluation function
def eval_():
    model = model_1.to(device=DEVICE, dtype=torch.float)
    optimizer = optim.AdamW(model.parameters(), betas=(0.5, 0.5), lr=0.0001)
    
    checkpoint = torch.load(path_to_checkpoints, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    loss = check_error(val_loader, model, DEVICE, dataset.scaler)

# Paths and configurations
csv_path_val = r'C:\My_Data\BRC_Project\data\workstream_3\data_csv\data1/'
val_loader = Data_Loader(csv_path_val, batch_size=1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
path_to_checkpoints = r"C:\My_Data\BRC_Project\data\workstream_3\data_csv\Model1.pth.tar"
csv_file_path = os.path.join(r'C:\My_Data\BRC_Project\data\workstream_3\data_csv/', 'output_G23.csv')

model_1 = Net()
dataset = Dataset_val(csv_path_val)

if __name__ == "__main__":
    eval_()

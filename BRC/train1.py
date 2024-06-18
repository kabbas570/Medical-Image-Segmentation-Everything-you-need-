import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

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
        self.h2o = nn.Linear(2*hidden_size, output_size, bias=True)
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
            extracted_columns = extracted_columns.replace({'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0, 'African': 0, 'White/Europid': 1, 'Mixed/other': 2, 'South Asian': 3, 'Oriental': 4})
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
        extracted_columns = extracted_columns.replace({'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0, 'African': 0, 'White/Europid': 1, 'Mixed/other': 2, 'South Asian': 3, 'Oriental': 4})
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

csv_path_train = r'C:\My_Data\BRC_Project\data\workstream_3\data_csv\data1/'
csv_path_val = r'C:\My_Data\BRC_Project\data\workstream_3\data_csv\data1/'

train_loader = Data_Loader(csv_path_train, batch_size=2)
val_loader = Data_Loader(csv_path_val, batch_size=1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.MSELoss()

def train_fn(loader_train, loader_valid, model, optimizer, scaler, scheduler):
    train_losses = []
    valid_losses = []

    model.train()
    loop = tqdm(loader_train, desc='Training')

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(device=DEVICE, dtype=torch.float)
        y = y.to(device=DEVICE, dtype=torch.float)

        with torch.cuda.amp.autocast():
            p1, p2, p3, p4 = model(x)
            y1, y2, y3, y4 = y[:, :, 0:1], y[:, :, 1:2], y[:, :, 2:3], y[:, :, 3:4]

            l1 = loss_fn(p1, y1)
            l2 = loss_fn(p2, y2)
            l3 = loss_fn(p3, y3)
            l4 = loss_fn(p4, y4)

            loss = (l1 + l2 + l3 + l4) / 4

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
        train_losses.append(loss.item())

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    model.eval()
    loop_v = tqdm(loader_valid, desc='Validation')

    with torch.no_grad():
        for x, y in loop_v:
            x = x.to(device=DEVICE, dtype=torch.float)
            y = y.to(device=DEVICE, dtype=torch.float)

            p1, p2, p3, p4 = model(x)
            y1, y2, y3, y4 = y[:, :, 0:1], y[:, :, 1:2], y[:, :, 2:3], y[:, :, 3:4]

            l1 = loss_fn(p1, y1)
            l2 = loss_fn(p2, y2)
            l3 = loss_fn(p3, y3)
            l4 = loss_fn(p4, y4)

            loss = (l1 + l2 + l3 + l4) / 4

            loop_v.set_postfix(loss=loss.item())
            valid_losses.append(loss.item())

    return np.mean(train_losses), np.mean(valid_losses)

def check_error(loader, model, device="cuda"):
    model.eval()
    Overall_Error = 0
    Error_1 = 0
    Error_2 = 0
    Error_3 = 0
    Error_4 = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=DEVICE, dtype=torch.float)
            y = y.to(device=DEVICE, dtype=torch.float)

            p1, p2, p3, p4 = model(x)
            y1, y2, y3, y4 = y[:, :, 0:1], y[:, :, 1:2], y[:, :, 2:3], y[:, :, 3:4]

            Error_1 += torch.sum(torch.abs(p1 - y1)).item()
            Error_2 += torch.sum(torch.abs(p2 - y2)).item()
            Error_3 += torch.sum(torch.abs(p3 - y3)).item()
            Error_4 += torch.sum(torch.abs(p4 - y4)).item()

    Overall_Error = (Error_1 + Error_2 + Error_3 + Error_4) / 4
    print(f"Error_1  : {Error_1/len(loader)}")
    print(f"Error_2  : {Error_2/len(loader)}")
    print(f"Error_3  : {Error_3/len(loader)}")
    print(f"Error_4  : {Error_4/len(loader)}")
    print(f"Overall_Error  : {Overall_Error/len(loader)}")

    return Overall_Error / len(loader)

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def main():
    max_dice_val = float('inf')
    model = Net().to(DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.AdamW(model.parameters(), betas=(0.5, 0.5), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200, 400, 500], gamma=0.5)

    avg_train_losses_seg = []
    avg_valid_losses_seg = []
    avg_valid_DS_ValSet_seg = []

    path_to_save_checkpoints = r'C:\My_Data\BRC_Project\data\workstream_3\data_csv\Model1.pth.tar'

    for epoch in range(100):
        train_loss_seg, valid_loss_seg = train_fn(train_loader, val_loader, model, optimizer, scaler, scheduler)
        scheduler.step()

        print(f'[{epoch+1:03d}/100] train_loss_seg: {train_loss_seg:.5f} valid_loss_seg: {valid_loss_seg:.5f}')

        Dice_val = check_error(val_loader, model, device=DEVICE)
        avg_valid_DS_ValSet_seg.append(Dice_val)

        if Dice_val < max_dice_val:
            max_dice_val = Dice_val
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=path_to_save_checkpoints)

        avg_train_losses_seg.append(train_loss_seg)
        avg_valid_losses_seg.append(valid_loss_seg)

    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(avg_train_losses_seg) + 1), avg_train_losses_seg, label='Training Loss')
    plt.plot(range(1, len(avg_valid_losses_seg) + 1), avg_valid_losses_seg, label='Validation Loss')
    plt.plot(range(1, len(avg_valid_DS_ValSet_seg) + 1), avg_valid_DS_ValSet_seg, label='Validation Error')
    plt.axvline(avg_valid_losses_seg.index(min(avg_valid_losses_seg)) + 1, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.title("Learning Curve Graph", fontsize=20)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(min(min(avg_train_losses_seg), min(avg_valid_losses_seg)) - 0.1, max(max(avg_train_losses_seg), max(avg_valid_losses_seg)) + 0.1)
    plt.xlim(1, len(avg_train_losses_seg))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(r'C:\My_Data\BRC_Project\data\workstream_3\data_csv\Model1_Learning_Curve.png')
    plt.show()

if __name__ == "__main__":
    main()

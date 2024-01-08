import torch
import torch.nn as nn
import matplotlib.pyplot as plt
      
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(mid_channels),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01 , inplace=True),
            
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01 , inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv_s2(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1,stride=2),
            #nn.BatchNorm2d(mid_channels),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01 , inplace=True),
            
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01 , inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_s2 = nn.Sequential(
            DoubleConv_s2(in_channels, out_channels)
        )

    def forward(self, x):
        return self.conv_s2(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class BaseLine2(nn.Module):
    def __init__(self, n_channels = 1):
        super(BaseLine2, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        
        self.up0 = Up(1024, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        
        
        self.up0_ = Up(512, 256)
        self.up1_ = Up(256, 128)
        self.up2_ = Up(128, 64)
        self.up3_ = Up(64, 32)
        self.up4_ = Up(64, 32)
        
        
        self.outc = OutConv(32,2)
        self.outc4 = OutConv(32,4)
        
        self.dropout1 = nn.Dropout2d(p=0.30)  
        self.dropout2 = nn.Dropout2d(p=0.30)  

        
    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) 
        x5 = self.dropout1(x5)
        x6 = self.down5(x5) 
        x6 = self.dropout2(x6)
        
        z1 = self.up0(x6, x5)
        z2 = self.up1(z1, x4)
        z3 = self.up2(z2, x3)
        z4 = self.up3(z3, x2)
        z5 = self.up4(z4, x1)

        logits1 = self.outc(z5)
        
        y1 = self.up0_(z1, z2)
        y2 = self.up1_(y1, z3)
        y3 = self.up2_(y2, z4)
        y4 = self.up3_(y3, z5)
        
        logits2 = self.outc4(y4)
    
        return logits1,logits2
    
# Input_Image_Channels = 1
# def model() -> BaseLine2:
#     model = BaseLine2()
#     return model
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# from torchsummary import summary
# model = model()
# model.to(device=DEVICE,dtype=torch.float)
# summary(model, [(Input_Image_Channels, 256,256)])



import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import pandas as pd
import SimpleITK as sitk
#from typing import List, Union, Tuple

import torchio as tio
           ###########  Dataloader  #############
NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 256
   
def Generate_Meta_Data(vendors_): 
    temp = np.zeros([8,8])
    if vendors_=='GE MEDICAL SYSTEMS': 
        temp[0:4,0:4] = 0.8
        temp[0:4,4:9] = 1
        temp[4:9,0:4] = 0.7
        temp[4:9,4:9] = 0.9 ### m = 0.85 and std = 0.11
    if vendors_=='SIEMENS':
        temp[0:4,0:4] = 0.1
        temp[0:4,4:9] = 0.0
        temp[4:9,0:4] = 1.0
        temp[4:9,4:9] = 0.9 ### m = 0.5 and std = 0.45
    if vendors_=='Philips Medical Systems':
        temp[0:4,0:4] = 0.1
        temp[0:4,4:9] = 0.3
        temp[4:9,0:4] = 0.0
        temp[4:9,4:9] = 0.7 ### m = 0.275 and std = 0.268
    return temp
           
     
def resample_itk_image_LA(itk_image):
    # Get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    out_spacing = (1,1,1)

    # Calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    # Instantiate resample filter with properties
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    # Execute resampling
    resampled_image = resample.Execute(itk_image)
    return resampled_image


    
def crop_center_3D(img,cropx=DIM_,cropy=DIM_):
    z,x,y = img.shape
    startx = x//2 - cropx//2
    starty = (y)//2 - cropy//2    
    return img[:,startx:startx+cropx, starty:starty+cropy]

def Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_):# org_dim3->numof channels
    
    if org_dim1<DIM_ and org_dim2<DIM_:
        padding1=int((DIM_-org_dim1)//2)
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,padding1:org_dim1+padding1,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = temp
    if org_dim1>DIM_ and org_dim2>DIM_:
        img_ = crop_center_3D(img_)        
        ## two dims are different ####
    if org_dim1<DIM_ and org_dim2>=DIM_:
        padding1=int((DIM_-org_dim1)//2)
        temp=np.zeros([org_dim3,DIM_,org_dim2])
        temp[:,padding1:org_dim1+padding1,:] = img_[:,:,:]
        img_=temp
        img_ = crop_center_3D(img_)
    if org_dim1==DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_=temp
    
    if org_dim1>DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,org_dim1,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = crop_center_3D(temp)   
    return img_


def Normalization_1(img):
 return (img - 112.5312) / 143.1712

def Normalization_2(img):
        mean=np.mean(img)
        std=np.std(img)
        img=(img-mean)/std
        return img 

def normalize_P(img):
 return (img - 72.8425) / 91.4957

def normalize_G(img):
    return (img - 197.8624) / 284.7200

def normalize_S(img):
    return (img - 117.1235) / 137.8422

geometrical_transforms = tio.OneOf([
    tio.RandomFlip(axes=([1, 2])),
    tio.RandomElasticDeformation(num_control_points=(5, 5, 5), locked_borders=1, image_interpolation='nearest'),
    tio.RandomAffine(degrees=(-45, 45), center='image'),
])

intensity_transforms = tio.OneOf([
    tio.RandomBlur(),
    tio.RandomGamma(log_gamma=(-0.2, -0.2)),
    tio.RandomNoise(mean=0.1, std=0.1),
    tio.RandomGhosting(axes=([1, 2])),
])

transforms_2d = tio.Compose({
    geometrical_transforms: 0.3,  # Probability for geometric transforms
    intensity_transforms: 0.3,   # Probability for intensity transforms
    tio.Lambda(lambda x: x): 0.4 # Probability for no augmentation (original image)
})
   
def generate_label(gt):
        temp_ = np.zeros([4,DIM_,DIM_])
        temp_[0:1,:,:][np.where(gt==1)]=1
        temp_[1:2,:,:][np.where(gt==2)]=1
        temp_[2:3,:,:][np.where(gt==3)]=1
        temp_[3:4,:,:][np.where(gt==0)]=1
        return temp_
 
class Dataset_io(Dataset): 
    def __init__(self, df, images_folder,transformations=transforms_2d):  ## If I apply Data Augmentation here, the validation loss becomes None. 
        self.df = df
        self.images_folder = images_folder
        self.gt_folder = self.images_folder[:-5] + 'gts'
        self.vendors = df['VENDOR']       
        self.images_name1 = df['SUBJECT_CODE'] 
        self.transformations = transformations
        
        self.images_name = os.listdir(images_folder)
        
    def __len__(self):
       return len(self.images_name)
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder,str(self.images_name[index]).zfill(3))
        img = sitk.ReadImage(img_path)    ## --> [H,W,C]
        img = resample_itk_image_LA(img)      ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]

        number = int(self.images_name[index][:3])
        ind = self.df['SUBJECT_CODE'].loc[lambda x: x==number].index[0]
        vendor = self.vendors[ind]
        
        if vendor=='GE MEDICAL SYSTEMS': 
            img = normalize_G(img)

        if vendor=='SIEMENS':
            img = normalize_S(img)

        if vendor=='Philips Medical Systems':
            img = normalize_P(img)
            
        C = img.shape[0]
        H = img.shape[1]
        W = img.shape[2]
        img = Cropping_3d(C,H,W,256,img)
        img = np.expand_dims(img, axis=3)
        
        gt_path = os.path.join(self.gt_folder,str(self.images_name[index]).zfill(3))
        gt_path = gt_path[:-7]+'_gt.nii.gz'
        gt = sitk.ReadImage(gt_path)    ## --> [H,W,C]
        gt = resample_itk_image_LA(gt)      ## --> [H,W,C]
        gt = sitk.GetArrayFromImage(gt)   ## --> [C,H,W]
        C = gt.shape[0]
        H = gt.shape[1]
        W = gt.shape[2]
        gt = Cropping_3d(C,H,W,256,gt)
        gt = np.expand_dims(gt, axis=3)
        
        d = {}
        d['Image'] = tio.Image(tensor = img, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = gt, type=tio.LABEL)
        sample = tio.Subject(d)
        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img = transformed_tensor['Image'].data
            gt = transformed_tensor['Mask'].data
    
        gt = gt[...,0]
        img = img[...,0] 
        gt_all = np.zeros_like(gt)
        gt_all[np.where(gt!=0)]=1
        gt_all = np.concatenate((gt_all, (1-gt_all)), axis=0)

        gt = generate_label(gt)

        M = Generate_Meta_Data(vendor)
        M = np.expand_dims(M, axis=0)
        
        return img,gt,gt_all,M
        
def Data_Loader_io_transforms(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_io(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

class Dataset_val(Dataset): 
    def __init__(self, df, images_folder):  ## If I apply Data Augmentation here, the validation loss becomes None. 
        self.df = df
        self.images_folder = images_folder
        self.gt_folder = self.images_folder[:-5] + 'gts'
        self.vendors = df['VENDOR']       
        self.images_name1 = df['SUBJECT_CODE'] 
        
        self.images_name = os.listdir(images_folder)
        
    def __len__(self):
       return len(self.images_name)
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder,str(self.images_name[index]).zfill(3))
        img = sitk.ReadImage(img_path)    ## --> [H,W,C]
        img = resample_itk_image_LA(img)      ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        
        number = int(self.images_name[index][:3])
        ind = self.df['SUBJECT_CODE'].loc[lambda x: x==number].index[0]
        vendor = self.vendors[ind]
        
        if vendor=='GE MEDICAL SYSTEMS': 
            img = normalize_G(img)

        if vendor=='SIEMENS':
            img = normalize_S(img)

        if vendor=='Philips Medical Systems':
            img = normalize_P(img)
            
        C = img.shape[0]
        H = img.shape[1]
        W = img.shape[2]
        img = Cropping_3d(C,H,W,256,img)
        
        gt_path = os.path.join(self.gt_folder,str(self.images_name[index]).zfill(3))
        gt_path = gt_path[:-7]+'_gt.nii.gz'
        gt = sitk.ReadImage(gt_path)    ## --> [H,W,C]
        gt = resample_itk_image_LA(gt)      ## --> [H,W,C]
        gt = sitk.GetArrayFromImage(gt)   ## --> [C,H,W]
        C = gt.shape[0]
        H = gt.shape[1]
        W = gt.shape[2]
        gt = Cropping_3d(C,H,W,256,gt)
        
        gt_all = np.zeros_like(gt)
        gt_all[np.where(gt!=0)]=1
        gt_all = np.concatenate((gt_all, (1-gt_all)), axis=0)
        
        gt = generate_label(gt)

        M = Generate_Meta_Data(vendor)
        M = np.expand_dims(M, axis=0)
        
        return img,gt,gt_all,M
        
def Data_Loader_val(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_val(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

from DC1 import DiceLoss
loss_fn_DC1 = DiceLoss()
from DC2 import DiceLoss
loss_fn_DC2 = DiceLoss()
    
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
    
    
for fold in range(1,6):
    fold = str(fold)  ## training fold number 
    
    train_imgs = "/data/scratch/acw676/MNM2/five_folds/F"+fold+"/train/imgs/"
    train_csv_path = "/data/scratch/acw676/MNM2/five_folds/F"+fold+"/train/F"+fold+"_train.csv"
    df_train = pd.read_csv(train_csv_path)
    
    val_imgs  = "/data/scratch/acw676/MNM2/five_folds/F"+fold+"/val/imgs/"
    val_csv_path  = "/data/scratch/acw676/MNM2/five_folds/F"+fold+"/val/F"+fold+"_val.csv"
    df_val  = pd.read_csv(val_csv_path)
    
    Batch_Size = 16
    Max_Epochs = 500
    
    train_loader = Data_Loader_io_transforms(df_train,train_imgs,batch_size = Batch_Size)
    val_loader = Data_Loader_val(df_val,val_imgs,batch_size = Batch_Size)
    
    
    
    print(len(train_loader)) ### this shoud be = Total_images/ batch size
    print(len(val_loader))   ### same here
    #print(len(test_loader))   ### same here
    
    ### Specify all the Losses (Train+ Validation), and Validation Dice score to plot on learing-curve
    
    avg_train_losses1_seg = []   # losses of all training epochs
    avg_valid_losses1_seg = []  #losses of all training epochs
    
    avg_valid_DS_ValSet = []  # all training epochs
    avg_valid_DS_TrainSet = []  # all training epochs
    ### Next we have all the funcitons which will be called in the main for training ####
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    

      #### Specify all the paths here #####
       
    
    path_to_save_Learning_Curve = '/data/scratch/acw676/MNM2/weights/'+'/F'+fold+'_BseLine2_VI'
    path_to_save_check_points = '/data/scratch/acw676/MNM2/weights/'+'/F'+fold+'_BseLine2_VI'
    
    ### 3 - this function will save the check-points 
    def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
        print("=> Saving checkpoint")
        torch.save(state, filename)
        
    
    def check_Dice_Score(loader, model1, device=DEVICE):
        
        Dice_score_LA_LV = 0
        Dice_score_LA_MYO = 0
        Dice_score_LA_RV = 0
    
        loop = tqdm(loader)
        model1.eval()
        
        for batch_idx, (img,gt,gt_all,M) in enumerate(loop):
            
            img = img.to(device=DEVICE,dtype=torch.float)  
            gt = gt.to(device=DEVICE,dtype=torch.float)  
            gt_all = gt_all.to(device=DEVICE,dtype=torch.float) 
            M = M.to(device=DEVICE,dtype=torch.float) 
        
            with torch.no_grad(): 
                _,pre_2d = model1(img) 
                pre_2d = (pre_2d >= 0.5)*1
    
                out_LV = pre_2d[:,0:1,:]
                out_MYO = pre_2d[:,1:2,:]
                out_RV = pre_2d[:,2:3,:]
                
                ## Dice Score for ES-LA ###
    
                single_LA_LV = (2 * (out_LV * gt[:,0:1,:]).sum()) / (
                   (out_LV + gt[:,0:1,:]).sum() + 1e-8)
               
                Dice_score_LA_LV +=single_LA_LV
               
                single_LA_MYO = (2 * (out_MYO * gt[:,1:2,:]).sum()) / (
       (out_MYO + gt[:,1:2,:]).sum() + 1e-8)
                
                Dice_score_LA_MYO += single_LA_MYO
    
                single_LA_RV = (2 * (out_RV * gt[:,2:3,:]).sum()) / (
           (out_RV + gt[:,2:3,:]).sum() + 1e-8)
                
                Dice_score_LA_RV += single_LA_RV
    
        print(f"Dice_score_LA_RV  : {Dice_score_LA_RV/len(loader)}")
        print(f"Dice_score_LA_MYO  : {Dice_score_LA_MYO/len(loader)}")
        print(f"Dice_score_LA_LV  : {Dice_score_LA_LV/len(loader)}")
    
        Overall_Dicescore_LA =( Dice_score_LA_RV + Dice_score_LA_MYO + Dice_score_LA_LV ) /3
        
        return Overall_Dicescore_LA/len(loader)
    
    
    ### 2- the main training fucntion to update the weights....
    def train_fn(loader_train1,loader_valid1,model1, optimizer1, scaler,loss_fn_DC1,loss_fn_DC2):  ### Loader_1--> ED and Loader2-->ES
        train_losses1_seg  = [] # loss of each batch
        valid_losses1_seg  = []  # loss of each batch
    
        loop = tqdm(loader_train1)
        model1.train()
        
        for batch_idx,(img,gt,gt_all,M) in enumerate(loop):
            
            img = img.to(device=DEVICE,dtype=torch.float)  
            gt = gt.to(device=DEVICE,dtype=torch.float)  
            gt_all = gt_all.to(device=DEVICE,dtype=torch.float) 
            M = M.to(device=DEVICE,dtype=torch.float) 
    
            with torch.cuda.amp.autocast():
                out_all, pre_2d = model1(img) ## loss1 is for 4 classes
                loss1 = loss_fn_DC1(pre_2d,gt)
                loss2 = loss_fn_DC2(out_all,gt_all)
                loss = (loss1+loss2)/2
    
                            # backward
            optimizer1.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer1)
    
            scaler.update()
            # update tqdm loop
            loop.set_postfix(loss = loss.item())   ## loss = loss1.item()
            train_losses1_seg.append(float(loss))
            
        loop_v = tqdm(loader_valid1)
        model1.eval() 
        
        for batch_idx,(img,gt,gt_all,M) in enumerate(loop_v):
            
            img = img.to(device=DEVICE,dtype=torch.float)  
            gt = gt.to(device=DEVICE,dtype=torch.float)  
            gt_all = gt_all.to(device=DEVICE,dtype=torch.float) 
            M = M.to(device=DEVICE,dtype=torch.float) 
            
            with torch.no_grad(): 
                out_all, pre_2d = model1(img) ## loss1 is for 4 classes
                loss1 = loss_fn_DC1(pre_2d,gt)
                loss2 = loss_fn_DC2(out_all,gt_all)
                loss = (loss1+loss2)/2
    
            # backward
            loop_v.set_postfix(loss = loss.item())
            valid_losses1_seg.append(float(loss))
    
        train_loss_per_epoch1_seg = np.average(train_losses1_seg)
        valid_loss_per_epoch1_seg  = np.average(valid_losses1_seg)
        
        
        avg_train_losses1_seg.append(train_loss_per_epoch1_seg)
        avg_valid_losses1_seg.append(valid_loss_per_epoch1_seg)
        
        return train_loss_per_epoch1_seg , valid_loss_per_epoch1_seg
    
    
    model_1 = BaseLine2()
    

    
    epoch_len = len(str(Max_Epochs))
    
    def main():
        model1 = model_1.to(device=DEVICE,dtype=torch.float)
        
        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(Max_Epochs):
            
            if epoch<=10:
              LEARNING_RATE = 0.0008
              
            if epoch>10:
              LEARNING_RATE = 0.0005
            
            if epoch>50:
              LEARNING_RATE = 0.0001
              
            if epoch>100:
              LEARNING_RATE = 0.00008
            
            if epoch>200:
              LEARNING_RATE = 0.00005
            
            if epoch>300:
              LEARNING_RATE = 0.00001
              
            #optimizer1 = optim.SGD(model1.parameters(),lr=LEARNING_RATE)
            optimizer1 = optim.Adam(model1.parameters(),betas=(0.9, 0.99),lr=LEARNING_RATE)
            train_loss,valid_loss = train_fn(train_loader,val_loader, model1, optimizer1,scaler,loss_fn_DC1,loss_fn_DC2)
            
            print_msg = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')
            
            print(print_msg)
            Dice_val = check_Dice_Score(val_loader, model1, device=DEVICE)
            avg_valid_DS_ValSet.append(Dice_val.detach().cpu().numpy())
            
                # save model
        checkpoint = {
            "state_dict": model1.state_dict(),
            "optimizer":optimizer1.state_dict(),
        }
        save_checkpoint(checkpoint)
            
    
    if __name__ == "__main__":
        main()
    
    ### This part of the code will generate the learning curve ......
    
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    #plt.plot(range(1,len(avg_train_losses1_clip)+1),avg_train_losses1_clip, label='Training CLIP Loss')
    #plt.plot(range(1,len(avg_valid_losses1_clip)+1),avg_valid_losses1_clip,label='Validation CLIP Loss')
    
    plt.plot(range(1,len(avg_train_losses1_seg)+1),avg_train_losses1_seg, label='Training Segmentation Loss')
    plt.plot(range(1,len(avg_valid_losses1_seg)+1),avg_valid_losses1_seg,label='Validation Segmentation Loss')
    
    plt.plot(range(1,len(avg_valid_DS_ValSet)+1),avg_valid_DS_ValSet,label='Validation DS')
    plt.plot(range(1,len(avg_valid_DS_TrainSet)+1),avg_valid_DS_TrainSet,label='Train DS')
    
    # find position of lowest validation loss
    minposs = avg_valid_losses1_seg.index(min(avg_valid_losses1_seg))+1 
    plt.axvline(minposs,linestyle='--', color='r',label='Early Stopping Checkpoint')
    font1 = {'size':20}
    plt.title("Learning Curve Graph",fontdict = font1)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(-1, 1) # consistent scale
    plt.xlim(0, len(avg_train_losses1_seg)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(path_to_save_Learning_Curve+'.png', bbox_inches='tight')

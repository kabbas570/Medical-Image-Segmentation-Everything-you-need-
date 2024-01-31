import torch
import torch.nn as nn


class SingleLinear(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.linear_1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(negative_slope=0.01),
        )

    def forward(self, x):
        return self.linear_1(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01),
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv_s2(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_s2 = nn.Sequential(DoubleConv_s2(in_channels, out_channels))

    def forward(self, x):
        return self.conv_s2(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
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

class ClassMLPNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.m_inc1 = SingleLinear(4, 32 * 2, 0.2)
        self.m_inc2 = SingleLinear(32 * 2, 64 * 2, 0.2)
        self.m_inc3 = SingleLinear(64 * 2, 128 * 2, 0.2)
        self.m_inc4 = SingleLinear(128 * 2, 256 * 2, 0.2)
        self.m_inc5 = SingleLinear(256 * 2, 512 * 2, 0.2)
        self.m_inc6 = SingleLinear(512 * 2, 1024 * 2, 0.2)

        self.linear1 = nn.Linear(1024 * 2, 128)
        self.linear_v = nn.Linear(128, 3)
        self.linear_s = nn.Linear(128, 9)
        self.linear_d = nn.Linear(128, 6)
        self.linear_f = nn.Linear(128, 2)

        self.act = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, Meta_Data):
        m1 = self.m_inc1(Meta_Data)
        m2 = self.m_inc2(m1)
        m3 = self.m_inc3(m2)
        m4 = self.m_inc4(m3)
        m5 = self.m_inc5(m4)
        m6 = self.m_inc6(m5)
        m = self.act(self.dropout(self.linear1(m6)))
        
        gammam1, betam1 = torch.split(m1, 32, dim=1)
        gammam2, betam2 = torch.split(m2, 64, dim=1)
        gammam3, betam3 = torch.split(m3, 128, dim=1)
        gammam4, betam4 = torch.split(m4, 256, dim=1)
        gammam5, betam5 = torch.split(m5, 512, dim=1)
        gammam6, betam6 = torch.split(m6, 1024, dim=1)
        gammam = [gammam1, gammam2, gammam3, gammam4, gammam5, gammam6]
        betam = [betam1, betam2, betam3, betam4, betam5, betam6]

        logits_v = self.linear_v(m)
        logits_s = self.linear_s(m)
        logits_d = self.linear_d(m)
        logits_f = self.linear_f(m)

        return logits_v, logits_s, logits_d, logits_f, gammam, betam


class SegNet(nn.Module):
    def __init__(self, n_channels=1):
        super(SegNet, self).__init__()
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

        self.outc = OutConv(32, 2)
        self.outc4 = OutConv(32, 4)

        self.dropout5E = nn.Dropout2d(p=0.20)
        self.dropout6E = nn.Dropout2d(p=0.20)

        self.dropout1D = nn.Dropout2d(p=0.20)
        self.dropout2D = nn.Dropout2d(p=0.20)

    def forward(self, x, gamma_x=None, beta_x=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dropout5E(x5)
        x6 = self.down5(x5)
        x6 = self.dropout6E(x6)

        if gamma_x is not None and beta_x is not None:
            x1 = LearnedModulation(x1, gamma_x[0], beta_x[0])
            x2 = LearnedModulation(x2, gamma_x[1], beta_x[1])
            x3 = LearnedModulation(x3, gamma_x[2], beta_x[2])
            x4 = LearnedModulation(x4, gamma_x[3], beta_x[3])
            x5 = LearnedModulation(x5, gamma_x[4], beta_x[4])
            x6 = LearnedModulation(x6, gamma_x[5], beta_x[5])

        z1 = self.up0(x6, x5)
        z1 = self.dropout1D(z1)
        z2 = self.up1(z1, x4)
        z2 = self.dropout2D(z2)
        z3 = self.up2(z2, x3)
        z4 = self.up3(z3, x2)
        z5 = self.up4(z4, x1)
        logits1 = self.outc(z5)

        y1 = self.up0_(z1, z2)
        y2 = self.up1_(y1, z3)
        y3 = self.up2_(y2, z4)
        y4 = self.up3_(y3, z5)

        logits2 = self.outc4(y4)

        return logits1, logits2


def LearnedModulation(x, gamma, beta):
    if x.shape[1] != gamma.shape[1]:
        raise ValueError("Input and gamma must have the same number of channels")
    if x.shape[1] != beta.shape[1]:
        raise ValueError("Input and beta must have the same number of channels")
    if gamma.ndim == 2:
        gamma = gamma[:, :, None, None]
    if beta.ndim == 2:
        beta = beta[:, :, None, None]
    return gamma * x + beta


class Baseline_MLP(nn.Module):
    def __init__(self):
        super(Baseline_MLP, self).__init__()
        self.SegNetwork = SegNet()
        self.ClassNetwork = ClassMLPNet()

    def forward(self, x, Meta_Data):
        logits_v, logits_s, logits_d, logits_f, gammam, betam = self.ClassNetwork(Meta_Data)
        logits1, logits2 = self.SegNetwork(x, gammam, betam)
        return logits1, logits2, logits_v, logits_s, logits_d, logits_f

# from torchsummary import summary
# if __name__ == "__main__":
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    

#     # model = model()
#     model = Baseline_MLP()
#     model.to(device=DEVICE, dtype=torch.float)
#     summary(model, [(1, 256, 256), (4,)])
#     image = torch.randn(10, 1, 256, 256).to(device=DEVICE, dtype=torch.float)
#     meta = torch.randn(10, 4).to(device=DEVICE, dtype=torch.float)
#     output = model(image, meta)

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
        mean=np.mean(img)
        std=np.std(img)
        img=(img-mean)/std
        return img 

geometrical_transforms = tio.OneOf([
    tio.RandomFlip(axes=([1, 2])),
    #tio.RandomElasticDeformation(num_control_points=(5, 5, 5), locked_borders=1, image_interpolation='nearest'),
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


def gen_meta(vendor, scanners, disease, field, size):
    temp = np.zeros(size)
    # Mapping vendors to numerical values
    vendor_mapping = {'Philips Medical Systems': 1, 'SIEMENS': 2, 'GE MEDICAL SYSTEMS': 3}
    temp[0] = vendor_mapping.get(vendor, 0)
    # Mapping scanners to numerical values
    scanners_mapping = {
        'Symphony': 4, 'SIGNA EXCITE': 3, 'Signa Explorer': 2,
        'SymphonyTim': 1, 'Avanto Fit': 0, 'Avanto': -1,
        'Achieva': -2, 'Signa HDxt': -3, 'TrioTim': -4
    }
    temp[1] = scanners_mapping.get(scanners, 0)
    # Mapping diseases to numerical values
    disease_mapping = {'NOR': 0, 'LV': 0.5, 'HCM': -0.5, 'ARR': 0.9, 'FALL': -0.9, 'CIA': -1.5}
    temp[2] = disease_mapping.get(disease, 0)
    
    # Mapping field to numerical values
    temp[3] = float(field)
    return temp

vendor_gt = {
    'GE MEDICAL SYSTEMS': 0,
    'SIEMENS': 1,
    'Philips Medical Systems': 2,
}

scanner_gt = {
        'Symphony': 0, 'SIGNA EXCITE': 1, 'Signa Explorer': 2,
        'SymphonyTim': 3, 'Avanto Fit': 4, 'Avanto': 5,
        'Achieva': 6, 'Signa HDxt': 7, 'TrioTim': 8
    }

disease_gt = {'NOR': 0, 'LV': 1, 'HCM': 2, 'ARR': 3, 'FALL': 4, 'CIA': 5}
field_gt = {'1.5': 0, '3.0': 1}

class Dataset_io(Dataset): 
    def __init__(self, df, images_folder,transformations=transforms_2d):  ## If I apply Data Augmentation here, the validation loss becomes None. 
        self.df = df
        self.images_folder = images_folder
        self.gt_folder = self.images_folder[:-5] + 'gts'
        
        self.vendors = df['VENDOR']
        self.scanners = df['SCANNER']
        self.diseases = df['DISEASE']
        self.fields = df['FIELD']      
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
        img = Normalization_1(img)
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
        number = int(self.images_name[index][:3])
        ind = self.df['SUBJECT_CODE'].loc[lambda x: x==number].index[0]
        vendor = self.vendors[ind]
        scanner = self.scanners[ind]
        disease = self.diseases[ind]
        field = self.fields[ind]
        
        v_gt = vendor_gt.get(vendor, 0)
        s_gt = scanner_gt.get(scanner, 0)
        d_gt = disease_gt.get(disease, 0)
        f_gt = field_gt.get(str(field), 0)
        meta_16 = gen_meta(vendor, scanner, scanner, field, (4))

        return img,gt,gt_all,meta_16,v_gt,s_gt,d_gt,f_gt
        
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
        self.scanners = df['SCANNER']
        self.diseases = df['DISEASE']
        self.fields = df['FIELD']      
        self.images_name1 = df['SUBJECT_CODE'] 
        
        self.images_name = os.listdir(images_folder)
        
    def __len__(self):
       return len(self.images_name)
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder,str(self.images_name[index]).zfill(3))
        img = sitk.ReadImage(img_path)    ## --> [H,W,C]
        img = resample_itk_image_LA(img)      ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Normalization_1(img)
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
        number = int(self.images_name[index][:3])
        ind = self.df['SUBJECT_CODE'].loc[lambda x: x==number].index[0]
        vendor = self.vendors[ind]
        scanner = self.scanners[ind]
        disease = self.diseases[ind]
        field = self.fields[ind]
        
        v_gt = vendor_gt.get(vendor, 0)
        s_gt = scanner_gt.get(scanner, 0)
        d_gt = disease_gt.get(disease, 0)
        f_gt = field_gt.get(str(field), 0)
        meta_16 = gen_meta(vendor, scanner, scanner, field, (4))
                
        return img,gt,gt_all,meta_16,v_gt,s_gt,d_gt,f_gt
        
def Data_Loader_val(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_val(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader


# val_imgs = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\F2\train\imgs/' ## path to images
# val_csv_path = r"C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\F2\train\F2_train.csv"
# df_val = pd.read_csv(val_csv_path)
# train_loader = Data_Loader_io_transforms(df_val,val_imgs,batch_size = 10)
# a = iter(train_loader)
# a1 =next(a) 

import matplotlib.pyplot as plt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm import tqdm
import torch.optim as optim


def check_Dice_Score(loader, model1, device=DEVICE):
    
    Dice_score_LA_LV = 0
    Dice_score_LA_MYO = 0
    Dice_score_LA_RV = 0
        
    correct_v = 0
    correct_s = 0
    correct_d = 0
    correct_f = 0
    num_samples = 0

    loop = tqdm(loader)
    model1.eval()
    
    for batch_idx, (img,gt,gt_all,meta_16,v_gt,s_gt,d_gt,f_gt) in enumerate(loop):
        
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt = gt.to(device=DEVICE,dtype=torch.float)  
        gt_all = gt_all.to(device=DEVICE,dtype=torch.float) 
        
        meta_16 = meta_16.to(device=DEVICE,dtype=torch.float) 
        v_gt = v_gt.to(device=DEVICE) 
        s_gt = s_gt.to(device=DEVICE) 
        d_gt = d_gt.to(device=DEVICE) 
        f_gt = f_gt.to(device=DEVICE)
        
        with torch.no_grad(): 
            _,pre_2d,v_pre,s_pre,d_pre,f_pre = model1(img,meta_16) 
            
            ## classification ###
            
            _, predictions_v = v_pre.max(1)
            correct_v += (predictions_v == v_gt).sum()
            num_samples += predictions_v.size(0)
            
            _, predictions_s = s_pre.max(1)
            correct_s += (predictions_s == s_gt).sum()
            
            _, predictions_d = d_pre.max(1)
            correct_d += (predictions_d == d_gt).sum()
            
            _, predictions_f = f_pre.max(1)
            correct_f += (predictions_f == f_gt).sum()
            
            ## segemntaiton ##
            
            pred = torch.argmax(pre_2d, dim=1)

            out_LV = torch.zeros_like(pred)
            out_LV[torch.where(pred==0)] = 1
                
            out_MYO = torch.zeros_like(pred)
            out_MYO[torch.where(pred==1)] = 1
                
            out_RV = torch.zeros_like(pred)
            out_RV[torch.where(pred==2)] = 1
            
            # pre_2d = (pre_2d >= 0.5)*1
            # out_LV = pre_2d[:,0:1,:]
            # out_MYO = pre_2d[:,1:2,:]
            # out_RV = pre_2d[:,2:3,:]
                        
            single_LA_LV = (2 * (out_LV * gt[:,0,:]).sum()) / (
               (out_LV + gt[:,0,:]).sum() + 1e-8)
           
            Dice_score_LA_LV +=single_LA_LV
           
            single_LA_MYO = (2 * (out_MYO * gt[:,1,:]).sum()) / (
   (out_MYO + gt[:,1,:]).sum() + 1e-8)
            
            Dice_score_LA_MYO += single_LA_MYO

            single_LA_RV = (2 * (out_RV * gt[:,2,:]).sum()) / (
       (out_RV + gt[:,2,:]).sum() + 1e-8)
            
            Dice_score_LA_RV += single_LA_RV

    ## classification ###

    print(f"Vendor Accuracy  : {correct_v/num_samples }")
    print(f"Scanner Accuracy  : {correct_s/num_samples }")
    print(f"Disease Accuracy  : {correct_d/num_samples }")
    print(f"Field Accuracy  : { correct_f/num_samples }")
    
    Classificatio_Accuracy = (correct_v/num_samples+correct_s/num_samples+correct_d/num_samples+correct_f/num_samples)/4
    
    ## segemntaiton ##
    print(f"Dice_score_LA_RV  : {Dice_score_LA_RV/len(loader)}")
    print(f"Dice_score_LA_MYO  : {Dice_score_LA_MYO/len(loader)}")
    print(f"Dice_score_LA_LV  : {Dice_score_LA_LV/len(loader)}")

    Overall_Dicescore_LA = ( Dice_score_LA_RV + Dice_score_LA_MYO + Dice_score_LA_LV )/3
    
    return Overall_Dicescore_LA/len(loader),Classificatio_Accuracy/len(loader)
    

criterion = nn.CrossEntropyLoss()

def train_fn(loader_train1,loader_valid1,model1, optimizer1,optimizer2, scaler1,scaler2,loss_fn_DC1,loss_fn_DC2): ### Loader_1--> ED and Loader2-->ES

    train_losses1_seg  = [] # loss of each batch
    valid_losses1_seg  = []  # loss of each batch
    
    
    train_losses1_class  = [] # loss of each batch
    valid_losses1_class = []  # loss of each batch

    loop = tqdm(loader_train1)
    model1.train()
    
    for batch_idx,(img,gt,gt_all,meta_16,v_gt,s_gt,d_gt,f_gt)  in enumerate(loop):
        
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt = gt.to(device=DEVICE,dtype=torch.float)  
        gt_all = gt_all.to(device=DEVICE,dtype=torch.float) 
        
        meta_16 = meta_16.to(device=DEVICE,dtype=torch.float) 
        v_gt = v_gt.to(device=DEVICE) 
        s_gt = s_gt.to(device=DEVICE) 
        d_gt = d_gt.to(device=DEVICE) 
        f_gt = f_gt.to(device=DEVICE)

        with torch.cuda.amp.autocast():
            out_all,pre_2d,v_pre,s_pre,d_pre,f_pre = model1(img,meta_16)    ## loss1 is for 4 classes
            ## segmentation losses ##
            loss1 = loss_fn_DC1(pre_2d,gt)
            loss2 = loss_fn_DC2(out_all,gt_all)
            loss = (loss1+loss2)/2
            ## classification  losses ##
            cl_loss_v = criterion(v_pre,v_gt)
            cl_loss_s = criterion(s_pre,s_gt)
            cl_loss_d = criterion(d_pre,d_gt)
            cl_loss_f = criterion(f_pre,f_gt)
            cl_loss = (cl_loss_v+cl_loss_s+cl_loss_d+cl_loss_f)/4

                        # backward
                        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        scaler1.scale(loss).backward()
        scaler2.scale(cl_loss).backward()
        
        scaler1.step(optimizer1)
        scaler2.step(optimizer2)
        
        scaler1.update()
        scaler2.update()
        
        
        # update tqdm loop
        loop.set_postfix(loss = loss.item())   ## loss = loss1.item()
        
        train_losses1_seg.append(float(loss))
        train_losses1_class.append(float(cl_loss))
        
    loop_v = tqdm(loader_valid1)
    model1.eval() 
    
    for batch_idx,(img,gt,gt_all,meta_16,v_gt,s_gt,d_gt,f_gt) in enumerate(loop_v):
        
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt = gt.to(device=DEVICE,dtype=torch.float)  
        gt_all = gt_all.to(device=DEVICE,dtype=torch.float) 
        
        meta_16 = meta_16.to(device=DEVICE,dtype=torch.float) 
        v_gt = v_gt.to(device=DEVICE) 
        s_gt = s_gt.to(device=DEVICE) 
        d_gt = d_gt.to(device=DEVICE) 
        f_gt = f_gt.to(device=DEVICE)
      
        with torch.no_grad(): 
            out_all,pre_2d,v_pre,s_pre,d_pre,f_pre  = model1(img,meta_16)    ## loss1 is for 4 classes
            ## segmentation losses ##
            loss1 = loss_fn_DC1(pre_2d,gt)
            loss2 = loss_fn_DC2(out_all,gt_all)
            loss = (loss1+loss2)/2
            ## classification  losses ##
            cl_loss_v = criterion(v_pre,v_gt)
            cl_loss_s = criterion(s_pre,s_gt)
            cl_loss_d = criterion(d_pre,d_gt)
            cl_loss_f = criterion(f_pre,f_gt)
            cl_loss = (cl_loss_v+cl_loss_s+cl_loss_d+cl_loss_f)/4

        # backward
        loop_v.set_postfix(loss = loss.item())
        valid_losses1_seg.append(float(loss))
        valid_losses1_class.append(float(cl_loss))
        

    train_loss_per_epoch1_seg = np.average(train_losses1_seg)
    valid_loss_per_epoch1_seg  = np.average(valid_losses1_seg)
    
    avg_train_losses1_seg.append(train_loss_per_epoch1_seg)
    avg_valid_losses1_seg.append(valid_loss_per_epoch1_seg)
    
    train_loss_per_epoch1_class = np.average(train_losses1_class)
    valid_loss_per_epoch1_class  = np.average(valid_losses1_class)
    
    avg_train_losses1_class.append(train_loss_per_epoch1_class)
    avg_valid_losses1_class.append(valid_loss_per_epoch1_class)
    
    return train_loss_per_epoch1_seg,valid_loss_per_epoch1_seg,train_loss_per_epoch1_class,valid_loss_per_epoch1_class

  
from DC1 import DiceLoss
loss_fn_DC1 = DiceLoss()
from DC2 import DiceLoss
loss_fn_DC2 = DiceLoss()

for fold in range(1,6):

  fold = str(fold)  ## training fold number 
  
  train_imgs = "/data/scratch/acw676/MNM2/five_folds/F"+fold+"/train/imgs/"
  train_csv_path = "/data/scratch/acw676/MNM2/five_folds/F"+fold+"/train/F"+fold+"_train.csv"
  df_train = pd.read_csv(train_csv_path)
  
  val_imgs  = "/data/scratch/acw676/MNM2/five_folds/F"+fold+"/val/imgs/"
  val_csv_path  = "/data/scratch/acw676/MNM2/five_folds/F"+fold+"/val/F"+fold+"_val.csv"
  df_val  = pd.read_csv(val_csv_path)
  
  Batch_Size = 16
  Max_Epochs = 200
  
  train_loader = Data_Loader_io_transforms(df_train,train_imgs,batch_size = Batch_Size)
  val_loader = Data_Loader_val(df_val,val_imgs,batch_size = Batch_Size)
  
  
  
  print(len(train_loader)) ### this shoud be = Total_images/ batch size
  print(len(val_loader))   ### same here
  #print(len(test_loader))   ### same here
  
  avg_train_losses1_seg = []   # losses of all training epochs
  avg_valid_losses1_seg = []  #losses of all training epochs
    
  avg_valid_DS_ValSet_seg = []  # all training epochs
  avg_valid_DS_TrainSet_seg = []  # all training epochs
    
  avg_train_losses1_class = []   # losses of all training epochs
  avg_valid_losses1_class = []  #losses of all training epochs
    
  avg_valid_DS_ValSet_class = []  # all training epochs
  avg_valid_DS_TrainSet_class = []  # all training epochs
     
  
  path_to_save_Learning_Curve = '/data/scratch/acw676/MNM2/Feb/'+'/F'+fold+'Baseline_MLP_1'
  path_to_save_check_points = '/data/scratch/acw676/MNM2/Feb/'+'/F'+fold+'Baseline_MLP_1'
  
  ### 3 - this function will save the check-points 
  def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
      print("=> Saving checkpoint")
      torch.save(state, filename)
      
  model_1 = Baseline_MLP()
  epoch_len = len(str(Max_Epochs))

  def main():
      model1 = model_1.to(device=DEVICE,dtype=torch.float)
      
      scaler1 = torch.cuda.amp.GradScaler()
      scaler2 = torch.cuda.amp.GradScaler()

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
            
          optimizer1 = optim.Adam(model1.SegNetwork.parameters(),betas=(0.9, 0.99),lr=LEARNING_RATE)
          optimizer2 = optim.Adam(model1.ClassNetwork.parameters(),betas=(0.9, 0.99),lr=LEARNING_RATE)
          
          train_loss_seg,valid_loss_seg ,train_loss_class,valid_loss_class = train_fn(train_loader,val_loader, model1, optimizer1,optimizer2,scaler1,scaler2,loss_fn_DC1,loss_fn_DC2)
          
          print_msg1 = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss_seg: {train_loss_seg:.5f} ' +
                     f'valid_loss_seg: {valid_loss_seg:.5f}')
        
          print_msg2 = (f'[{epoch:>{epoch_len}}/{Max_Epochs:>{epoch_len}}] ' +
                     f'train_loss_class: {train_loss_class:.5f} ' +
                     f'valid_loss_class: {valid_loss_class:.5f}')
        
          print(print_msg1)
          print(print_msg2)
          Dice_val,Class_Acc = check_Dice_Score(val_loader, model1, device=DEVICE)
          avg_valid_DS_ValSet_seg.append(Dice_val.detach().cpu().numpy())
          avg_valid_DS_ValSet_class.append(Class_Acc.detach().cpu().numpy())
          
              # save model
      checkpoint = {
          "state_dict": model1.state_dict(),
          "optimizer":optimizer1.state_dict(),
      }
      save_checkpoint(checkpoint)
          
  
  if __name__ == "__main__":
      main()
  
  fig = plt.figure(figsize=(10,8))
    
  plt.plot(range(1,len(avg_train_losses1_seg)+1),avg_train_losses1_seg, label='Training Segmentation Loss')
  plt.plot(range(1,len(avg_valid_losses1_seg)+1),avg_valid_losses1_seg,label='Validation Segmentation Loss')
    
  plt.plot(range(1,len(avg_train_losses1_class)+1),avg_train_losses1_class, label='Training Classsification Loss')
  plt.plot(range(1,len(avg_valid_losses1_class)+1),avg_valid_losses1_class,label='Validation Classsification Loss')
    
  plt.plot(range(1,len(avg_valid_DS_ValSet_seg)+1),avg_valid_DS_ValSet_seg,label='Validation DS')
  plt.plot(range(1,len(avg_valid_DS_ValSet_class)+1),avg_valid_DS_ValSet_class,label='Validation Acc')
    
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

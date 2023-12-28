import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk
#from typing import List, Union, Tuple
import torch
import albumentations as A
import cv2
from torch.utils.data import SubsetRandomSampler
import torchio as tio
from sklearn.model_selection import KFold
           ###########  Dataloader  #############

NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 256
   
def Generate_Meta_(vendors_,scanners_,diseases_): 
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
        mean=np.mean(img)
        std=np.std(img)
        img=(img-mean)/std
        return img 

     
def generate_label(gt,org_dim3):
        temp_ = np.zeros([4,org_dim3,DIM_,DIM_])
        temp_[0,:,:,:][np.where(gt==1)]=1
        temp_[1,:,:,:][np.where(gt==2)]=1
        temp_[2,:,:,:][np.where(gt==3)]=1
        temp_[3,:,:,:][np.where(gt==0)]=1
        return temp_


    
class Dataset_io(Dataset): 
    def __init__(self, df, images_folder,transformations=None):  ## If I apply Data Augmentation here, the validation loss becomes None. 
        self.df = df
        self.images_folder = images_folder
        self.gt_folder = self.images_folder[:-4] + 'gts'
        self.vendors = df['VENDOR']       
        #self.images_name = df['SUBJECT_CODE'] 
        self.transformations = transformations
        
        self.images_name = os.listdir(images_folder)
        
    def __len__(self):
        return self.vendors.shape[0]
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder,str(self.images_name[index]).zfill(3))
        print(img_path)
        img = sitk.ReadImage(img_path)    ## --> [H,W,C]
        img = resample_itk_image_LA(img)      ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        C = img.shape[0]
        H = img.shape[1]
        W = img.shape[2]

        img = Cropping_3d(C,H,W,256,img)
        print(img.shape)

        
        gt_path = os.path.join(self.gt_folder,str(self.images_name[index]).zfill(3))
        gt_path = gt_path[:-7]+'_gt.nii.gz'
        gt = sitk.ReadImage(gt_path)    ## --> [H,W,C]
        gt = resample_itk_image_LA(gt)      ## --> [H,W,C]
        gt = sitk.GetArrayFromImage(gt)   ## --> [C,H,W]
        
        C = gt.shape[0]
        H = gt.shape[1]
        W = gt.shape[2]

        gt = Cropping_3d(C,H,W,256,gt)
        print(gt.shape)
        
        return img,gt
        
        
        
def Data_Loader_io_transforms(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_io(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader


val_imgs = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\F1\val\imgs' ## path to images
val_csv_path = r"C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\F1\val\F1_val.csv"
df_val = pd.read_csv(val_csv_path)
train_loader = Data_Loader_io_transforms(df_val,val_imgs,batch_size = 1)
a = iter(train_loader)
a1 =next(a)


img=a1[0]
plt.figure()
plt.imshow(img[0,0,:])
gt=a1[1]
plt.figure()
plt.imshow(gt[0,0,:])

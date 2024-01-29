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


def Normalization_2(img):
        mean=np.mean(img)
        std=np.std(img)
        img=(img-mean)/std
        return img 
           
geometrical_transforms = tio.OneOf([
    tio.RandomFlip(axes=([1, 2])),
    #tio.RandomElasticDeformation(num_control_points=(2, 2, 2), locked_borders=1, image_interpolation='nearest'),
    tio.RandomAffine(degrees=(-45, 45), center='image'),
])

intensity_transforms = tio.OneOf([
    tio.RandomBlur(),
    tio.RandomGamma(log_gamma=(-0.2, -0.2)),
    tio.RandomNoise(mean=0.1, std=0.1),
    tio.RandomGhosting(axes=([1, 2])),
])

transforms_2d = tio.Compose({
    geometrical_transforms: 0.2,  # Probability for geometric transforms
    intensity_transforms: 0.2,   # Probability for intensity transforms
    tio.Lambda(lambda x: x): 0.6 # Probability for no augmentation (original image)
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
    temp[0, :] = vendor_mapping.get(vendor, 0)
    
    # Mapping scanners to numerical values
    scanners_mapping = {
        'Symphony': 4, 'SIGNA EXCITE': 3, 'Signa Explorer': 2,
        'SymphonyTim': 1, 'Avanto Fit': 0, 'Avanto': -1,
        'Achieva': -2, 'Signa HDxt': -3, 'TrioTim': -4
    }
    temp[1, :] = scanners_mapping.get(scanners, 0)
    
    # Mapping diseases to numerical values
    disease_mapping = {'NOR': 0, 'LV': 0.5, 'HCM': -0.5, 'ARR': 0.9, 'FAIL': -0.9, 'CIA': -1.5}
    temp[2, :] = disease_mapping.get(disease, 0)
    
    # Mapping field to numerical values
    temp[3, :] = float(field)

    return temp

class Dataset_io(Dataset): 
    def __init__(self, df, images_folder,transformations=transforms_2d):  ## If I apply Data Augmentation here, the validation loss becomes None. 
        self.df = df
        self.images_folder = images_folder
        self.gt_folder = self.images_folder[:-4] + 'gts'
        
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

        number = int(self.images_name[index][:3])
        ind = self.df['SUBJECT_CODE'].loc[lambda x: x==number].index[0]
        vendor = self.vendors[ind]
        scanner = self.scanners[ind]
        disease = self.diseases[ind]
        field = self.fields[ind]
        
        meta_256 = gen_meta(vendor, scanner, disease, field, (4, 256, 256))
        meta_128 = gen_meta(vendor, scanner, disease, field, (4, 128, 128))
        meta_64 = gen_meta(vendor, scanner, disease, field, (4, 4, 64))
        meta_32 = gen_meta(vendor, scanner, disease, field, (4, 32, 32))
        meta_16 = gen_meta(vendor, scanner, disease, field, (4, 16, 16))
        meta_8 = gen_meta(vendor, scanner, disease, field, (4, 8, 8))
        
        img = Normalization_2(img)

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
        
        img = np.concatenate((img, meta_256), axis=0)
                
        return img,gt,gt_all,meta_256,meta_128,meta_64,meta_32,meta_16,meta_8
        
def Data_Loader_io_transforms(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_io(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

class Dataset_val(Dataset): 
    def __init__(self, df, images_folder):  
        self.df = df
        self.images_folder = images_folder
        self.gt_folder = self.images_folder[:-4] + 'gts'
        
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
        
        number = int(self.images_name[index][:3])
        ind = self.df['SUBJECT_CODE'].loc[lambda x: x==number].index[0]
        vendor = self.vendors[ind]
        scanner = self.scanners[ind]
        disease = self.diseases[ind]
        field = self.fields[ind]
        
        meta_256 = gen_meta(vendor, scanner, disease, field, (4, 256, 256))
        meta_128 = gen_meta(vendor, scanner, disease, field, (4, 128, 128))
        meta_64 = gen_meta(vendor, scanner, disease, field, (4, 4, 64))
        meta_32 = gen_meta(vendor, scanner, disease, field, (4, 32, 32))
        meta_16 = gen_meta(vendor, scanner, disease, field, (4, 16, 16))
        meta_8 = gen_meta(vendor, scanner, disease, field, (4, 8, 8))
        

       
        img = Normalization_2(img)

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
        img = np.concatenate((img, meta_256), axis=0)
    
        return img,gt,gt_all,meta_256,meta_128,meta_64,meta_32,meta_16,meta_8
        
def Data_Loader_val(df,images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_val(df=df ,images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

# val_imgs = r'C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\F2\train\imgs' ## path to images
# val_csv_path = r"C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\F2\train\F2_train.csv"
# df_val = pd.read_csv(val_csv_path)
# train_loader = Data_Loader_val(df_val,val_imgs,batch_size = 4)
# a = iter(train_loader)
# a1 =next(a) 
# t= a1[3][0,:].numpy()

# import matplotlib.pyplot as plt
# for t in range(1):
#     a1 =next(a) 
# for p in range(1):
#     plt.figure()
#     plt.imshow(a1[0][0,p,:])    
# for p in range(4):
#     plt.figure()
#     plt.imshow(a1[1][0,p,:])   
# for p in range(2):
#     plt.figure()
#     plt.imshow(a1[2][0,p,:]) 
# for p in range(1):
#     plt.figure()
#     plt.imshow(a1[3][0,p,:]) 

# for p in range(1):
#     plt.figure()
#     plt.imshow(a1[4][0,p,:]) 
# m1 = a1[0][0,0,:].numpy()
# m2 = a1[3][0,0,:].numpy()
# m1 = a1[0][0,0,:].numpy()
# m2 = a1[0][0,1,:].numpy()

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm import tqdm
import torch.optim as optim


def check_Dice_Score(loader, model1, device=DEVICE):
    
    Dice_score_LA_LV = 0
    Dice_score_LA_MYO = 0
    Dice_score_LA_RV = 0

    loop = tqdm(loader)
    model1.eval()
    
    for batch_idx, (img,gt,gt_all,s_256,s_128,s_64,s_32,s_16,s_8) in enumerate(loop):
        
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt = gt.to(device=DEVICE,dtype=torch.float)  
        gt_all = gt_all.to(device=DEVICE,dtype=torch.float) 
        
        s_256 = s_256.to(device=DEVICE,dtype=torch.float) 
        s_128 = s_128.to(device=DEVICE,dtype=torch.float) 
        s_64 = s_64.to(device=DEVICE,dtype=torch.float) 
        s_32 = s_32.to(device=DEVICE,dtype=torch.float) 
        s_16 = s_16.to(device=DEVICE,dtype=torch.float) 
        s_8 = s_8.to(device=DEVICE,dtype=torch.float) 
        with torch.no_grad(): 
            _,pre_2d = model1(img,s_256,s_128,s_64,s_32,s_16,s_8) 
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


def train_fn(loader_train1,loader_valid1,model1, optimizer1, scaler,loss_fn_DC1,loss_fn_DC2):  ### Loader_1--> ED and Loader2-->ES
    train_losses1_seg  = [] # loss of each batch
    valid_losses1_seg  = []  # loss of each batch

    loop = tqdm(loader_train1)
    model1.train()
    
    for batch_idx,(img,gt,gt_all,s_256,s_128,s_64,s_32,s_16,s_8)  in enumerate(loop):
        
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt = gt.to(device=DEVICE,dtype=torch.float)  
        gt_all = gt_all.to(device=DEVICE,dtype=torch.float) 
        
        s_256 = s_256.to(device=DEVICE,dtype=torch.float) 
        s_128 = s_128.to(device=DEVICE,dtype=torch.float) 
        s_64 = s_64.to(device=DEVICE,dtype=torch.float) 
        s_32 = s_32.to(device=DEVICE,dtype=torch.float) 
        s_16 = s_16.to(device=DEVICE,dtype=torch.float) 
        s_8 = s_8.to(device=DEVICE,dtype=torch.float) 

        with torch.cuda.amp.autocast():
            out_all, pre_2d = model1(img,s_256,s_128,s_64,s_32,s_16,s_8)   ## loss1 is for 4 classes
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
    
    for batch_idx,(img,gt,gt_all,s_256,s_128,s_64,s_32,s_16,s_8) in enumerate(loop_v):
        
        img = img.to(device=DEVICE,dtype=torch.float)  
        gt = gt.to(device=DEVICE,dtype=torch.float)  
        gt_all = gt_all.to(device=DEVICE,dtype=torch.float) 
        
        s_256 = s_256.to(device=DEVICE,dtype=torch.float) 
        s_128 = s_128.to(device=DEVICE,dtype=torch.float) 
        s_64 = s_64.to(device=DEVICE,dtype=torch.float) 
        s_32 = s_32.to(device=DEVICE,dtype=torch.float) 
        s_16 = s_16.to(device=DEVICE,dtype=torch.float) 
        s_8 = s_8.to(device=DEVICE,dtype=torch.float) 
      
        with torch.no_grad(): 
            out_all, pre_2d = model1(img,s_256,s_128,s_64,s_32,s_16,s_8)   ## loss1 is for 4 classes
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
  

    #### Specify all the paths here #####
     
  
  path_to_save_Learning_Curve = '/data/scratch/acw676/MNM2/Feb/'+'/F'+fold+'_BseLine2_Norm_2_M2_withnorm'
  path_to_save_check_points = '/data/scratch/acw676/MNM2/Feb/'+'/F'+fold+'_BseLine2_Norm_2_M2_withnorm'
  
  ### 3 - this function will save the check-points 
  def save_checkpoint(state, filename=path_to_save_check_points+".pth.tar"):
      print("=> Saving checkpoint")
      torch.save(state, filename)
      
    
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

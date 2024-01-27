import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import glob
import pandas as pd

np.random.seed(42)

mean_AllData = 125.667
std_AllData = 208.77

 ### Siemen ##
mean_All_S = 117.698
std_All_S = 161.122

mean_B_S = 115.115
std_B_S = 166.147

mean_F_S = 166.474
std_F_S = 99.211

###   GE  ###
mean_All_GE = 212.584
std_All_GE= 316.465

mean_B_GE = 245.733
std_B_GE = 332.52

mean_F_GE = 495.01
std_F_GE= 223.476

###   Philip  ###
mean_All_P = 72.875
std_All_P = 102.589

mean_B_P = 78.205
std_B_P = 104.051

mean_F_P = 162.632
std_F_P = 98.043

pixels_G_F = []  ## Accumulate all foreground-pixels
pixels_S_F = []
pixels_P_F = []

pixels_G_B = []  ## Accumulate all Bakground-pixels
pixels_S_B = []
pixels_P_B = []

pixels_G =[]  ## All pixels 
pixels_S =[]
pixels_P =[]

df  = pd.read_csv(r"C:\My_Data\M2M Data\data\train.csv")
vendors = df['VENDOR']
counts = vendors.value_counts()

def Normalization_2(img):
        mean=np.mean(img)
        std=np.std(img)
        img=(img-mean)/std
        return img 
    
def Normalization_1(img):
 return (img - 125.667 ) / 208.77

# hist1 = np.random.normal(mean_F_GE, std_F_GE, size=(64, 64))
# hist1 = Normalization_1(hist1)

# print(np.mean(hist1))
# print(np.std(hist1))

# plt.figure()
# values, counts = np.unique(hist1, return_counts=True)
# plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
# plt.xlabel('Unique Values')
# plt.ylabel('Frequency')
# plt.title(' Philips_FG Histogram ')
# mean = np.round(np.mean(hist1),decimals=3)
# plt.text(0.95, 0.95, f'Mean =  {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
# std = np.round(np.std(hist1),decimals=3)
# plt.text(0.95, 0.90, f'STD =  {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
# plt.show()


# hist2 = np.random.normal(mean_F_S, std_F_S, size=(64, 64))
# hist2 = Normalization_1(hist2)
# print(np.mean(hist2))
# print(np.std(hist2))


# plt.figure()
# values, counts = np.unique(hist2, return_counts=True)
# plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
# plt.xlabel('Unique Values')
# plt.ylabel('Frequency')
# plt.title(' Philips_FG Histogram ')
# mean = np.round(np.mean(hist2),decimals=3)
# plt.text(0.95, 0.95, f'Mean =  {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
# std = np.round(np.std(hist2),decimals=3)
# plt.text(0.95, 0.90, f'STD =  {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
# plt.show()


# hist3 = np.random.normal(mean_F_P, std_F_P, size=(64, 64))
# hist3 = Normalization_1(hist3)

# print(np.mean(hist3))
# print(np.std(hist3))

# plt.figure()
# values, counts = np.unique(hist3, return_counts=True)
# plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
# plt.xlabel('Unique Values')
# plt.ylabel('Frequency')
# plt.title(' Philips_FG Histogram ')
# mean = np.round(np.mean(hist3),decimals=3)
# plt.text(0.95, 0.95, f'Mean =  {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
# std = np.round(np.std(hist3),decimals=3)
# plt.text(0.95, 0.90, f'STD =  {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
# plt.show()



def crop_center_3D(img,cropx=256,cropy=256):
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
def get_pixel_intensities(img_paths):
    for i in range(320):
        image = sitk.GetArrayFromImage(sitk.ReadImage(img_paths[i]))
        C = image.shape[0]
        H = image.shape[1]
        W = image.shape[2]
        image = Cropping_3d(C,H,W,256,image)        
        image = image[0,:]
        
        name = img_paths[i][-16:]
        number1 = int(name[:-13])
        ind = df['SUBJECT_CODE'].loc[lambda x: x==number1].index[0]
        vendor = vendors[ind]
        
        gt_path  = img_paths[i][:-21] + 'gts/' + name[:-7]+ '_gt.nii.gz'
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)) 
        C = gt.shape[0]
        H = gt.shape[1]
        W = gt.shape[2]
        gt = Cropping_3d(C,H,W,256,gt)        
        gt = gt[0,:]
        
        if vendor=='GE MEDICAL SYSTEMS': 
             hist = np.random.normal(mean_F_GE, std_F_GE, size=(256, 256))
             hist = Normalization_1(hist)
             img1 = image*hist
             img1 = Normalization_2(img1)
             #image1 = img1.copy()
             #image2 = img1.copy()
             #image1[np.where(gt==0)] = 0 # extract foreground 
             #image2[np.where(gt!=0)] = 0 # extract foreground 
             #pixels_G_F.append(image1.flatten())
             #pixels_G_B.append(image2.flatten())
             pixels_G.append(img1.flatten())
             plt.imsave(r'C:\My_Data\M2M Data\data\data_2\three_vendor\all\with-hist\NR\G/' + name + '_bg.png',img1)
             
        if vendor=='SIEMENS':
             hist = np.random.normal(mean_F_S, std_F_S, size=(256, 256))
             hist = Normalization_1(hist)
             img1 = image*hist
             img1 = Normalization_2(img1)
             #image1 = img1.copy()
             #image2 = img1.copy()
             #image1[np.where(gt==0)] = 0 # extract foreground 
             #image2[np.where(gt!=0)] = 0 # extract foreground 
             #pixels_S_F.append(image1.flatten())
            # pixels_S_B.append(image2.flatten())
             pixels_S.append(img1.flatten())
             plt.imsave(r'C:\My_Data\M2M Data\data\data_2\three_vendor\all\with-hist\NR\S/' + name + '_bg.png',img1)
        if vendor=='Philips Medical Systems':
             hist = np.random.normal(mean_F_P, std_F_P, size=(256, 256))
             hist = Normalization_1(hist)
             img1 = image*hist
             img1 = Normalization_2(img1)

             #image1 = img1.copy()
             #image2 = img1.copy()
             #image1[np.where(gt==0)] = 0 # extract foreground 
             #image2[np.where(gt!=0)] = 0 # extract foreground 
             #pixels_P_F.append(image1.flatten())
             #pixels_P_B.append(image2.flatten())
             pixels_P.append(img1.flatten())
             plt.imsave(r'C:\My_Data\M2M Data\data\data_2\three_vendor\all\with-hist\NR\P/' + name + '_bg.png',img1)
                
if __name__ == "__main__":
    path_to_imgs = r"C:\My_Data\M2M Data\data\data_2\three_vendor\all\LA\imgs/"
    img_paths = sorted(glob.glob(path_to_imgs + '*.nii.gz'))
    get_pixel_intensities(img_paths)



def gen_meta(vendor,size):
    
    if vendor=='Philips Medical Systems':
        hist = np.random.normal(mean_F_P, std_F_P, size=size)
        hist = Normalization_1(hist)
    if vendor=='SIEMENS':
        hist = np.random.normal(mean_F_S, std_F_S, size=size)
        hist = Normalization_1(hist)
    if vendor=='GE MEDICAL SYSTEMS': 
         hist = np.random.normal(mean_F_GE, std_F_GE, size=size)
         hist = Normalization_1(hist)
    return hist


    
    
'''
pixels_P_F = np.concatenate(pixels_P_F)
pixels_P_F = pixels_P_F[pixels_P_F != 0]
# pixels_P_B = np.concatenate(pixels_P_B)
# pixels_P_B = pixels_P_B[pixels_P_B != 0]
# pixels_P = np.concatenate(pixels_P)

pixels_G_F = np.concatenate(pixels_G_F)
pixels_G_F = pixels_G_F[pixels_G_F != 0]
# pixels_G_B = np.concatenate(pixels_G_B)
# pixels_G_B = pixels_G_B[pixels_G_B != 0]
# pixels_G = np.concatenate(pixels_G)

pixels_S_F = np.concatenate(pixels_S_F)
pixels_S_F = pixels_S_F[pixels_S_F != 0]
# pixels_S_B = np.concatenate(pixels_S_B)
# pixels_S_B = pixels_S_B[pixels_S_B != 0]
# pixels_S = np.concatenate(pixels_S)

                    ### Plotting ForeGround Histrograms #### 
plt.figure()
values, counts = np.unique(pixels_P_F, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title(' Philips_FG Histogram ')
mean = np.round(np.mean(pixels_P_F),decimals=3)
plt.text(0.95, 0.95, f'Mean =  {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_P_F),decimals=3)
plt.text(0.95, 0.90, f'STD =  {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
plt.show()

plt.figure()
values, counts = np.unique(pixels_G_F, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title('GE_FG Histogram')
mean = np.round(np.mean(pixels_G_F),decimals=3)
plt.text(0.95, 0.95, f'Mean =  {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_G_F),decimals=3)
plt.text(0.95, 0.90, f'STD =  {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
plt.show()

plt.figure()
values, counts = np.unique(pixels_S_F, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title('SIEMENS_FG Histogram')
mean = np.round(np.mean(pixels_S_F),decimals=3)
plt.text(0.95, 0.95, f'Mean =  {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_S_F),decimals=3)
plt.text(0.95, 0.90, f'STD = {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
plt.show()

                     ### Plotting bACKGROUND Histrograms #### 
plt.figure()
values, counts = np.unique(pixels_P_B, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title(' Philips_BG Histogram ')
mean = np.round(np.mean(pixels_P_B),decimals=3)
plt.text(0.95, 0.95, f'Mean = {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_P_B),decimals=3)
plt.text(0.95, 0.90, f'STD =  {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
plt.show()

plt.figure()
values, counts = np.unique(pixels_G_B, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title('GE_BG Histogram')
mean = np.round(np.mean(pixels_G_B),decimals=3)
plt.text(0.95, 0.95, f'Mean =  {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_G_B),decimals=3)
plt.text(0.95, 0.90, f'STD = {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
plt.show()

plt.figure()
values, counts = np.unique(pixels_S_B, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title('SIEMENS_BG Histogram')
mean = np.round(np.mean(pixels_S_B),decimals=3)
plt.text(0.95, 0.95, f'Mean =  {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_S_B),decimals=3)
plt.text(0.95, 0.90, f'STD =  {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
plt.show()

                 ### Plotting All Intensities Histrograms #### 
plt.figure()
values, counts = np.unique(pixels_P, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title(' Philips_All_Pixels Histogram ')
mean = np.round(np.mean(pixels_P),decimals=3)
plt.text(0.95, 0.95, f'Mean = {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_P),decimals=3)
plt.text(0.95, 0.90, f'STD =  {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
plt.show()

plt.figure()
values, counts = np.unique(pixels_G, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title('GE_All_Pixels Histogram')
mean = np.round(np.mean(pixels_G),decimals=3)
plt.text(0.95, 0.95, f'Mean =  {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_G),decimals=3)
plt.text(0.95, 0.90, f'STD =  {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
plt.show()

plt.figure()
values, counts = np.unique(pixels_S, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title('SIEMENS_All_Pixels Histogram')
mean = np.round(np.mean(pixels_S),decimals=3)
plt.text(0.95, 0.95, f'Mean =  {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_S),decimals=3)
plt.text(0.95, 0.90, f'STD =  {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
plt.show()'''



all_1 = []

all_1.append(pixels_S)
all_1.append(pixels_P)
all_1.append(pixels_G)

all_1 = np.concatenate(all_1)

plt.figure()
values, counts = np.unique(all_1, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title('SIEMENS_All_Pixels Histogram')
mean = np.round(np.mean(all_1),decimals=3)
plt.text(0.95, 0.95, f'Mean =  {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(all_1),decimals=3)
plt.text(0.95, 0.90, f'STD =  {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
plt.show()

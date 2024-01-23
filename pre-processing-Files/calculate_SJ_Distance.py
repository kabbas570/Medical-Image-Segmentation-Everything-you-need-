from scipy.spatial.distance import jensenshannon
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import glob

DIM_ = 256

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

def normalize_1(img):
 return (img - 125.667 ) / 208.77

def cal_SJ(img_paths):
    summ_all = []
    for i in range(10):
        image1 =  sitk.GetArrayFromImage(sitk.ReadImage(img_paths[i]))
        image1 = normalize_1(image1)
        hist1, _ = np.histogram(image1, bins=256, range=(0,600), density=True)

        for j in range(10):  
            if j!=i:
                image2 =  sitk.GetArrayFromImage(sitk.ReadImage(img_paths[j]))
                image2 = normalize_1(image2)
                hist2, _ = np.histogram(image2, bins=256, range=(0,600), density=True)
                js_distance = jensenshannon(hist1, hist2)
                summ_all.append(js_distance)
    
    return np.mean(summ_all)
            
            
if __name__ == "__main__":
    path_to_imgs = r'C:\My_Data\M2M Data\data\data_2\three_vendor\all\LA\imgs/'
    img_paths = sorted(glob.glob(path_to_imgs + '*.nii.gz'))

    avg_js_dist_1= cal_SJ(img_paths)

    print('Average JS Distance =', avg_js_dist_1)

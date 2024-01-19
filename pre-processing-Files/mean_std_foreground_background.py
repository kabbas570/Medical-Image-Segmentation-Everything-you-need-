import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import glob
import pandas as pd




df  = pd.read_csv(r"C:\My_Data\M2M Data\data\train.csv")
vendors = df['VENDOR']
counts = vendors.value_counts()


pixels_G_F = []  ## Accumulate all foreground-pixels
pixels_S_F = []
pixels_P_F = []

pixels_G_B = []  ## Accumulate all Bakground-pixels
pixels_S_B = []
pixels_P_B = []

def get_pixel_intensities(img_paths):
    for i in range(320):
        image = sitk.GetArrayFromImage(sitk.ReadImage(img_paths[i]))        
        image1 = image.copy()
        image2 = image.copy()
        name = img_paths[i][-16:]
        number1 = int(name[:-13])
        ind = df['SUBJECT_CODE'].loc[lambda x: x==number1].index[0]
        vendor = vendors[ind]
        

        gt_path  = img_paths[i][:-21] + 'gts/' + name[:-7]+ '_gt.nii.gz'
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)) 
        
        image1[np.where(gt==0)] = 0 # extract foreground 
        image2[np.where(gt!=0)] = 0 # extract foreground 
        if vendor=='GE MEDICAL SYSTEMS': 
             plt.imsave(r'C:\My_Data\M2M Data\data\data_2\three_vendor\all\G/' + name + '.png',image1[0,:])
             plt.imsave(r'C:\My_Data\M2M Data\data\data_2\three_vendor\all\G/' + name + '_bg.png',image2[0,:])
             pixels_G_F.append(image1.flatten())
             pixels_G_B.append(image2.flatten())
        if vendor=='SIEMENS':
             plt.imsave(r'C:\My_Data\M2M Data\data\data_2\three_vendor\all\S/' + name + '.png',image1[0,:])
             plt.imsave(r'C:\My_Data\M2M Data\data\data_2\three_vendor\all\S/' + name + '_bg.png',image2[0,:])
             pixels_S_F.append(image.flatten())
             pixels_S_B.append(image2.flatten())
        if vendor=='Philips Medical Systems':
             plt.imsave(r'C:\My_Data\M2M Data\data\data_2\three_vendor\all\P/' + name + '.png',image1[0,:])
             plt.imsave(r'C:\My_Data\M2M Data\data\data_2\three_vendor\all\P/' + name + '_bg.png',image2[0,:])
             pixels_P_F.append(image.flatten())
             pixels_P_B.append(image2.flatten())
        
        
if __name__ == "__main__":
    path_to_imgs = r"C:\My_Data\M2M Data\data\data_2\three_vendor\all\LA\imgs/"
    img_paths = sorted(glob.glob(path_to_imgs + '*.nii.gz'))

    aall_inten = get_pixel_intensities(img_paths)
    #plot_histrogram(aall_inten)



print('Philips Medical Systems')

print('P_Mean_Foreground   -->',np.mean(np.concatenate(pixels_P_F)))
print('P_Mean_Bckground  -->',np.mean(np.concatenate(pixels_P_B)))

print('P_STD_Foreground   -->',np.std(np.concatenate(pixels_P_F)))
print('P_STD_Bckground    -->',np.std(np.concatenate(pixels_P_B)))


print( 'GE MEDICAL SYSTEMS')

print('G_Mean_Foreground    -->',np.mean(np.concatenate(pixels_G_F)))
print('G_Mean_Bckground    -->',np.mean(np.concatenate(pixels_G_B)))

print('G_STD_Foreground    -->',np.std(np.concatenate(pixels_G_F)))
print('G_STD_Bckground    -->',np.std(np.concatenate(pixels_G_B)))



print('SIEMENS')
print('S_Mean_Foreground    -->',np.mean(np.concatenate(pixels_S_F)))
print('S_Mean_Bckground    -->',np.mean(np.concatenate(pixels_S_B)))

print('S_STD_Foreground    -->',np.std(np.concatenate(pixels_S_F)))

print('S_STD_Bckground   -->',np.std(np.concatenate(pixels_S_B)))

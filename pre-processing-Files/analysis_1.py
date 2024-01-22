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

pixels_G =[]  ## All pixels 
pixels_S =[]
pixels_P =[]


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
             pixels_G.append(image.flatten())
        if vendor=='SIEMENS':
             plt.imsave(r'C:\My_Data\M2M Data\data\data_2\three_vendor\all\S/' + name + '.png',image1[0,:])
             plt.imsave(r'C:\My_Data\M2M Data\data\data_2\three_vendor\all\S/' + name + '_bg.png',image2[0,:])
             pixels_S_F.append(image1.flatten())
             pixels_S_B.append(image2.flatten())
             pixels_S.append(image.flatten())
        if vendor=='Philips Medical Systems':
             plt.imsave(r'C:\My_Data\M2M Data\data\data_2\three_vendor\all\P/' + name + '.png',image1[0,:])
             plt.imsave(r'C:\My_Data\M2M Data\data\data_2\three_vendor\all\P/' + name + '_bg.png',image2[0,:])
             pixels_P_F.append(image1.flatten())
             pixels_P_B.append(image2.flatten())
             pixels_P.append(image.flatten())
        
        
if __name__ == "__main__":
    path_to_imgs = r"C:\My_Data\M2M Data\data\data_2\three_vendor\all\LA\imgs/"
    img_paths = sorted(glob.glob(path_to_imgs + '*.nii.gz'))

    aall_inten = get_pixel_intensities(img_paths)
    #plot_histrogram(aall_inten)

pixels_P_F = np.concatenate(pixels_P_F)
pixels_P_F = pixels_P_F[pixels_P_F != 0]
pixels_P_B = np.concatenate(pixels_P_B)
pixels_P_B = pixels_P_B[pixels_P_B != 0]
pixels_P = np.concatenate(pixels_P)


pixels_G_F = np.concatenate(pixels_G_F)
pixels_G_F = pixels_G_F[pixels_G_F != 0]
pixels_G_B = np.concatenate(pixels_G_B)
pixels_G_B = pixels_G_B[pixels_G_B != 0]
pixels_G = np.concatenate(pixels_G)


pixels_S_F = np.concatenate(pixels_S_F)
pixels_S_F = pixels_S_F[pixels_S_F != 0]
pixels_S_B = np.concatenate(pixels_S_B)
pixels_S_B = pixels_S_B[pixels_S_B != 0]
pixels_S = np.concatenate(pixels_S)



print('Philips Medical Systems')
print('P_Mean_Foreground   -->',np.mean(pixels_P_F))
print('P_Mean_Bckground  -->',np.mean(pixels_P_B))
print('P_Mean_All   -->',np.mean(pixels_P))
print('P_STD_Foreground   -->',np.std(pixels_P_F))
print('P_STD_Bckground    -->',np.std((pixels_P_B)))
print('P_STD_All    -->',np.std((pixels_P)))


print( 'GE MEDICAL SYSTEMS')
print('G_Mean_Foreground    -->',np.mean(pixels_G_F))
print('G_Mean_Bckground    -->',np.mean(pixels_G_B))
print('S_Mean_All    -->',np.mean(pixels_G))
print('G_STD_Foreground    -->',np.std(pixels_G_F))
print('G_STD_Bckground    -->',np.std(pixels_G_B))
print('S_STD_All    -->',np.std((pixels_G)))


print('SIEMENS')
print('S_Mean_Foreground    -->',np.mean(pixels_S_F))
print('S_Mean_Bckground    -->',np.mean(pixels_S_B))
print('G_Mean_All    -->',np.mean(pixels_S))
print('S_STD_Foreground    -->',np.std(pixels_S_F))
print('S_STD_Bckground   -->',np.std(pixels_S_B))
print('G_STD_All    -->',np.std((pixels_S)))


                         
            
                    ### Plotting ForeGround Histrograms #### 
values, counts = np.unique(pixels_P_F, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title(' Philips_FG Histogram ')
plt.show()
mean = np.round(np.mean(pixels_P_F),decimals=3)
plt.text(0.95, 0.95, f'Mean = : {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_P_F),decimals=3)
plt.text(0.95, 0.90, f'STD = : {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)

values, counts = np.unique(pixels_G_F, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title('GE_FG Histogram')
plt.show()
mean = np.round(np.mean(pixels_G_F),decimals=3)
plt.text(0.95, 0.95, f'Mean = : {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_G_F),decimals=3)
plt.text(0.95, 0.90, f'STD = : {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)


values, counts = np.unique(pixels_S_F, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title('SIEMENS_FG Histogram')
plt.show()
mean = np.round(np.mean(pixels_S_F),decimals=3)
plt.text(0.95, 0.95, f'Mean = : {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_S_F),decimals=3)
plt.text(0.95, 0.90, f'STD = : {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)



                     ### Plotting ForeGround Histrograms #### 

values, counts = np.unique(pixels_P_B, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title(' Philips_BG Histogram ')
plt.show()
mean = np.round(np.mean(pixels_P_B),decimals=3)
plt.text(0.95, 0.95, f'Mean = : {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_P_B),decimals=3)
plt.text(0.95, 0.90, f'STD = : {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)



values, counts = np.unique(pixels_G_B, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title('GE_BG Histogram')
plt.show()
mean = np.round(np.mean(pixels_G_B),decimals=3)
plt.text(0.95, 0.95, f'Mean = : {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_G_B),decimals=3)
plt.text(0.95, 0.90, f'STD = : {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)



values, counts = np.unique(pixels_S_B, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title('SIEMENS_BG Histogram')
plt.show()
mean = np.round(np.mean(pixels_S_B),decimals=3)
plt.text(0.95, 0.95, f'Mean = : {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_S_B),decimals=3)
plt.text(0.95, 0.90, f'STD = : {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)


                 ### Plotting All Intensities Histrograms #### 

values, counts = np.unique(pixels_P, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title(' Philips_All_Pixels Histogram ')
plt.show()
mean = np.round(np.mean(pixels_P),decimals=3)
plt.text(0.95, 0.95, f'Mean = : {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_P),decimals=3)
plt.text(0.95, 0.90, f'STD = : {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)


values, counts = np.unique(pixels_G, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title('GE_All_Pixels Histogram')
plt.show()
mean = np.round(np.mean(pixels_G),decimals=3)
plt.text(0.95, 0.95, f'Mean = : {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_G),decimals=3)
plt.text(0.95, 0.90, f'STD = : {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)


values, counts = np.unique(pixels_S, return_counts=True)
plt.bar(values, counts, width=1.0, color='blue', edgecolor='black')
plt.xlabel('Unique Values')
plt.ylabel('Frequency')
plt.title('SIEMENS_All_Pixels Histogram')
plt.show()
mean = np.round(np.mean(pixels_S),decimals=3)
plt.text(0.95, 0.95, f'Mean = : {mean}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)
std = np.round(np.std(pixels_S),decimals=3)
plt.text(0.95, 0.90, f'STD = : {std}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=12)

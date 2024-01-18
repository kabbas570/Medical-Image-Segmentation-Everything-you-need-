import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import glob
import pandas as pd

def Normalization_1(img):
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

def Normalization_All(img):
 return (img - 112.5312 ) / 143.1712

def Normalization_MinMax(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


df  = pd.read_csv(r"C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\test\test.csv")
vendors = df['VENDOR']

def get_pixel_intensities(img_paths):
    all_pixels = []
    for i in range(320):
        image1 = sitk.GetArrayFromImage(sitk.ReadImage(img_paths[i]))
        image1=Normalization_MinMax(image1)
        
        # number1 = img_paths[i][-16:]
        # number1 = int(number1[:-13])
        # ind = df['SUBJECT_CODE'].loc[lambda x: x==number1].index[0]
        # vendor = vendors[ind]
        # if vendor=='GE MEDICAL SYSTEMS': 
        #     image1 = normalize_G(image1)
        # if vendor=='SIEMENS':
        #     image1 = normalize_S(image1)
        # if vendor=='Philips Medical Systems':
        #     image1 = normalize_P(image1)

        all_pixels.append(image1.flatten())
    return np.concatenate(all_pixels).flatten()
        

def plot_histrogram(pixels_intensities):
    

        common_range = np.min(pixels_intensities), np.max(pixels_intensities)

        bin1 = int(np.sqrt(len(pixels_intensities)))


        hist1, bin_centers1 = np.histogram(pixels_intensities,bins=bin1,range=common_range, density=True)
        
        #plt.plot(bin_centers1[:-1], hist1)
        plt.plot(hist1)
        plt.title('Histogram of Pixel Intensities')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Normalized Frequency')
        plt.show()
        
        # #pixels_intensities = pixels_intensities.mean()
        
        # counts, bins = np.histogram(pixels_intensities, int(np.max(pixels_intensities)))
        # # plot histogram centered on values 0..255
        # plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
        # plt.xlim()
        # plt.show()
        
        # b, bins, patches = plt.hist(pixels_intensities,  int(np.max(pixels_intensities)),density=True)
        # plt.xlim()
        # plt.show()

        
        
        ## plot histrogram here
        
if __name__ == "__main__":
    path_to_imgs = r"C:\My_Data\M2M Data\data\data_2\five_fold\sep\LA_Data\test\imgs/"
    img_paths = sorted(glob.glob(path_to_imgs + '*.nii.gz'))

    aall_inten = get_pixel_intensities(img_paths)
    plot_histrogram(aall_inten)


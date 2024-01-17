import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import SimpleITK as sitk
import dit
from dit.divergences import jensen_shannon_divergence
import glob

def compute_intensity_histogram(image):
    hist, edges = np.histogram(image.flatten(), bins=256, range=[0, 256], density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    return hist, bin_centers

def js_distance(hist1, hist2):
    P = hist1 / np.sum(hist1)
    Q = hist2 / np.sum(hist2)
    M = 0.5 * (P + Q)
    kl_divergence_PM = entropy(P, M)
    kl_divergence_QM = entropy(Q, M)
    js_distance = np.sqrt( (kl_divergence_PM + kl_divergence_QM))
    return js_distance

def visualize_histogram(image, title):
    hist, bin_centers = compute_intensity_histogram(image)
    plt.plot(bin_centers, hist, label='Image Histogram')
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.show()

def convert_to_distribution(hist, bin_centers):
    outcomes = [int(b) for b in bin_centers]
    pmf = np.asarray(hist)
    dist = dit.ScalarDistribution(outcomes, pmf)
    return dist

def calculate_average_js_distances(img_paths):
    sum_all_1 = []
    sum_all_2 = []

    for i in range(320):
        image1 = sitk.GetArrayFromImage(sitk.ReadImage(img_paths[i]))

        for j in range(320):
            if j!=i:
                image2 = sitk.GetArrayFromImage(sitk.ReadImage(img_paths[j]))
    
                hist1, bin_centers1 = compute_intensity_histogram(image1)
                hist2, bin_centers2 = compute_intensity_histogram(image2)
    
                js_dist1 = js_distance(hist1, hist2)
                sum_all_1.append(js_dist1)
    
                dist1 = convert_to_distribution(hist1, bin_centers1)
                dist2 = convert_to_distribution(hist2, bin_centers2)
    
                js_dist2 = np.sqrt(jensen_shannon_divergence([dist1, dist2]))
                sum_all_2.append(js_dist2)

    return np.mean(sum_all_1), np.mean(sum_all_2)

if __name__ == "__main__":
    path_to_imgs = '/data/scratch/acw676/MNM2/five_folds/test/imgs/'
    img_paths = sorted(glob.glob(path_to_imgs + '*.nii.gz'))

    avg_js_dist_1, avg_js_dist_2 = calculate_average_js_distances(img_paths)

    print('Average JS Distance (Direct Calculation):', avg_js_dist_1)
    print('Average JS Distance (dit Library):', avg_js_dist_2)

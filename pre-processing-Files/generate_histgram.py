import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_gaussian_image(mean, std, size=(256, 256), seed=None):
    np.random.seed(seed)  # Set seed for reproducibility
    image_data = np.random.normal(mean, std, size)
    #image_data = np.clip(image_data, 0, 255)
    image_data = image_data.astype(np.uint8)
    
    plt.subplot(1, 2, 1)
    plt.imshow(image_data, cmap='gray')
    plt.title(f'Generated Image\n(Mean={mean}, Std={std}, Seed={seed})')

    plt.subplot(1, 2, 2)
    plt.hist(image_data.flatten(), bins=50, density=True, color='b', alpha=0.7, label='Histogram')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std)
    plt.plot(x, p, 'k', linewidth=2, label='Expected Gaussian Distribution')

    plt.title('Intensity Histogram and Expected Gaussian Distribution')
    #plt.legend()

    plt.show()

# Example usage with seed for reproducibility:
generate_gaussian_image(mean=117.1235, std=137.8422, seed=42)
generate_gaussian_image(mean=197.8624, std=284.7200, seed=42)
generate_gaussian_image(mean=72.8425, std=91.4957, seed=42)

generate_gaussian_image(mean=112.5312, std=143.1712, seed=42)


np.random.seed(42) 
image_data = np.random.normal(112.5312, 143.1712, (16,16))


def normalize_1(img):
 return (img - 112.5312 ) / 143.1712

n= normalize_1(image_data)

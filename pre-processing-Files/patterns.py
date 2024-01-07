import numpy as np
import matplotlib.pyplot as plt

def normalize_P(img):
 return (img - 72.8425) / 91.4957

def normalize_G(img):
    return (img - 197.8624) / 284.7200

def normalize_S(img):
    return (img - 117.1235) / 137.8422


# Dimensions of the extra channel
width = 64
height = 64

# Function to generate a unique harmonic pattern for each scanner
def generate_harmonic_pattern1(mean, std, scanner):
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x**2 + y**2)
    
    if scanner == 1:
        z = mean + std * (np.sin(5 * r * np.pi) ** 2)
    elif scanner == 2:
        z = mean + std * (np.cos(5 * r * np.pi) ** 2)
    elif scanner == 3:
        z = mean + std * np.sin(x+y)
    
    return z

def generate_harmonic_pattern2(mean, std, scanner):
    x = np.linspace(0, 4*np.pi, width)
    y = np.linspace(0, 4*np.pi, height)
    x, y = np.meshgrid(x, y)
    
    if scanner == 1:
        z = mean + std * np.sin(2*x) * np.sin(2*y)
    elif scanner == 2:
        z = mean + std * np.sin(x) * np.cos(y)
    elif scanner == 3:
        z = mean + std * np.sin(x+y)
    
    return z


# Scanner-specific mean and standard deviation values
scanner_P_mean = 72.8425
scanner_P_std = 91.4957

scanner_G_mean = 197.8624
scanner_G_std = 284.7200

scanner_S_mean = 117.1235
scanner_S_std = 137.8422

# Generate harmonic patterns for each scanner
harmonic_pattern_scanner_P = generate_harmonic_pattern1(scanner_P_mean, scanner_P_std, 1)
harmonic_pattern_scanner_G = generate_harmonic_pattern2(scanner_G_mean, scanner_G_std, 2)
harmonic_pattern_scanner_S = generate_harmonic_pattern2(scanner_S_mean, scanner_S_std, 3)

# Visualize harmonic patterns for each scanner
plt.figure(figsize=(10, 4))

plt.subplot(131)
plt.title('Scanner P Harmonic Pattern')
plt.imshow(harmonic_pattern_scanner_P, cmap='viridis')
plt.colorbar()

plt.subplot(132)
plt.title('Scanner G Harmonic Pattern')
plt.imshow(harmonic_pattern_scanner_G, cmap='viridis')
plt.colorbar()

plt.subplot(133)
plt.title('Scanner S Harmonic Pattern')
plt.imshow(harmonic_pattern_scanner_S, cmap='viridis')
plt.colorbar()

plt.tight_layout()
plt.show()


p = normalize_P(harmonic_pattern_scanner_P)
g = normalize_P(harmonic_pattern_scanner_G)
s = normalize_P(harmonic_pattern_scanner_S)

print(np.mean(p))
print(np.std(p))







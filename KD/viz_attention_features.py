import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

def show_features_custom_grid(features):

    B, N , H, W = features.shape
    num_cols = int(np.ceil(np.sqrt(N)))  # Set columns based on square root of N
    num_rows = int(np.ceil(N / num_cols))  # Compute the required number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(N):
        ax = axes[i]
        ax.imshow(features[0,i, :, :],)
        ax.set_title(f'Feature {i+1}')
        ax.axis('off')

    for j in range(N, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
features = sitk.ReadImage(r"C:\My_Data\DIST\Features\001_LA_ED_n1.nii.gz")    ## --> [H,W,C]
features_org = sitk.GetArrayFromImage(features)   ## --> [C,H,W]
features_org = np.transpose(features_org, (3, 2, 0, 1))
show_features_custom_grid(features_org)

class LearnedChannelSelfAttentionPooling_V3(nn.Module):
    def __init__(self, teacher_dim, student_dim):
        super().__init__()

        if teacher_dim % student_dim != 0:
            raise ValueError("teacher_dim should be divisible by student_dim")
        
        self.chunk_size = teacher_dim // student_dim  
        self.num_chunks = teacher_dim//self.chunk_size
        self.student_dim = student_dim
        
        # Self-attention components for each chunk
        self.query = nn.ModuleList([nn.Conv2d(self.chunk_size,self.chunk_size, kernel_size=1) for _ in range(self.num_chunks)])
        self.key = nn.ModuleList([nn.Conv2d(self.chunk_size, self.chunk_size, kernel_size=1) for _ in range(self.num_chunks)])
        self.value = nn.ModuleList([nn.Conv2d(self.chunk_size, self.chunk_size, kernel_size=1) for _ in range(self.num_chunks)])

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D input")
        
        batch_size, channels, height, width = x.shape
        assert channels == self.chunk_size * self.student_dim, "Mismatch between input channels and expected dimensions"
        
        chunks = torch.split(x, split_size_or_sections=self.chunk_size, dim=1)
        all_chunks = torch.stack(chunks, dim=1) ## [B,num_chunks,chunk_size,H,W]

        attended_chunks = []
        for i, chunk in enumerate(chunks):
            # Apply self-attention to each chunk separately
            q = self.query[i](chunk).view(batch_size, -1, height * width)
            k = self.key[i](chunk).view(batch_size, -1, height * width)
            v = self.value[i](chunk).view(batch_size, -1, height * width)
            
            attn = (q @ k.transpose(-2, -1))
            attn = attn/math.sqrt(height*width)
            attn = F.softmax(attn, dim=-1)
            attn = (attn @ v)
            attn = attn.view(batch_size, -1, height, width)
            attended_chunks.append(attn)
        attended_x  = torch.stack(attended_chunks, dim=1) ## [B,num_chunks,chunk_size,H,W]
        pooled = torch.sum(all_chunks * attended_x, dim=2)
        return pooled
    

features_org = torch.tensor(features_org)
learned_pooling = LearnedChannelSelfAttentionPooling_V3(features_org.shape[1], features_org.shape[1]//4)
pooled_l = learned_pooling(features_org)
pooled_l = pooled_l.detach().numpy()
show_features_custom_grid(pooled_l)




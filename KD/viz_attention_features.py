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
features = sitk.ReadImage(r"C:\My_Data\DIST\Features\001_LA_ED_f3.nii.gz")    ## --> [H,W,C]
features_org = sitk.GetArrayFromImage(features)   ## --> [C,H,W]
features_org = np.transpose(features_org, (3, 2, 0, 1))
show_features_custom_grid(features_org)

class LearnedChannelSelfAttentionPooling_V4(nn.Module):
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
        
        self.query_G = nn.Conv2d(teacher_dim, teacher_dim, kernel_size=1)
        self.key_G = nn.Conv2d(teacher_dim, teacher_dim, kernel_size=1)
        self.value_G = nn.Conv2d(teacher_dim, teacher_dim, kernel_size=1)

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
            
            attn = (q.transpose(-2, -1) @ k)
            attn = attn/math.sqrt(height*width)
            attn = F.softmax(attn, dim=-1)
            attn = (v @ attn )
            attn = attn.view(batch_size, -1, height, width)
            attended_chunks.append(attn)
            
            # attn = (q @ k.transpose(-2, -1))
            # attn = attn/math.sqrt(height*width)
            # attn = F.softmax(attn, dim=-1)
            # attn = (attn @ v)
            # attn = attn.view(batch_size, -1, height, width)
            # attended_chunks.append(attn)
            
        attended_x  = torch.stack(attended_chunks, dim=1) ## [B,num_chunks,chunk_size,H,W]

        ### Global Attention ###
        
        q = self.query_G(x).view(batch_size, -1, height * width)
        k = self.key_G(x).view(batch_size, -1, height * width)
        v = self.value_G(x).view(batch_size, -1, height * width)
        
        
        # attn_G = (q @ k.transpose(-2, -1))
        # attn_G = attn_G/math.sqrt(height*width)
        # attn_G = F.softmax(attn_G, dim=-1)
        # attn_G = (attn_G @ v)
        # attn_G = attn_G.view(batch_size, -1, height, width)
        # attn_G = attn_G.view(batch_size, -1, self.chunk_size, height, width)
            
        
        attn_G = (q.transpose(-2, -1) @ k)
        attn_G = attn_G/math.sqrt(height*width)
        attn_G = F.softmax(attn_G, dim=-1)
        attn_G = (v @ attn_G )
        attn_G = attn_G.view(batch_size, -1, height, width)
        attn_G = attn_G.view(batch_size, -1, self.chunk_size, height, width)
        
        combined = attended_x + attn_G
        pooled = torch.sum(all_chunks * combined, dim=2)
        
        
        print(attn_G.shape)
        print(attended_x.shape)

        attn_G = attn_G.detach().numpy()
        attn_G = attn_G.reshape(1, 18 * 4, 64, 64)
        show_features_custom_grid(attn_G)
        
        attended_x = attended_x.detach().numpy()
        attended_x = attended_x.reshape(1, 18 * 4, 64, 64)
        show_features_custom_grid(attended_x)
        
        
        combined = combined.detach().numpy()
        combined = combined.reshape(1, 18 * 4, 64, 64)
        show_features_custom_grid(combined)
        
        #assert torch.allclose(attn_G, attended_x, atol=2)
        return pooled
    

class LearnedChannelSelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        if input_dim % output_dim != 0:
            raise ValueError("input_dim should be divisible by output_dim")
        
        self.factor = input_dim // output_dim
        
        # Self-attention components
        self.query = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.key = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.value = nn.Conv2d(input_dim, input_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, input_dim, H, W] - input features
        
        batch_size, channels, height, width = x.shape
        assert channels == self.factor * (x.shape[1] // self.factor), "Mismatch between input channels and expected dimensions"

        # Reshape the input tensor to group channels into chunks of size factor
        
        
        x_reshaped = x.view(batch_size, -1, self.factor, height, width)  # [B, output_dim, factor, H, W]
        
        # Self-attention mechanism
        query = self.query(x).view(batch_size, -1, height * width).transpose(1, 2)  # [B, H*W, output_dim]
        key = self.key(x).view(batch_size, -1, height * width)  # [B, output_dim, H*W]
        value = self.value(x).view(batch_size, -1, height * width)  # [B, output_dim, H*W]

        # Compute attention scores (Q * K^T)
        attention_scores = torch.bmm(query, key)  # [B, H*W, H*W]
        attention_weights = torch.softmax(attention_scores, dim=-1)  # [B, H*W, H*W]

        # Apply the attention weights to the values (V)
        weighted_values = torch.bmm(value, attention_weights.transpose(1, 2))  # [B, output_dim, H*W]
        weighted_values = weighted_values.view(batch_size, -1, height, width)  # [B, output_dim, H, W]
        #weighted_values = weighted_values.unsqueeze(2).expand(-1, -1, 4, -1, -1)
        weighted_values = weighted_values.view(batch_size, -1, self.factor, height, width)
        
        # Pool the tensor along the factor dimension
        pooled = torch.sum(x_reshaped * weighted_values, dim=2)  # Sum over the factor dimension
        
        attended_x = weighted_values.detach().numpy()
        attended_x = attended_x.reshape(1, 18 * 4, 64, 64)
        show_features_custom_grid(attended_x)

        return pooled
    
    
    
features_org = torch.tensor(features_org)
learned_pooling = LearnedChannelSelfAttentionPooling_V4(features_org.shape[1], features_org.shape[1]//4)
pooled_l = learned_pooling(features_org)
pooled_l = pooled_l.detach().numpy()
show_features_custom_grid(pooled_l)




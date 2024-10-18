import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LearnedChannelSelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        if input_dim % output_dim != 0:
            raise ValueError("input_dim should be divisible by output_dim")
        
        self.factor = input_dim // output_dim
        
        # Self-attention components
        self.query = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.key = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.value = nn.Conv2d(input_dim, output_dim, kernel_size=1)

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
        weighted_values = weighted_values.unsqueeze(2).expand(-1, -1, 4, -1, -1)
        
        # Pool the tensor along the factor dimension
        pooled = torch.sum(x_reshaped * weighted_values, dim=2)  # Sum over the factor dimension

        return pooled
        
class LearnedChannelSelfAttentionPooling(nn.Module):
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
            q = self.query[i](chunk).view(batch_size, -1, height * width).transpose(1, 2)
            k = self.key[i](chunk).view(batch_size, -1, height * width)
            v = self.value[i](chunk).view(batch_size, -1, height * width)

            attention_scores = torch.bmm(q, k) / math.sqrt(self.student_dim)
            attention_weights = F.softmax(attention_scores, dim=-1)
            attended_chunk = torch.bmm(attention_weights, v.transpose(1, 2))
            attended_chunk = attended_chunk.transpose(1, 2).view(batch_size, -1, height, width)
            attended_chunks.append(attended_chunk)

        # Combine the attended chunks
        attended_x = torch.cat(attended_chunks, dim=1)
        # Reshape for the final pooling step
        attended_x = attended_x.view(batch_size, -1, self.chunk_size, height, width)
        # Pool the tensor along the factor dimension
        pooled = torch.sum(all_chunks * attended_x, dim=2)
        
        return pooled
    
Teacher_Dim = 24
Student_Dim = 6
teacher_tensor = torch.randn(2, Teacher_Dim, 16, 16)

learned_pooling = LearnedChannelSelfAttentionPooling(Teacher_Dim, Student_Dim)
pooled_l = learned_pooling(teacher_tensor)
print(pooled_l.shape)

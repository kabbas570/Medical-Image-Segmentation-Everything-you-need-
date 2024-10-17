import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        
class LearnedChannelSelfAttentionPooling_V2(nn.Module):
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
            
        attended_x  = torch.stack(attended_chunks, dim=1) ## [B,num_chunks,chunk_size,H,W]
        pooled = torch.sum(all_chunks * attended_x, dim=2)
        
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

learned_pooling = LearnedChannelSelfAttentionPooling_V2(Teacher_Dim, Student_Dim)
pooled_l = learned_pooling(teacher_tensor)
print(pooled_l.shape)

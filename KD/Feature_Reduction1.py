import torch
import torch.nn as nn
def pool_spatial(tensor, pool_type='avg'):
    if pool_type == 'max':
        pooled = torch.max(tensor, dim=1, keepdim=True)[0] ## or Change to Avg
    elif pool_type == 'avg':
        pooled = torch.mean(tensor, dim=1, keepdim=True)
    return pooled

def reducing_1(teacher_tensor,Teacher_Dim,Student_Dim):
    Factor = Teacher_Dim//Student_Dim
    pooled_tensors = []
    splitted_tensors = torch.split(teacher_tensor, split_size_or_sections=Factor, dim=1)
    for t in splitted_tensors:
        pooled_tensors.append(pool_spatial(t))

    return torch.cat(pooled_tensors, dim=1)

# make a learnable pooling layer that reduces channels to match the output dim
class LearnedChannelAveragePooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        if input_dim % output_dim != 0:
            raise ValueError("input_dim should be divisible by output_dim")
        
        self.factor = input_dim // output_dim
        self.weighting = nn.Parameter(torch.ones(output_dim, self.factor)/self.factor, requires_grad=True).float()

    def forward(self, x):
        # split the input tensor into chunks of size factor
        splitted_tensors = torch.split(x, split_size_or_sections=self.factor, dim=1)

        # get weightings for each chunk
        weighting_softmax = torch.softmax(self.weighting, dim=1)

        # apply the weighted sum to each chunk
        pooled_tensors = []
        for i, t in enumerate(splitted_tensors):
            pooled_tensors.append(torch.sum(t * weighting_softmax[i, :].view(1, -1, 1, 1), dim=1, keepdim=True))

        return torch.cat(pooled_tensors, dim=1)

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

        ## x_reshaped --> entire attention
        ### x_reshaped = [B,16,4,16,16]
        # x_reshaped1 = [B,for(1-16),4,16,16] --> apply attention
        
        ## [B,16,0,16,16] --> [B,16,16*16,4] 
        ## [B,16,H, W]
        
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
    
Teacher_Dim = 48
Student_Dim = 12

teacher_tensor = torch.randn(2, Teacher_Dim, 16, 16)
pooled_t = reducing_1(teacher_tensor,Teacher_Dim,Student_Dim)
print(pooled_t.shape)

learned_pooling = LearnedChannelAveragePooling(Teacher_Dim, Student_Dim)
pooled_l = learned_pooling(teacher_tensor)
print(pooled_l.shape)

learned_pooling = LearnedChannelSelfAttentionPooling(Teacher_Dim, Student_Dim)
pooled_l = learned_pooling(teacher_tensor)
print(pooled_l.shape)

assert torch.allclose(pooled_t, pooled_l, atol=1e-6)

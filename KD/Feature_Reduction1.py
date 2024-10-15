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
    
Teacher_Dim = 48
Student_Dim = 12

teacher_tensor = torch.randn(2, Teacher_Dim, 16, 16)
pooled_t = reducing_1(teacher_tensor,Teacher_Dim,Student_Dim)
print(pooled_t.shape)

learned_pooling = LearnedChannelAveragePooling(Teacher_Dim, Student_Dim)
pooled_l = learned_pooling(teacher_tensor)
print(pooled_l.shape)

assert torch.allclose(pooled_t, pooled_l, atol=1e-6)

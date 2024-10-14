import torch

Teacher_Dim = 24
Student_Dim = 6
Factor = Teacher_Dim//Student_Dim
original_tensor = torch.randn(2, Teacher_Dim, 16, 16)
t1, t2 ,t3, t4, t5, t6 = torch.split(original_tensor, split_size_or_sections=Factor, dim=1)

# Print the shapes to verify
print("Original tensor shape:", original_tensor.shape)
print("t1 shape:", t1.shape)
print("t2 shape:", t2.shape)

# Function to pool along spatial dimensions
def pool_spatial(tensor):
    pooled = torch.max(tensor, dim=1, keepdim=True)[0] ## or Change to Avg
    return pooled

pooled_t = torch.cat([pool_spatial(t1),pool_spatial(t2),pool_spatial(t3),pool_spatial(t4),pool_spatial(t5),pool_spatial(t6)],1)


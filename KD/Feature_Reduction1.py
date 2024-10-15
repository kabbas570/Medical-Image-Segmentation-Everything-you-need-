import torch
def pool_spatial(tensor):
    pooled = torch.max(tensor, dim=1, keepdim=True)[0] ## or Change to Avg
    return pooled

def reducing_1(tensor,Teacher_Dim,Student_Dim):
    Factor = Teacher_Dim//Student_Dim
    pooled_tensors = []
    splitted_tensors = torch.split(teacher_tensor, split_size_or_sections=Factor, dim=1)
    for t in splitted_tensors:
        pooled_tensors.append(pool_spatial(t))

    return torch.cat(pooled_tensors, dim=1)

Teacher_Dim = 48
Student_Dim = 12

teacher_tensor = torch.randn(2, Teacher_Dim, 16, 16)
pooled_t = reducing_1(teacher_tensor,Teacher_Dim,Student_Dim)

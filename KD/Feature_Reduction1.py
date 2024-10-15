import torch


def pool_spatial(tensor):
    pooled = torch.max(tensor, dim=1, keepdim=True)[0] ## or Change to Avg
    return pooled

def reducing_1(tensor,Teacher_Dim,Student_Dim):
    Factor = Teacher_Dim//Student_Dim
    t1, t2 ,t3, t4, t5, t6 = torch.split(tensor, split_size_or_sections=Factor, dim=1)
    pooled_t = torch.cat([pool_spatial(t1),pool_spatial(t2),pool_spatial(t3),pool_spatial(t4),pool_spatial(t5),pool_spatial(t6)],1)
    return pooled_t

Teacher_Dim = 24
Student_Dim = 6

teacher_tensor = torch.randn(2, Teacher_Dim, 16, 16)
pooled_t = reducing_1(teacher_tensor,Teacher_Dim,Student_Dim)

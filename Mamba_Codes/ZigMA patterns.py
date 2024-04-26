import torch

def flip_Dim(x,dim):
    if dim == '12':
        x1 = x
    if dim == '14':
        x1 = x.permute(0, 1, 3, 2).contiguous()  
    if dim == '98':
        x1 = torch.flip(x,[2,3])
    if dim == '96':
        x1 = torch.flip(x,[2,3]).permute(0, 1, 3, 2).contiguous()  
    if dim == '78':
        x1 = torch.flip(x,[2])           
    if dim == '74':
        x1 = torch.flip(x,[2]).permute(0, 1, 3, 2).contiguous()  
    if dim == '32':
        x1 = torch.flip(x,[3])          
    if dim == '36':
        x1 = torch.flip(x,[3]).permute(0, 1, 3, 2).contiguous()
    return x1

def flip_Dim_back(x,dim):
    if dim == '12':
        x1 = x
    if dim == '14':
        x1 = x.permute(0, 1, 3, 2).contiguous()  
    if dim == '98':
        x1 = torch.flip(x,[2,3])
    if dim == '96':
        x1 = torch.flip(x,[2,3]).permute(0, 1, 3, 2).contiguous()  
    if dim == '78':
        x1 = torch.flip(x,[2])           
    if dim == '74':
        x1 = torch.flip(x,[3]).permute(0, 1, 3, 2).contiguous()  
    if dim == '32':
        x1 = torch.flip(x,[3])          
    if dim == '36':
        x1 = torch.flip(x,[2]).permute(0, 1, 3, 2).contiguous()
    return x1


inp = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(inp)
inp = torch.unsqueeze(inp, dim=0)
inp = torch.unsqueeze(inp, dim=0)

x1_98 = flip_Dim(inp,'78')

print(x1_98[0][0])

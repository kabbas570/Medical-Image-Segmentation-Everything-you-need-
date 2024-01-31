import torch 
x = torch.ones(4,3,2,2, requires_grad=True)

x = x*2

x[1,0,:] = 5
x[1,0,1,1] = -5
x[3,1,:] = 3
x[3,1,0,0] = -3


mean = torch.mean(x, dim=(2,3))
std = torch.std(x, dim=(2,3))


x_modulated= ((x.permute([2,3,0,1])*std)+mean).permute([2,3,0,1])


mean=mean.detach().numpy()
std=std.detach().numpy()

x_modulated=x_modulated.detach().numpy()

a1=x_modulated[3,:]


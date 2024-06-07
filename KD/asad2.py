import torch 
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.thetas = nn.Parameter(torch.ones(out_channels))
        

    def forward(self, x):
        
        x = self.double_conv(x)
        batch_size = x.size(0)
        thetas = self.thetas.view(1, -1).expand(batch_size, -1)
        thetas = thetas.unsqueeze(-1).unsqueeze(-1)

        soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, 10)
        print(soft_mask_variables)
        
        #print(thetas.shape)
        return  x*thetas
    
    
class UNet(nn.Module):
    def __init__(self, n_channels=1,):
        super(UNet, self).__init__()
        self.n_channels = n_channels

        self.inc = Conv(n_channels, 64)
 

    def forward(self, inp):
        x1 = self.inc(inp)

        return x1
        

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def model() -> UNet:
    model = UNet()
    model.to(device=DEVICE,dtype=torch.float)
    return model
from torchsummary import summary
model = model()
summary(model, [(1, 160,160)])
 
 
model = UNet()
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable parameters:", num_parameters)


import torch
import torch.nn.functional as F

# Initialize a tensor of kernels
kernels = torch.randn(1, 8, 1, 1)
print("Original Kernels:")
print(kernels)

# Apply Gumbel-Softmax to the kernels
a2 = F.gumbel_softmax(kernels, tau=1, hard=True, dim=1)
print("\nGumbel-Softmax Result:")
print(a2)

# Sum the elements in the resulting tensor
sum_a2 = torch.sum(a2)
print("\nSum of all elements in Gumbel-Softmax result:")
print(sum_a2)

